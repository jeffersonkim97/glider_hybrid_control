"""The Bellman solver and the Q-learning baseline must share one MDP.

These are equivalence tests, not behaviour tests.  They exist because the two
solvers reach the problem by different routes - the sweep walks motion-offset
slabs, the learner walks ``graph.adjacency`` - and nothing in the type system
stops those from drifting apart.  If they ever do, every comparison between the
two becomes meaningless, and it would look like an algorithmic result rather than
a bug.  So the successors, the costs, the terminal set and the evaluator are all
pinned against the exact solution itself.

The lattice is deliberately the coarse one: these assert correspondences, which
do not get truer at finer resolution, only slower to check.
"""

from __future__ import annotations

import unittest

import numpy as np

from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import (
    AttackerMDP, QLearningConfig, QTable, candidate_start_states, default_sensor,
    evaluate_trajectory, greedy_trajectory, solve_glide_mdp, switching_candidates,
    train,
)
from P1b_condition import ComputationCondition, build_scene


RESOLUTION_M = 100.0
SAMPLE_STATES = 400
TOLERANCE = 1.0e-9


class SharedMDPTests(unittest.TestCase):
    """One scene, one sensor, built once: these tests only read it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(
            ComputationCondition(spatial_resolution_m=RESOLUTION_M),
        )
        cls.sensor = default_sensor(cls.scene)
        cls.oracle = exact_best_response(cls.scene, cls.sensor, keep_solution=True)
        if not cls.oracle.feasible:
            raise unittest.SkipTest("no feasible exact best response at the seed sensor")
        cls.value = np.asarray(cls.oracle.solution.value, dtype=float)
        cls.goal_reachable = np.asarray(
            cls.oracle.solution.goal_reachable, dtype=bool,
        )
        cls.mdp = AttackerMDP(cls.scene, cls.sensor)

    def _sampled_states(self, *, terminal: bool) -> np.ndarray:
        terminal_mask = np.asarray(self.scene.graph.terminal_mask, dtype=bool)
        usable = np.flatnonzero(
            self.goal_reachable
            & np.isfinite(self.value)
            & (terminal_mask if terminal else ~terminal_mask)
        )
        if len(usable) <= SAMPLE_STATES:
            return usable
        rng = np.random.default_rng(0)
        return rng.choice(usable, size=SAMPLE_STATES, replace=False)

    def test_action_set_is_the_shared_successor_list(self) -> None:
        """The learner's action set is the shared adjacency, not a copy of it."""
        for state_id in self._sampled_states(terminal=False)[:50]:
            state_id = int(state_id)
            self.assertEqual(
                tuple(self.mdp.actions(state_id)),
                tuple(self.scene.graph.adjacency[state_id]),
                "the MDP must expose the shared adjacency, not a reconstruction",
            )

    def _optimality_shortfall(self, state_id: int) -> float:
        """min_a [ c(s,a) + V(s') ] - V(s), over the collision-checked actions."""
        edges = self.mdp.actions(state_id)
        if not len(edges):
            return 0.0
        candidates = self.mdp.cost_row(state_id) + np.asarray(
            [self.value[int(edge.target_id)] for edge in edges], dtype=float,
        )
        if not np.any(np.isfinite(candidates)):
            return 0.0
        best = float(np.min(np.where(np.isfinite(candidates), candidates, np.inf)))
        return best - float(self.value[state_id])

    def test_bellman_optimality_holds_under_the_learner_costs(self) -> None:
        """V(s) == min_a [ c(s,a) + V(s') ] with c taken from the MDP.

        This is the load-bearing test: it can only hold if the successor sets, the
        stage costs and the terminal convention all agree with the exact solver.

        It does not hold everywhere, and the exception is known.  The sweep screens
        terrain at the node level only - states *inside* a solid are removed by
        ``node_mask`` - while ``GlideTransitionModel.successors`` additionally
        rejects an edge whose segment passes *through* a solid even though both of
        its endpoints are outside.  So the sweep can relax a shortcut the shared
        adjacency does not contain, and V comes out below anything the adjacency
        can reach.  Measured at the seeded sensor: 0.3% of states at 100 m with a
        worst shortfall of 0.0071, and none at all at 25 m; the reported optimal
        trajectory is free of such edges at both resolutions.

        Rather than weaken the assertion to accommodate that, every violation is
        required to have exactly this cause.  A drift with any other cause - a
        changed cost, a changed terminal rule, a genuinely missing successor -
        still fails here.
        """
        unexplained: list[tuple[int, float]] = []
        checked = 0
        for state_id in self._sampled_states(terminal=False):
            state_id = int(state_id)
            if not len(self.mdp.actions(state_id)):
                continue
            checked += 1
            shortfall = self._optimality_shortfall(state_id)
            if shortfall <= 1.0e-9:
                continue
            if not self._has_terrain_rejected_successor(state_id):
                unexplained.append((state_id, shortfall))
        self.assertGreater(checked, 100, "too few states exercised to be meaningful")
        self.assertEqual(
            unexplained, [],
            "Bellman optimality fails at states with no through-terrain shortcut, "
            "so the two solvers disagree for some other reason",
        )

    def _has_terrain_rejected_successor(self, state_id: int) -> bool:
        """True when the graph builder threw away an edge for hitting terrain."""
        from bellman_geometry import GlideTransitionModel

        model = GlideTransitionModel(
            self.scene.grid, self.scene.terrain,
            self.scene.config.glider, self.scene.config.physical_scale,
        )
        _, _, rejected = model.successors(
            self.scene.grid.decode(state_id), include_rejected=True,
        )
        return any(item.reason == "terrain collision" for item in rejected)

    def test_reported_optimal_trajectory_stays_in_the_shared_graph(self) -> None:
        """Whatever the sweep allows internally, the answer it reports is clean."""
        ids = tuple(int(i) for i in self.oracle.glide_state_ids)
        self.assertGreater(len(ids), 1)
        for source_id, target_id in zip(ids, ids[1:]):
            targets = {int(edge.target_id) for edge in self.mdp.actions(source_id)}
            self.assertIn(
                target_id, targets,
                f"the reported optimal trajectory uses {source_id} -> {target_id}, "
                "which the collision-checked graph does not contain",
            )

    def test_terminal_states_agree_and_cost_nothing(self) -> None:
        terminal = self._sampled_states(terminal=True)
        self.assertGreater(len(terminal), 0, "the instance has no terminal states")
        for state_id in terminal:
            state_id = int(state_id)
            self.assertTrue(self.mdp.is_terminal(state_id))
            self.assertAlmostEqual(float(self.value[state_id]), 0.0, delta=TOLERANCE)
        for state_id in self._sampled_states(terminal=False)[:200]:
            self.assertFalse(self.mdp.is_terminal(int(state_id)))

    def test_transition_costs_agree_with_the_single_edge_integral(self) -> None:
        """The batched row and the per-edge integral are the same quantity."""
        from P1b_Exact_Local_SSE import _phase_cost

        worst = 0.0
        for state_id in self._sampled_states(terminal=False)[:40]:
            state_id = int(state_id)
            row = self.mdp.cost_row(state_id)
            for index, edge in enumerate(self.mdp.actions(state_id)):
                reference = _phase_cost(
                    self.mdp.edge_hazard(edge), edge.duration_s, self.mdp.objective,
                )
                worst = max(worst, abs(float(row[index]) - reference))
        self.assertLess(worst, 1.0e-12, f"batched cost drifted by {worst:.3e}")

    def test_evaluator_reproduces_the_exact_value_on_the_bellman_path(self) -> None:
        """The independent evaluator must score Bellman's own path correctly.

        If it does not, a gap measured with it says nothing about the learner.
        """
        ids = tuple(int(i) for i in self.oracle.glide_state_ids)
        self.assertGreater(len(ids), 1, "the oracle returned no glide trajectory")
        evaluation = evaluate_trajectory(self.mdp, ids)
        self.assertTrue(evaluation.reached_goal)
        self.assertAlmostEqual(
            evaluation.cost, float(self.value[ids[0]]), delta=1.0e-9,
            msg="the evaluator disagrees with the exact value function",
        )

    def test_evaluator_rejects_a_trajectory_that_leaves_the_graph(self) -> None:
        ids = tuple(int(i) for i in self.oracle.glide_state_ids)
        broken = (ids[0], ids[0])  # a state is never its own successor here
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.mdp, broken)


class LearnerStaysInsideTheGraphTests(unittest.TestCase):
    """Short training run, then check the learner never left the model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(
            ComputationCondition(spatial_resolution_m=RESOLUTION_M),
        )
        cls.sensor = default_sensor(cls.scene)
        cls.mdp = AttackerMDP(cls.scene, cls.sensor)
        candidates = switching_candidates(cls.scene, cls.sensor)
        if not candidates.by_state_id:
            raise unittest.SkipTest("no admissible switching state at the seed sensor")
        cls.starts = candidates.state_ids
        cls.config = QLearningConfig(episodes=300, evaluation_interval=100)
        cls.table = QTable(action_count=cls.mdp.max_action_count)
        cls.history = train(cls.mdp, cls.starts, cls.table, cls.config)

    def test_no_tried_action_is_outside_the_feasible_set(self) -> None:
        for state_id, mask in self.table.tried.items():
            count = self.mdp.action_count(state_id)
            self.assertFalse(
                bool(mask[count:].any()),
                f"state {state_id} has a tried action beyond its {count} successors",
            )

    def test_every_visited_state_is_a_graph_node(self) -> None:
        node_mask = np.asarray(self.scene.graph.node_mask, dtype=bool)
        for state_id in self.table.values:
            self.assertTrue(
                bool(node_mask[state_id]),
                f"state {state_id} is not a node of the shared graph",
            )

    def test_greedy_trajectory_edges_all_exist_in_the_graph(self) -> None:
        ids, _ = greedy_trajectory(
            self.mdp, self.table, int(self.starts[0]),
            max_steps=self.config.max_steps,
        )
        for source_id, target_id in zip(ids, ids[1:]):
            targets = {
                int(edge.target_id) for edge in self.mdp.actions(source_id)
            }
            self.assertIn(target_id, targets)

    def test_history_records_one_row_per_episode(self) -> None:
        self.assertEqual(len(self.history.episode), self.config.episodes)
        self.assertEqual(len(self.history.episode_cost), self.config.episodes)
        self.assertEqual(len(self.history.epsilon), self.config.episodes)
        self.assertTrue(all(
            value >= self.config.epsilon_min - TOLERANCE
            for value in self.history.epsilon
        ))

    def test_epsilon_decays_monotonically_toward_the_floor(self) -> None:
        epsilon = self.history.epsilon
        self.assertAlmostEqual(epsilon[0], self.config.epsilon, delta=TOLERANCE)
        self.assertLess(epsilon[1], epsilon[0])
        for earlier, later in zip(epsilon, epsilon[1:]):
            self.assertLessEqual(later, earlier + TOLERANCE)
            self.assertGreaterEqual(later, self.config.epsilon_min - TOLERANCE)

    def test_epsilon_reaches_the_floor_given_enough_episodes(self) -> None:
        """The floor is reached, just not within a short run.

        At the default 0.999 the anneal takes about three thousand episodes to fall
        from 1.0 to 0.05, so a test that asserted the floor after a few hundred
        would be asserting a fast decay - the thing the default exists to avoid.
        """
        config = QLearningConfig(episodes=1)
        epsilon = config.epsilon
        episodes = 0
        while epsilon > config.epsilon_min + TOLERANCE and episodes < 100_000:
            epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
            episodes += 1
        self.assertAlmostEqual(epsilon, config.epsilon_min, delta=TOLERANCE)
        self.assertGreater(episodes, 1_000, "the default anneal is too fast")
        self.assertLess(episodes, 10_000, "the default anneal never gets there")


class GlideSolveContractTests(unittest.TestCase):
    """The glide solve must start where Bellman started and report a real gap."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(
            ComputationCondition(spatial_resolution_m=RESOLUTION_M),
        )
        cls.sensor = default_sensor(cls.scene)
        cls.result = solve_glide_mdp(
            cls.scene, cls.sensor,
            QLearningConfig(episodes=500, evaluation_interval=100),
        )

    def test_starts_from_the_bellman_switching_state(self) -> None:
        oracle = exact_best_response(self.scene, self.sensor)
        self.assertEqual(
            self.result.start_state_ids, (int(oracle.switching_state_id),)
        )
        self.assertEqual(
            self.result.evaluation_state_id, int(oracle.switching_state_id)
        )

    def test_switching_is_a_readout_not_a_separate_solve(self) -> None:
        """No switching readout unless asked for, and it never retrains."""
        self.assertIsNone(self.result.switching)
        with_readout = solve_glide_mdp(
            self.scene, self.sensor,
            QLearningConfig(episodes=500, evaluation_interval=100),
            start_state_ids=candidate_start_states(self.scene, self.sensor),
            read_switching=True,
        )
        self.assertIsNotNone(with_readout.switching)
        readout = with_readout.switching
        candidates = candidate_start_states(self.scene, self.sensor)
        self.assertIn(readout.state_id, candidates)
        self.assertIn(readout.value_ranked_state_id, candidates)
        self.assertGreaterEqual(readout.worst_candidate_error, 0.0)

    def test_rollout_ranking_never_reports_a_cost_below_the_optimum(self) -> None:
        """The whole point of ranking by rollout: the numbers are achievable.

        A realised trajectory is scored by the original evaluator, so its cost is
        one the Attacker can actually pay.  It may be worse than the optimum, never
        better - if it ever were, the selection would be reading an estimate rather
        than a trajectory, which is the failure this ranking exists to avoid.
        """
        oracle = exact_best_response(self.scene, self.sensor)
        result = solve_glide_mdp(
            self.scene, self.sensor,
            QLearningConfig(episodes=500, evaluation_interval=100),
            start_state_ids=candidate_start_states(self.scene, self.sensor),
            read_switching=True,
        )
        readout = result.switching
        if readout.objective is None:
            self.skipTest("no candidate produced a goal-reaching rollout")
        self.assertGreaterEqual(
            readout.objective, float(oracle.attacker_objective) - 1.0e-9,
            "a rollout-scored objective came out below the exact optimum",
        )

    def test_q_learning_is_never_better_than_the_oracle(self) -> None:
        """A cost below the exact optimum would mean the two MDPs differ."""
        if self.result.q_cost is None:
            self.skipTest("the greedy policy did not reach the goal")
        self.assertGreaterEqual(
            self.result.q_cost, self.result.bellman_cost - 1.0e-9,
            "the learner beat the exact optimum, so the problems are not the same",
        )

    def test_coverage_fraction_is_a_real_fraction(self) -> None:
        self.assertGreater(self.result.reachable_state_action_pairs, 0)
        self.assertGreaterEqual(self.result.coverage_fraction, 0.0)
        self.assertLessEqual(self.result.coverage_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
