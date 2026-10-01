# DQN workspace

All new DQN environments, models, replay buffers, training code, evaluation code,
tests, notebooks, checkpoints, and DQN-specific figures belong in this directory.
`3D_0827` remains the source of truth for the existing terrain, transition,
hazard, objective, Bellman, and Stackelberg implementations.

The directory names begin with digits and therefore cannot be used as normal
Python package names.  In a notebook or script in this directory, use:

```python
import project_paths  # configures both sibling source directories

from P1b_condition import ComputationCondition, build_scene
from P1b_RL_approximation import AttackerMDP
from glider_gym_env import TerrainGlideEnv
```

The workspace virtual environment also has a `glider_hybrid_control_paths.pth`
file that exposes both directories to restarted kernels and Python processes,
regardless of their working directory.  `project_paths` is the portable fallback
for a new environment or clone.

Imports in the other direction use the future DQN module's filename directly.
Module filenames must therefore be unique across `3D_0827` and `3D_RL_DQN`.
