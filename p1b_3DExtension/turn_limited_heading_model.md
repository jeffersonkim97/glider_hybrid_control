# 3D heading-state and turn-rate update

## Why the legacy path looked straight

The legacy 3D prototype used spatial state

\[
s_k=(x_k,y_k,h_k)
\]

and independently selected heading as part of every action. Consequently,
the optimizer could keep one heading for many stages (a straight corridor)
and then change heading instantaneously at a grid node. The resulting path
was piecewise linear. A free heading action permits lateral motion, but does
not by itself model turning inertia.

Straight segments are still expected in open homogeneous space: once a
low-cost corridor has been selected, the shortest connection within that
corridor is locally straight. The unphysical feature was not the presence
of straight segments; it was an unlimited corner between them.

## Updated discrete dynamics

The Bellman state is now

\[
s_k=(x_k,y_k,h_k,\psi_k),
\]

where \(\psi_k\) is periodic heading. The selected course becomes the next
heading state and must satisfy

\[
\psi_{k+1}=\psi_k+\omega_k\Delta t,\qquad
|\omega_k|\leq\omega_{\max}.
\]

The spatial update uses the selected course (a semi-implicit discrete
kinematic update):

\[
\begin{aligned}
x_{k+1}&=x_k+v_k\cos\gamma_k\cos\psi_{k+1}\Delta t,\\
y_{k+1}&=y_k+v_k\cos\gamma_k\sin\psi_{k+1}\Delta t,\\
h_{k+1}&=h_k+v_k\sin\gamma_k\Delta t.
\end{aligned}
\]

The current configuration uses \(\omega_{\max}=5\) deg/s. At 20 m/s this
corresponds to a horizontal minimum-turn-radius scale of approximately
230 m. With the current 10-degree heading grid and approximately 3-second
coarse transition, realized heading changes occur in 10-degree increments.

The powered segment azimuth supplies the initial heading state at the
switching point. Therefore the glider cannot instantaneously choose a
completely different lateral direction at the switch.

## What should be said about the new plots

The authoritative result remains a coarse-grid trajectory, so its rendered
line is a polygonal approximation of a bounded-curvature path rather than a
fully smooth flight-mechanics trajectory. Grid snapping can also make
several small heading commands appear as one nearly straight screen-space
segment.

A concise explanation is:

> Heading now changes the reachable lateral corridor, but it is also carried
> as a dynamic state. Adjacent course commands obey a finite turn-rate limit.
> Straight portions remain optimal inside a corridor; gradual changes occur
> only where the route must bend. Any residual polygonal appearance is the
> coarse discretization, not instantaneous turning.

For publication-quality continuous curves, the next refinement should use
constant-turn arc integration (or bank-angle dynamics) inside each Bellman
transition and evaluate terrain/LOS along those arcs. A purely cosmetic
spline should not be presented as an authoritative optimized trajectory.

## Coarse replay finding (2026-08-06)

The first fixed-sensor coarse rerun satisfied the new heading-rate bound
(realized maximum 3.75 deg/s versus the configured 5 deg/s), but it exposed
a separate legacy defect. The Bellman path reaches the goal only because
each off-grid kinematic endpoint is reset to its nearest grid node. When the
same selected speed, flight-path angle, heading, duration, and terminal
fractions are integrated without those resets, the endpoint misses the goal
by approximately 995 m.

This makes the rerun a diagnostic result, not a presentation-ready mission
solution. The required next change is to port the physical successor-grid
contract from the current 2D solver and extend it to 3D heading state:

1. choose an exact grid-to-grid spatial successor;
2. derive edge heading, flight-path angle, length, and duration from the two
   endpoints;
3. enforce the heading-rate bound from the incoming heading state;
4. integrate detection hazard and certify terrain/LOS along that exact edge;
5. carry the derived heading into the successor state.

That construction removes endpoint snapping by definition. Constant-turn
arc edges can then replace straight chords if continuous curvature, rather
than a bounded-angle polygonal approximation, is required.
