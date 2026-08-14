# Optimal Lower Bound Analytic Equation

## 1. Attacker objective

Let $H_{\mathrm{mission}}$ denote the cumulative mission hazard and let
$T_{\mathrm{mission}}$ denote the total mission time. The general Attacker
objective is

$$
J_A
=
w_{\mathrm{PoD}}
\frac{H_{\mathrm{mission}}}{H_{\mathrm{ref}}}
+
w_{\mathrm{time}}
\frac{T_{\mathrm{mission}}}{T_{\mathrm{ref}}},
\qquad
H_{\mathrm{mission}}\ge 0.
$$

The current configuration uses $H_{\mathrm{ref}}=1$, but the reference is
retained in the formulation so that the normalization remains explicit.

## 2. Sensor-wise cumulative hazard

For sensor $s$, let $\lambda_{p,s}(t)$ and $\lambda_{g,s}(t)$ denote the
powered-phase and glide-phase detection hazard rates. For an additive-hazard
fusion model,

$$
H_{\mathrm{mission}}
=
\sum_s
\left[
\int_0^{T_p}\lambda_{p,s}(t)\,\mathrm{d}t
+
\int_{T_p}^{T_p+T_g}\lambda_{g,s}(t)\,\mathrm{d}t
\right].
$$

The corresponding cumulative mission probability of detection is

$$
P_{\mathrm{mission}}
=
1-\exp\!\left(-H_{\mathrm{mission}}\right),
$$

or equivalently,

$$
H_{\mathrm{mission}}
=
-\log\!\left(1-P_{\mathrm{mission}}\right).
$$

Thus the current Attacker objective uses additive cumulative hazard rather
than probability of detection directly.

## 3. Current sensor hazard-rate model

Let $R_s(x)$ be the slant range from state $x$ to sensor $s$, subject to the
configured range floor:

$$
R_{s,\mathrm{eff}}(x)
=
\max\!\left(R_s(x),R_{\mathrm{floor}}\right).
$$

The powered acoustic hazard rate is

$$
\lambda_{p,s}(x)
=
\frac{k_{a,s}v_p^{m_s}}
{R_{s,\mathrm{eff}}(x)^2}.
$$

The glide hazard rate is

$$
\lambda_{g,s}(x,v,\gamma)
=
\mathbf 1_{\mathrm{LOS},s}(x)
\left[
\frac{k_{r,s}\operatorname{RCS}(\theta_s)}
{R_{s,\mathrm{eff}}(x)^4}
+
\frac{k_{d,s}v_{r,s}^{2}}
{R_{s,\mathrm{eff}}(x)^4}
\right].
$$

Here $\theta_s$ is the sensor-relative aspect angle and $v_{r,s}$ is the
radial velocity relative to sensor $s$.

## 4. Sensor-wise analytical hazard-rate lower bounds

For a fixed sensor $s$ and bounded airspace $\Omega$, define the maximum
possible sensor range

$$
\overline R_s
=
\max_{x\in\Omega}\|x-s\|_2,
$$

and its range-floor-adjusted value

$$
\overline R_{s,\mathrm{eff}}
=
\max\!\left(\overline R_s,R_{\mathrm{floor}}\right).
$$

The powered hazard rate then satisfies

$$
\lambda_{p,s}(t)
\ge
\underline\lambda_{p,s}
=
\frac{k_{a,s}v_p^{m_s}}
{\overline R_{s,\mathrm{eff}}^2}.
$$

If the glide phase is constrained to the LOS-visible region, then
$\mathbf 1_{\mathrm{LOS},s}=1$. Since
$\operatorname{RCS}(\theta_s)\ge\operatorname{RCS}_{\min,s}$ and
$v_{r,s}^{2}\ge0$,

$$
\lambda_{g,s}(t)
\ge
\underline\lambda_{g,s}
=
\frac{k_{r,s}\operatorname{RCS}_{\min,s}}
{\overline R_{s,\mathrm{eff}}^4}.
$$

The Doppler contribution is conservatively set to zero in this lower bound.
If occluded glide states are admissible, the global lower bound on the
LOS-gated glide hazard rate must instead be set to zero.

For multiple sensors,

$$
\underline\lambda_p
=
\sum_s\underline\lambda_{p,s},
\qquad
\underline\lambda_g
=
\sum_s\underline\lambda_{g,s}.
$$

Consequently,

$$
H_{\mathrm{mission}}
\ge
\underline\lambda_pT_p
+
\underline\lambda_gT_g.
$$

## 5. Phase-weighted objective lower bound

Substituting the hazard-rate lower bounds into the Attacker objective gives

$$
J_A
\ge
\left(
w_{\mathrm{PoD}}
\frac{\underline\lambda_p}{H_{\mathrm{ref}}}
+
\frac{w_{\mathrm{time}}}{T_{\mathrm{ref}}}
\right)T_p
+
\left(
w_{\mathrm{PoD}}
\frac{\underline\lambda_g}{H_{\mathrm{ref}}}
+
\frac{w_{\mathrm{time}}}{T_{\mathrm{ref}}}
\right)T_g.
$$

Define

$$
\alpha_p
=
w_{\mathrm{PoD}}
\frac{\underline\lambda_p}{H_{\mathrm{ref}}}
+
\frac{w_{\mathrm{time}}}{T_{\mathrm{ref}}},
$$

and

$$
\alpha_g
=
w_{\mathrm{PoD}}
\frac{\underline\lambda_g}{H_{\mathrm{ref}}}
+
\frac{w_{\mathrm{time}}}{T_{\mathrm{ref}}}.
$$

Then

$$
J_A\ge\alpha_pT_p+\alpha_gT_g.
$$

## 6. Switching-state analytical lower bound

Let $x_0$ be the launch state, $x_g$ the goal state, and $\sigma$ the
powered-to-glide switching state. Under the current straight powered segment
at fixed powered speed $v_p$,

$$
T_p(\sigma)
=
\frac{\|\sigma-x_0\|_2}{v_p}.
$$

For any dynamically feasible glide trajectory with speed bounded above by
$v_{g,\max}$,

$$
T_g(\sigma)
\ge
\frac{\|x_g-\sigma\|_2}{v_{g,\max}}.
$$

Let $\Sigma_{\mathrm{relaxed}}$ be a switching-state set that contains every
switching state admitted by the original problem. A valid lower bound is

$$
\boxed{
J_{\mathrm{LB}}(s)
=
\inf_{\sigma\in\Sigma_{\mathrm{relaxed}}}
\left[
\alpha_p
\frac{\|\sigma-x_0\|_2}{v_p}
+
\alpha_g
\frac{\|x_g-\sigma\|_2}{v_{g,\max}}
\right]
}.
$$

For every fixed sensor configuration $s$, this satisfies

$$
J_{\mathrm{LB}}(s)
\le
J_A^*(s),
$$

provided that:

1. $\Sigma_{\mathrm{relaxed}}$ contains the original switching set;
2. the sensor-wise hazard-rate inequalities hold over the complete relevant
   airspace;
3. $v_{g,\max}$ is a valid upper bound on every admissible glide speed; and
4. any numerical evaluation of the infimum uses a certified lower enclosure.

## 7. Fully closed-form time-only fallback

Because $H_{\mathrm{mission}}\ge0$, let

$$
\overline v
=
\max(v_p,v_{g,\max}).
$$

Every trajectory from $x_0$ to $x_g$ satisfies

$$
T_{\mathrm{mission}}
\ge
\frac{\|x_g-x_0\|_2}{\overline v}.
$$

Therefore the following weaker but fully closed-form bound always holds:

$$
\boxed{
J_A^*
\ge
w_{\mathrm{time}}
\frac{\|x_g-x_0\|_2}
{\overline vT_{\mathrm{ref}}}
}.
$$

## 8. Analytical validity and numerical evaluation

The inequalities defining the bound are analytical. If the switching states
are restricted to the terrain-dependent LOS switching boundary, the final
infimum generally does not have a closed-form minimizer. In that case, a
certified one-dimensional global minimization or interval method may be used
to calculate a numerical value $L^-$ such that

$$
L^-
\le
\inf_{\sigma\in\Sigma_{\mathrm{relaxed}}}
\left[
\alpha_p
\frac{\|\sigma-x_0\|_2}{v_p}
+
\alpha_g
\frac{\|x_g-\sigma\|_2}{v_{g,\max}}
\right]
\le
J_A^*.
$$

The number reported as a certified lower bound must be the lower endpoint
$L^-$, not an unconstrained local optimizer result or an ordinary sampled
minimum.
