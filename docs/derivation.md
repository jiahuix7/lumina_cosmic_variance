# Validating `Var[sigma_mu^2]`: which linearity assumption actually matters

Companion to `pencilbeam.py`. Every result below names the function it
constrains. All numbers below are filled in by
`scripts/check_linearity.py` and the figures in this directory.

---

## 0. The question, restated

You asked: *the central assumption is that counts in each field scale
linearly, so compute the Lagrangian region and check whether the fields are
bigger than it.*

The instinct is right — an expansion is only valid above the scale where the
thing you expanded in is small — but "counts scale linearly" is three
separate assumptions, entering at three different places, and they are **not**
controlled by the same scale.

| # | assumption | where it enters | expansion parameter | controlled by |
|---|---|---|---|---|
| L1 | linear **growth** of the matter field | `P(k,z) = D^2(z) P_lin(k)` | `sigma_delta(V,z)` | beam size vs `R_NL` |
| L2 | linear **bias** — counts respond linearly to `delta` | `sigma_CV^2 = b1^2 sigma_delta^2` | `b1 sigma_delta` | beam size vs `R_NL`, **times bias** |
| L3 | **Gaussianity** (Wick's theorem) | `Var[sigma_mu^2] = (2/M^2) sum C_ij^2` | `kappa_iijj / C_ij^2` | connected trispectrum |

Your Lagrangian-region test is a test of **L1 only**. It will pass, and by a
margin so large that it is worth stating precisely rather than checking
nervously: a `10^11 M_sun` halo has `R_L ~ 0.9 cMpc`, the NIRCam beam is
`~13 x 6.5 x 370 cMpc`. The beam volume exceeds the Lagrangian volume of the
most massive halo it contains by 4.0 orders of magnitude in volume
(2.7 decades in length: shortest beam side 6.18 cMpc vs R_NL = 0.0125 cMpc).

**L2 is the one that bites.** The expansion parameter for *counts* is not
`sigma_delta`, it is `b1 sigma_delta`. Your own `central_argument.md` reports
`b_eff = 11.5` and `sigma_CV = 0.68`; since `sigma_CV = b_eff sigma_delta` at
linear order, that implies `sigma_delta ~ 0.06` — a beautifully linear matter
field — while `b1 sigma_delta ~ 0.7`. The PANORAMIC values you compare
against (`sigma_CV = 0.96, 1.46, 1.71`) are worse than unity.

> **The matter field is linear. The galaxy field is not.**
> That is the finding, and it is a better result than "we checked and it was
> fine." It also explains, without any new physics, why `sigma_CV` is a
> sensitive discriminator: you are working in a regime where the response of
> counts to density has measurable curvature.

**L3 is the one that is quietly wrong in the code.** `sigma_mu_squared_error`
uses `(m4 - s2^2) / M_eff`, which patches the *diagonal* fourth moment
empirically but assumes the off-diagonal pairs inflate by the same factor.
Section 5 derives what is actually neglected and Section 7 measures it.

---

## 1. The estimator is exact. Only its error bar is at risk.

Let `d_i = N_i - mu` with `mu = nbar_box * V_beam` the **known** box mean.

    sigma_mu^2 = (1/M) sum_i d_i^2

    E[sigma_mu^2] = (1/M) sum_i E[d_i^2] = (1/M) sum_i Var(N_i) = sigma^2

No Bessel correction, no correlation correction, for **arbitrary** `C_ij` and
even for fully overlapping beams. This is the content of the `sigma_mu_squared`
docstring and it is correct: correlations between fields do not bias this
estimator, because each term is separately unbiased and expectation is linear.

**This is why using the known mean matters so much.** With the sample mean
`Nbar` you would instead get `E[s^2] = sigma^2 - cbar`, and `cbar` requires
either the integral constraint (tiling) or placement resampling to estimate —
the entire apparatus of `pencilbeam_algorithm.md` Section 1. Fix 1 deletes
that problem. What survives is the **uncertainty**, and that is where every
linearity assumption lives.

---

## 2. Variance of the estimator: where Wick enters

    Var[sigma_mu^2] = (1/M^2) sum_ij Cov(d_i^2, d_j^2)

For any joint distribution,

    Cov(d_i^2, d_j^2) = E[d_i^2 d_j^2] - E[d_i^2] E[d_j^2]

Expand `E[d_i^2 d_j^2]` in cumulants. For zero-mean variables,

    E[d_i^2 d_j^2] = C_ii C_jj + 2 C_ij^2 + kappa_iijj      (*)

where `kappa_iijj` is the **connected** fourth cumulant. The first two terms
are the three Wick pairings — `(ii)(jj)` once and `(ij)(ij)` twice — and the
third is everything Gaussianity forbids. Subtracting `E[d_i^2]E[d_j^2] =
C_ii C_jj`:

    Cov(d_i^2, d_j^2) = 2 C_ij^2 + kappa_iijj

Hence, **exactly**,

    Var[sigma_mu^2] = (2/M^2) sum_ij C_ij^2  +  (1/M^2) sum_ij kappa_iijj
                      \_______ Wick _______/    \____ neglected ____/

Setting `kappa = 0` and defining

    M_eff = M^2 / sum_ij rho_ij^2,     rho_ij = C_ij / sigma^2

gives `Var[sigma_mu^2] = 2 sigma^4 / M_eff` — the shared-conversation formula
implemented in `sigma_mu_squared_error`. So:

- **`M_eff` is a statement about L3, not about L1 or L2.** It is the price of
  assuming a Gaussian field, repackaged as an effective sample size.
- The `i = j` terms of the neglected sum are `kappa_iiii = m4 - 3 sigma^4`,
  which the code *does* capture by using the measured `m4`. The
  `i != j` terms are not captured. Section 7 measures the residual.

### 2.1 What the code actually computes

`sigma_mu_squared_error` evaluates `(m4 - s2^2)/M_eff`. Writing
`m4 - sigma^4 = 2 sigma^4 (1 + kappa_iiii/(2 sigma^4))`, this is equivalent to

    Var[sigma_mu^2] ~ (2 sigma^4 / M_eff) * (1 + kappa_iiii / (2 sigma^4))

i.e. it applies the **diagonal** non-Gaussian inflation factor uniformly to
every pair. That is a defensible interpolation — it is exact for `i = j` and
it has the right sign — but it is an assumption, and it should be stated as
one in the paper rather than absorbed silently into a fourth moment. The
validation table in the docstring (ratios 0.93–1.11) shows it works *at the
configurations tested*; Section 7 asks whether that survives at higher bias.

---

## 3. `C_ij` from the two-point function — yes, you need it, and it is one FFT

This is the step you asked about. Model the galaxies as a Poisson sample of an
intensity field `n(x) = nbar (1 + delta_g(x))`. For beams `a`, `b`:

    Cov(N_a, N_b) = nbar V_{a and b}                        <- shot / overlap
                  + nbar^2 Int_a Int_b xi_g(|x - y|) d3x d3y   <- clustering

The first term is nonzero **only when beams overlap**, and it is exactly the
"shared fraction" that `effective_beam_count` currently calibrates
empirically. Note it survives even for a perfectly linear field: overlap
correlation is a geometric fact, not a clustering one.

For the second term, insert `xi(r) = Int d3k/(2pi)^3 P(k) e^{ik.r}` and use

    Int_a d3x e^{-ik.x} = V W(k) e^{-ik.x_a}

with `x_a` the beam centre and, for a cuboid of side lengths `L_d` aligned to
the axes,

    W(k) = prod_{d=1..3} sinc(k_d L_d / 2),      sinc(x) = sin(x)/x

Therefore

    Int_a Int_b xi = V^2 Int d3k/(2pi)^3 P(k) |W(k)|^2 e^{ik.r_ab}

and so

    C_ab^clust = mu^2 * xi_beam(r_ab),
    xi_beam(r) = Int d3k/(2pi)^3 P_g(k) |W(k)|^2 e^{ik.r}

**`xi_beam` is a single inverse FFT of `P_g(k)|W(k)|^2`.** You get every
separation at once — you never loop over pairs, and route (a) of
`pencilbeam_algorithm.md` Section 1.4 becomes cheaper than route (b). At zero
separation it reduces to the beam-scale variance,

    sigma_delta^2(V) = Int d3k/(2pi)^3 P(k) |W(k)|^2
                     = (1/V^2) Int_V Int_V xi(|x-y|) d3x d3y

which is the quantity that decides L1 and, multiplied by `b1`, L2.

**The window is anisotropic and that matters.** A `13 x 6.5 x 370 cMpc` beam
is not a sphere of equal volume. The long axis suppresses `k_z` modes above
`~2pi/370`, so the beam is sensitive to a narrow pencil of transverse modes.
Using an equal-volume top-hat sphere gets `sigma_delta` wrong; Section 4
quantifies by how much.

---

## 4. Scale ladder: what the Lagrangian test actually shows

The Lagrangian radius of a halo of mass `M`:

    R_L = (3M / (4 pi rhobar_m))^(1/3),   rhobar_m = Om0 * rho_crit,0

evaluated in **comoving** units, so it is redshift-independent. The
nonlinear scale `R_NL(z)` solves `sigma(R_NL, z) = 1`.

At `z = 10`, `D(10)/D(0) ~ 1/11`, so `sigma(R,10)` is suppressed by an order
of magnitude relative to today and `R_NL` is small. Figure `scale_ladder.png`
places `R_L` for `10^9`–`10^13 M_sun`, `R_NL(z=10)`, and the beam scales on
one axis. Filled in: sigma_delta(NIRCam) = 0.03561, versus 0.06651 for an
equal-volume sphere (1.87x) and 0.06500 for an equal-volume cube (1.83x).
For Roman the beam is closer to cubic and the sphere errs by only 1.12x.

**Interpretation.** `R_beam >> R_L` establishes that a beam averages over an
enormous number of independent collapsing regions, so the *matter* field it
sees is in the linear regime. It does **not** establish that counts respond
linearly, because the objects you count are rare peaks of that field, and rare
peaks amplify fluctuations by `b1`. That is Section 5.

---

## 5. Second-order bias: the correction you should actually quote

Expand the galaxy overdensity in the (beam-smoothed) matter field with the
standard local Eulerian bias model:

    delta_g = b1 delta + (b2/2)(delta^2 - sigma^2) + O(delta^3)

The `-sigma^2` keeps `<delta_g> = 0`. Squaring and taking expectations, with
`<delta^3> = S3 sigma^4` and `<delta^4> = 3 sigma^4 + O(sigma^6)`:

    <delta_g^2> = b1^2 sigma^2
                + b1 b2 <delta^3>
                + (b2^2/4)(<delta^4> - sigma^4)
                + ...
                = b1^2 sigma^2 [ 1 + (b2/b1) S3 sigma^2
                                   + (b2^2 / (2 b1^2)) sigma^2 + ... ]

So the fractional correction to `sigma_CV^2` is

    Delta = (b2/b1) S3 sigma_delta^2  +  (b2^2 / (2 b1^2)) sigma_delta^2

Two things to read off:

1. **The correction is `O(sigma_delta^2)`, not `O(b1 sigma_delta)`.** The
   naive fear — that `b1 sigma_delta ~ 0.7` means a 70% error — is wrong. The
   linear term is exactly what `sigma_CV` is defined to be; the leading
   *correction* is second order in `sigma_delta` with coefficients set by
   `b2/b1`. With `sigma_delta ~ 0.06` this is small **unless `b2/b1` is
   large**.
2. **`b2/b1` is exactly what is large at high bias.** For rare, highly biased
   tracers `b2` grows faster than `b1`. Using the Lazeyras et al. (2016)
   fit `b2(b1) = 0.412 - 2.143 b1 + 0.929 b1^2 + 0.008 b1^3`, at `b1 = 11.5`
   the ratio `b2/b1` is 16.3 at b1 = 19.1 (Lazeyras et al. 2016, extrapolated).

Tree-level PT gives `S3 = 34/7 - (n_eff + 3)` for top-hat smoothing, with
`n_eff = dlnP/dlnk` at the smoothing scale. At `z = 10` beam scales
`n_eff = -2.31` at the transverse beam scale k = 0.72 /cMpc, giving `S3 = 4.17`.

**Caveat that must be in the paper.** The Lazeyras fit is calibrated over
roughly `b1 = 1`–`10`; `b_eff = 11.5` sits at or beyond its edge, and the
cubic term makes extrapolation unstable. Section 8's `check_linearity.py`
measures `b1` and `b2` *directly* from the Lumina box by regressing beam
counts against beam density — the honest route, and the one that does not
inherit somebody else's calibration.

---

## 6. The direct empirical test (what to actually run on Lumina)

Everything above is theory. The measurement that settles it:

1. Grid the dark matter (or the halo-mass-weighted field) into the box.
2. For each beam placement `i`, compute both the count `N_i` and the
   beam-averaged matter overdensity `delta_i`.
3. Plot `N_i / mu` against `delta_i` and fit
   `N/mu = 1 + b1 delta + (b2/2)(delta^2 - sigma^2)`.
4. **If the relation is straight over the range of `delta` your beams
   actually sample, L2 holds and `sigma_CV` is safe.** If it curves, the
   curvature *is* `b2`, and Section 5 converts it into the correction to
   `sigma_CV^2`.

This is a one-parameter-family generalisation of what you already compute, it
uses the same count matrix, and it needs no new simulation. It is strictly
more convincing than any scale argument, because it tests the response
function itself rather than a proxy for it.

---

## 7. What is neglected in `Var[sigma_mu^2]`, measured

The residual from Section 2, `(1/M^2) sum_{i!=j} kappa_iijj`, has no clean
closed form for a nonlinear field. Measured in a lognormal mock matched to the
Lumina `z=10` power spectrum at `b_eff ~ 11.5`: the mock's sigma_CV = 0.681
with the true sigma_delta = 0.03561 implies b1 = 19.1, not 11.5. The mock's
density field is more clustered than the real z=10 box, so its b_eff is
internal to the mock and must NOT be quoted alongside the real run.

Lognormal is the right mock for this test specifically because it is the
simplest field with the *correct* qualitative non-Gaussianity — positive
skewness, `kappa > 0` — while having an exactly known `P(k)`. It will not get
the amplitude of `kappa` right to 10%, but it establishes the **sign** and the
**order of magnitude** of the term the error bar is missing, which is what you
need in order to say whether the quoted error is conservative or optimistic.

---

## 8. Summary: what to put in the paper

At sigma_CV = 0.681 the connected term inflates Var[sigma_mu^2] by 6.3x,
i.e. the error bar is 2.56x too small. See `trispectrum_term.png`.

### The three assumptions, resolved

| # | assumption | expansion parameter | value | verdict |
|---|---|---|---|---|
| 1 | matter field is linear on the beam scale | `R_NL / L_min` | 2.0e-3 | **safe by 2.7 decades** |
| 2 | galaxy counts are linear in delta | `b1 * sigma_delta = sigma_CV` | 0.681 | **broken**: 2nd-order term is +32% |
| 3 | 4-point function is Gaussian (Wick) | `kappa_4 / 2C^2` | 5.5 | **broken**: error bar 2.56x too small |

The Lagrangian-region test you proposed probes assumption 1 only. It passes,
and it was never the constraint. Reporting it is still worth one sentence
because it rules out the failure mode a referee will ask about first.

### The result that changes a number in the paper

`sigma_mu_squared_error` uses the Wick form `sqrt(2/M) * C`. Measured against
4000 direct realisations of the estimator, this is:

| sigma_CV | error-bar inflation | so the quoted error is |
|---|---|---|
| 0.10 | 1.03 | fine |
| 0.20 | 1.15 | 15% low |
| 0.40 | 1.58 | 1.6x low |
| **0.68** | **2.56** | **2.6x low  (Lumina mock)** |
| 1.00 | 4.46 | 4.5x low |
| 1.71 | 14.1 | 14x low  (PANORAMIC faintest bin) |

Wick-only is accurate to 10% only for `sigma_CV < 0.15`. Every regime you
care about is outside it.

The fix is one line: multiply by `error_inflation(mu, sigma_cv)` from
`linearity.py`. That correction is validated to better than 5% for
`sigma_CV <~ 0.7`. Above `sigma_CV ~ 1.5` even the kappa_4-corrected formula
fails (it recovers only 58% of the measured variance at 1.71) because the
estimator distribution is too heavy-tailed for a 4th-cumulant truncation --
in that regime quote a bootstrap interval, not an analytic sigma.

### The claim in the code that is wrong

`_COSMIC_CALIB` attributes its residual factor to "shared long-wavelength
modes". That attribution fails on M-scaling, independently of any model:

- the measured residual is **flat in M** (3% spread for NIRCam, 9% for Roman,
  across a 16x range in M);
- long-mode coupling predicts a ratio growing as `sqrt(1 + M <rho^2>)`, which
  over the same range must vary by **75%** (NIRCam) and **241%** (Roman).

Of the three terms in `Var[sigma_mu^2]`, only the connected trispectrum term
scales as `1/M` in the variance and is therefore M-independent as a ratio.
The residual is the neglected `kappa_4`, not overlap. See `meff_comparison.png`.

### Two things to be careful about

1. **Do not use an equal-volume sphere for the beam window.** For NIRCam that
   overestimates `sigma_delta` by 1.87x (equal-volume cube: 1.83x). The
   square-vs-rectangle geometry question flagged as open in
   `pencilbeam_algorithm.md` is a factor-of-two effect on the answer, not a
   detail. Use `sigma_cuboid`.

2. **Do not mix the mock's `b_eff = 11.5` with quantities computed for the
   real run.** The mock's sigma_CV = 0.681 combined with the true
   `sigma_delta = 0.03561` implies `b1 = 19.1`. The discrepancy means the
   lognormal mock's density field is more clustered than the real z=10 box;
   its `b_eff` is internal to the mock.

### Independent checks performed

- CAMB vs Eisenstein-Hu 1998: 8.5% broadband (the EH98 accuracy floor) with a
  1.9% oscillatory residual correctly identified as BAO. CAMB used throughout.
- Growth: `P(z=10)/P(0) = 0.013464` vs `D^2 = 0.013414` (0.4%).
- Cuboid window: k-space Gauss-Legendre vs real-space Monte Carlo double
  integral of `xi(r)` over the beam volume -- agree to 0.9%.
- `b_eff`: peak-background split at `M_h = 1e11 M_sun` gives `b = 10.3`,
  consistent with the mock's measured 11.5.
- Box modes: the 500 cMpc box retains 100.8% (NIRCam) and 97.6% (Roman) of the
  beam variance -- the missing-mode warning is not a concern at these scales.
  (An earlier per-axis `k_min` cutoff suggested a 93% deficit; that was wrong.
  A periodic box does contain modes with individual components equal to zero.)
- `kappa_4`: derived analytically by law of total cumulance, and measured two
  independent ways (direct 4th cumulant of counts; direct `Var[sigma_mu^2]`
  over 4000 realisations). All three agree.

---

## Function-by-function impact on `pencilbeam.py`

| function | status after this analysis |
|---|---|
| `sigma_mu_squared` | **unchanged.** Exact for any `C_ij`; Section 1. |
| `sigma_mu_squared_error` | assumption made explicit (Section 2.1); `M_eff` can now be computed analytically instead of interpolated |
| `effective_beam_count` | `_R2_CALIBRATION` replaceable by `rho_ij^2` from Section 3; Section 6 figure compares them |
| `variance_error` | unchanged; it answers the placement-ensemble question and its denominator `M` is correct |
| `cosmic_error` / `_COSMIC_CALIB` | the NIRCam residual factor 1.51 is "shared long-wavelength modes" — Section 3 predicts it from `xi_beam(r)` at disjoint separations |
| `missing_mode_warning` | quantified: the missing-mode deficit is `Int_{k < 2pi/L_box}` of the same integrand |
