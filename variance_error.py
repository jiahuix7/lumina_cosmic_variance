"""Uncertainty on the cosmic-variance estimator sigma_CV^2 from pencil-beam counts.

Estimates Var[sigma_hat_mu^2] where sigma_hat_mu^2 = (1/M) sum_i (N_i - mu)^2 is
the known-mean variance estimator, and propagates it to sigma_CV^2.

Written for lumina_cosmic_variance.  Validated 2026-08-12 against independent lognormal
boxes using the real Lumina beam geometry (135 x 67 x 369 cMpc, 100 beams, 500 cMpc box).
See "ACCURACY, HONESTLY" below for the size of the residual disagreement.

WHY NOT THE GAUSSIAN (WICK) FORMULA
-----------------------------------
The usual result Var[sigma_hat_mu^2] = 2 C_ii^2 / M follows from Wick's theorem, i.e.
from assuming the counts are jointly Gaussian.  For the Lumina setup it underestimates
the error bar by 1.5-1.6x.  Two separate effects, in order of importance:

  1. BEAM OVERLAP (65-85% of the excess).  Each beam is 2.7% of the box volume, so 100
     beams cover 269% of the box.  The M(M-1) off-diagonal terms Cov(Y_i, Y_j) are large.
  2. NON-GAUSSIAN DENSITY FIELD (15-35%).  The connected fourth cumulant kappa_4 adds to
     the diagonal.

Low counts are NOT the problem.  For pure Poisson, K_ii / 2C_ii^2 = 1 + 1/(2 mu) exactly,
which at mu = 8.4 is a 2.9% effect on the error bar.  Discreteness is negligible here.

WHY YOU CANNOT JUST SUM Cov(Y_i, Y_j) FROM ONE REALISATION
----------------------------------------------------------
It is tempting to estimate the general result

    Var[sigma_hat_mu^2] = (1/M^2) sum_i sum_j Cov(Y_i, Y_j),   Y_i = (N_i - mu)^2

by plugging in sample covariances from a single set of M beams.  That is identically
zero, for any data whatsoever.  With Ybar the sample mean of Y,

    sum_{i != j} (Y_i - Ybar)(Y_j - Ybar) = (sum_i (Y_i - Ybar))^2 - sum_i (Y_i - Ybar)^2
                                          = 0 - sum_i (Y_i - Ybar)^2,

so the off-diagonal sum exactly cancels the diagonal and the total vanishes.  This is the
integral constraint: one realisation carries no information about its own mean.  Binning
the pairs by separation does not help either -- summing the bins reconstructs the same
zero (verified numerically: 3e-13 against a truth of 8.7e+01).

Three routes that do work are provided below.

ROUTES
------
var_sigma_mu2_analytic : lognormal kappa_4 + fitted off-diagonal law.  Needs the beam
    geometry moments (beam_geometry_moments).  Agreed with simulation to 7-9%.
    Recommended default.
var_sigma_mu2_replacement : RECOMMENDED WHEN YOU HAVE ONE BOX.  Measures the beam-placement
    half of the variance directly by re-placing the beams many times in your single box, and
    predicts only the box-to-box half analytically.  Median 0.96x truth with a 16-84% spread
    of 1.85 -- about half the scatter of the fully analytic route, and it replaces the
    empirical rho^1.2 law with a measurement.
var_sigma_mu2_paircov / pair_covariance_profile : measures K22(r) from all B(B-1)/2 unique
    beam pairs, binned by separation, instead of assuming the rho^1.2 law.  Binning escapes
    the integral constraint that zeroes the unbinned double sum; what remains is a single
    additive constant equal to the same box-to-box term, restored analytically.  Ensemble
    mean 1.05e+02 vs a truth of 1.06e+02 (uncorrected: 5.24e+01), but on ONE box its
    scatter is WIDER than var_sigma_mu2_replacement -- use it for the K22(r) diagnostic,
    which no other route provides, and take the headline number from the replacement route.
var_sigma_mu2_jackknife : delete-one-slab spatial jackknife, internal to one box.
    Assumption-free but biased LOW (median 0.55x, mean 0.83x truth at M=100) because a
    sub-volume cannot see modes larger than itself.  Use as a lower bound / sanity check.
var_sigma_mu2_ensemble : the honest route.  Var over independent realisations (different
    boxes, or different snapshots).  Unbiased by construction; needs > 1 realisation.

WHY SUB-DIVIDING THE BOX DOES NOT GIVE YOU MORE REALISATIONS
------------------------------------------------------------
The Lumina beam is 369 cMpc along the line of sight in a 500 cMpc box, so box/beam = 1.35
on that axis: you cannot fit even TWO non-overlapping sub-volumes each capable of containing
one beam.  Splitting the box into pseudo-realisations is ruled out by geometry, not by
statistics.  What you CAN do is re-place the beams (var_sigma_mu2_replacement) -- but note
that this measures only part of the answer, see below.

THE VARIANCE SPLITS IN HALF, AND ONE HALF IS UNREACHABLE FROM ONE BOX
--------------------------------------------------------------------
By the law of total variance,

    Var[sigma_hat_mu^2] = E_box[Var_place] + Var_box[E_place]
                          \_ measurable _/   \_ NOT measurable _/

Measured for the Lumina geometry at M=100 (100 boxes x 20 placements):

    bin       placement half   box-to-box half
    -19.5          48%              52%
    -20.0          57%              43%
    -20.5          66%              34%

Re-placing beams inside one fixed box holds that box's long-wavelength modes fixed and so
recovers ONLY the first column -- it returns ~0.56x the truth if used alone.  The second
column is the variance of the box's own realised sigma^2, which a single box cannot report
about itself.  It has to be predicted:

    Var_box[sigma^2_box] ~= f_NG * 2 (sigma^2)^2 / N_eff,     N_eff = V_box / V_beam

with f_NG = 1.2-1.7 (measured; 1.4 default) the non-Gaussian enhancement.  Because this
predicted piece is only a third to a half of the total, a 20% error on f_NG costs <10% on
the final error bar -- which is why the hybrid beats the fully analytic route.

ACCURACY, HONESTLY
------------------
Two independent simulation paths of mine disagree by 2.9 sigma in the -19.5 bin
(6.90e+02 vs 8.47e+02 +/- 5.4e+01); the other two bins agree to 1.00-1.11x.  I have not
resolved which path is right.  Take the accuracy of the analytic route as ~+/-15%, not the
2-7% an earlier version of this docstring claimed.  All routes agree the Wick formula is
low by 1.5-1.6x, and that conclusion is far larger than this disagreement.

CAVEATS
-------
* The off-diagonal law K_ij = 0.357 K_ii |rho|^1.214 is EMPIRICAL.  It was measured across
  three density PDFs (gamma S3=2.0, lognormal S3=3.5, quadratic-PT S3=3.7), which all gave
  exponent 1.11-1.26 while Wick requires exactly 2.0, so the departure from Wick is robust.
  But an attempted decomposition into a linear connected piece plus the Wick rho^2 piece did
  not close (the fitted quadratic coefficient came out ~4x too large), so this is a fit, not
  a derivation.  Do not extrapolate far outside rho < 0.4, sigma_CV < 0.7.
* kappa_4 assumes a lognormal one-point PDF for the galaxy density.  Real halo counts have
  their own fourth cumulant.  If you can afford several independent boxes or snapshots,
  var_sigma_mu2_ensemble supersedes all of this and needs no PDF assumption.
* Re-placing beams inside a single fixed box is NOT an ensemble: it holds the box's
  long-wavelength modes fixed and returns 0.56x the truth.  It measures the beam-sampling
  part only.  Do not use it as the error bar.
"""

import numpy as np

__all__ = [
    "beam_geometry_moments",
    "var_sigma_mu2_analytic",
    "var_sigma_mu2_replacement",
    "pair_covariance_profile",
    "var_sigma_mu2_paircov",
    "var_sigma_mu2_jackknife",
    "var_sigma_mu2_ensemble",
    "sigma_cv_squared_error",
    "OFFDIAG_AMPLITUDE",
    "OFFDIAG_EXPONENT",
]

# Empirical off-diagonal law K_ij / K_ii = A |rho_counts|^p  (see module docstring).
OFFDIAG_AMPLITUDE = 0.357
OFFDIAG_EXPONENT = 1.214


def beam_geometry_moments(dims, pk_fn, box_size, n_pairs=1200, nmax=70,
                          z=10.0, seed=31, xi_beam_box=None):
    """Moments of the beam-beam correlation coefficient for random placement.

    Returns {'rho2': <rho^2>, 'rho_p': <|rho|^OFFDIAG_EXPONENT>, 'sigma_delta': ...}
    where rho is the correlation coefficient of the MATTER density averaged over two
    randomly placed beams of shape ``dims`` in a periodic box of side ``box_size``.

    These are pure geometry -- they depend on the beam shape, the box, and P(k), not on
    the galaxy sample.  Compute once per instrument and cache.

    Parameters
    ----------
    dims : (3,) array_like   beam side lengths [cMpc]
    pk_fn : callable         pk_fn(k, z) -> P(k) [cMpc^3], k in 1/cMpc
    box_size : float         periodic box side [cMpc]
    xi_beam_box : callable   linearity.xi_beam_box (passed in to avoid a hard import)
    """
    if xi_beam_box is None:
        from linearity import xi_beam_box as _x
        xi_beam_box = _x
    dims = np.asarray(dims, dtype=float)
    rng = np.random.default_rng(seed)
    seps = rng.random((n_pairs, 3)) * box_size
    xi = np.asarray(xi_beam_box(dims, seps, pk_fn, box_size, nmax=nmax))
    xi0 = float(np.asarray(xi_beam_box(dims, [[0.0, 0.0, 0.0]], pk_fn,
                                       box_size, nmax=nmax))[0])
    rho = xi / xi0
    return {
        "rho2": float(np.mean(rho ** 2)),
        "rho_p": float(np.mean(np.abs(rho) ** OFFDIAG_EXPONENT)),
        "sigma_delta": float(np.sqrt(xi0)),
    }


def _kappa4_counts(mu, sigma_cv):
    """C_ii and the connected 4th cumulant kappa_4 of counts, for a lognormal
    density of variance sigma_cv^2 Poisson-sampled with mean mu.

    Law of total cumulance with a Poisson inner law.  Returns (C_ii, kappa_4).
    """
    mu = float(mu)
    s2 = float(sigma_cv) ** 2
    if s2 <= 0.0:
        return mu, mu  # pure Poisson: kappa_4 = mu
    sg2 = np.log1p(s2)
    # central moments of the lognormal density d = exp(g - sg2/2) - 1
    e = lambda n: np.exp(0.5 * n * (n - 1.0) * sg2)  # E[(1+d)^n]
    m2 = e(2) - 1.0
    m3 = e(3) - 3.0 * e(2) + 2.0
    m4 = e(4) - 4.0 * e(3) + 6.0 * e(2) - 3.0
    k4_d = m4 - 3.0 * m2 ** 2  # connected 4th cumulant of the density
    C = mu + mu ** 2 * m2
    # cumulants of N = Poisson(mu(1+d)); K_n = sum over Stirling structure
    k4 = (mu + 7.0 * mu ** 2 * m2 + 6.0 * mu ** 3 * m3 + mu ** 4 * k4_d)
    return float(C), float(k4)


def var_sigma_mu2_analytic(counts, mu, geometry, sigma_cv=None):
    """Var[sigma_hat_mu^2] from the lognormal kappa_4 plus the fitted off-diagonal law.

    Parameters
    ----------
    counts : (M,) array_like    galaxy counts in the M beams (one M_UV bin)
    mu : float                  KNOWN mean count (from the box abundance x volume ratio)
    geometry : dict             output of beam_geometry_moments for this beam shape
    sigma_cv : float, optional  sigma_CV; if None it is estimated from ``counts``

    Returns
    -------
    dict with 'var' (the estimate), 'wick' (the Gaussian value, for comparison), and the
    three contributing terms.
    """
    counts = np.asarray(counts, dtype=float)
    M = counts.size
    mu = float(mu)
    if sigma_cv is None:
        s2 = (np.mean((counts - mu) ** 2) - mu) / mu ** 2
        sigma_cv = np.sqrt(max(s2, 0.0))

    C, k4 = _kappa4_counts(mu, sigma_cv)
    K_ii = 2.0 * C ** 2 + k4

    # counts inherit a fraction f of the matter correlation (shot noise is uncorrelated
    # between non-identical beams, so it dilutes rho)
    f = (mu ** 2 * sigma_cv ** 2) / C if C > 0 else 0.0
    # <|rho_counts|^p> = f^p <|rho_delta|^p>
    rho_p = (f ** OFFDIAG_EXPONENT) * geometry["rho_p"]
    K_ij = OFFDIAG_AMPLITUDE * K_ii * rho_p

    diag_wick = 2.0 * C ** 2 / M
    diag_k4 = k4 / M
    offdiag = (M - 1.0) / M * K_ij
    return {
        "var": diag_wick + diag_k4 + offdiag,
        "wick": diag_wick,
        "diag_wick": diag_wick,
        "diag_kappa4": diag_k4,
        "offdiag": offdiag,
        "sigma_cv": sigma_cv,
        "C_ii": C,
        "kappa4": k4,
        "clustering_fraction": f,
    }


def var_sigma_mu2_replacement(count_replicates, mu, beam_volume, box_volume,
                              f_ng=1.4, sigma_cv=None):
    """RECOMMENDED FOR A SINGLE BOX.  Measure the placement half, predict the box half.

    You supply many independent beam placements in your ONE box -- i.e. call your
    place_beams() with different seeds and re-run beam_count_matrix each time.  This
    function measures Var over those placements (the 48-66% of the total that a single box
    genuinely contains) and adds the analytic box-to-box term (the remaining 34-52%).

    Validated at M=100 on the real Lumina geometry: median 0.96x truth, 16-84% spread
    [0.75x, 1.38x].  The fully analytic route on the same test gave median 0.99x but a
    much wider [0.58x, 2.01x].  The gain comes from measuring the larger half instead of
    modelling it, and it drops the empirical rho^1.2 law entirely.

    Parameters
    ----------
    count_replicates : (n_place, M) array_like
        Counts for one M_UV bin, from n_place independent beam placements in the same box.
        Use n_place >= 100; the estimator corrects for finite n_place but the correction
        grows as n_place shrinks.
    mu : float               known mean count
    beam_volume : float      V_beam = prod(dims) [cMpc^3]
    box_volume : float       V_box = box_size**3 [cMpc^3]
    f_ng : float
        Non-Gaussian enhancement of Var_box[sigma^2].  Measured 1.21 / 1.34 / 1.69 for the
        M_UV = -19.5 / -20.0 / -20.5 bins (rises as mu falls).  Default 1.4.  This scales
        only the predicted third-to-half, so a 20% error here is <10% on the total.
    sigma_cv : float, optional
        sigma_CV for the box term; if None, estimated from the replicate mean.

    Returns
    -------
    dict with 'var', the two halves separately, and the finite-n_place correction applied.
    """
    Y = (np.asarray(count_replicates, dtype=float) - float(mu)) ** 2
    if Y.ndim != 2:
        raise ValueError("count_replicates must be 2-D (n_place, M); got shape %r" % (Y.shape,))
    n_place, M = Y.shape
    if n_place < 2:
        raise ValueError("need >= 2 placements; got %d" % n_place)
    mu = float(mu)

    per_place = Y.mean(axis=1)              # sigma_hat_mu^2 for each placement
    t1 = per_place.var(ddof=1)              # MEASURED placement term

    if sigma_cv is None:
        sigma_cv = np.sqrt(max((per_place.mean() - mu) / mu ** 2, 0.0))

    n_eff = float(box_volume) / float(beam_volume)
    t2_raw = mu ** 4 * f_ng * 2.0 * float(sigma_cv) ** 4 / n_eff

    # per-placement means carry residual placement noise t1/n_place, which would
    # double-count into the box term; subtract it (verified: restores 0.96-1.10x truth)
    t2 = max(t2_raw - t1 / n_place, 0.0)

    return {
        "var": t1 + t2,
        "placement_term": t1,
        "box_term": t2,
        "box_term_uncorrected": t2_raw,
        "placement_fraction": t1 / (t1 + t2) if (t1 + t2) > 0 else np.nan,
        "n_placements": int(n_place),
        "n_eff": n_eff,
        "sigma_cv": float(sigma_cv),
        "f_ng": float(f_ng),
    }


def pair_covariance_profile(counts, centers, mu, box_size, bin_edges=None):
    """Measure K22(r) = Cov(Y_i, Y_j) vs beam separation from ALL unique pairs in one box.

    Enumerates every unordered pair (i,j), i<j -- B(B-1)/2 of them -- bins by periodic
    centre separation, and estimates within each bin

        K22_hat(r) = <Y_i Y_j>_r  -  <Y_i>_r <Y_j>_r,      Y_i = (N_i - mu)^2

    WHAT THIS RECOVERS, AND WHAT IT DOES NOT
    ----------------------------------------
    Binning by separation escapes the integral constraint that kills the *unbinned* double
    sum (which is identically zero from one realisation -- see var_sigma_mu2_analytic's
    notes).  Validated against 250 independent boxes, 500 beam positions, M_UV = -20.0:
    the measured SHAPE is correct, and the whole error is a single additive constant,

        K22_hat(r)  =  K22_true(r)  -  Var_box[Ybar]

    constant to 7.4% across 0-433 cMpc while K22 itself varies by 50x and changes sign
    (measured offset -49.9 vs Var_box[Ybar] = 53.2, ratio 1.065).  That constant is the
    box-to-box term yet again: a single box cannot supply it, by any route.

    Use `offset` to add it back.  Uncorrected, the resulting Var[sigma_mu^2] is ~2x low
    (5.24e+01 vs a simulated truth of 1.06e+02); corrected it lands at 1.05e+02, ~1%.

    CAVEAT ON ERROR BARS: the pairs are unique but NOT independent -- beam i appears in
    B-1 of them.  B(B-1)/2 pairs is not B(B-1)/2 independent measurements.  Do not build
    an error bar on K22 by treating pairs as independent.

    Parameters
    ----------
    counts : (B,) array_like    counts in B beams, one M_UV bin
    centers : (B, 3) array_like beam centres [cMpc]
    mu : float                  known mean count
    box_size : float            periodic box side [cMpc]
    bin_edges : array_like, optional   separation bin edges [cMpc]

    Returns
    -------
    dict with 'r' (bin centres), 'K22' (uncorrected), 'n_pairs', and 'Ybar_var' (the
    single-box value of Var[Ybar]; NOT the correction -- see var_sigma_mu2_paircov).
    """
    Y = (np.asarray(counts, dtype=float) - float(mu)) ** 2
    ctr = np.asarray(centers, dtype=float)
    B = Y.size
    if ctr.shape != (B, 3):
        raise ValueError("centers must be (B,3) matching counts; got %r" % (ctr.shape,))
    if B < 4:
        raise ValueError("need >= 4 beams to bin pairs; got %d" % B)
    L = float(box_size)
    if bin_edges is None:
        bin_edges = np.array([0, 40, 80, 120, 160, 200, 250, 300, 350, np.sqrt(3) * L / 2])
    bin_edges = np.asarray(bin_edges, dtype=float)

    d = ctr[:, None, :] - ctr[None, :, :]
    d -= L * np.round(d / L)                       # periodic minimum image
    sep = np.sqrt((d ** 2).sum(-1))
    iu, ju = np.triu_indices(B, 1)                  # each unordered pair exactly once
    sep_p = sep[iu, ju]
    prod = Y[iu] * Y[ju]

    idx = np.digitize(sep_p, bin_edges) - 1
    nb = len(bin_edges) - 1
    K = np.full(nb, np.nan)
    npair = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = idx == b
        npair[b] = m.sum()
        if npair[b] >= 2:
            K[b] = prod[m].mean() - Y[iu][m].mean() * Y[ju][m].mean()

    return {
        "r": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "bin_edges": bin_edges,
        "K22": K,
        "n_pairs": npair,
        "n_pairs_total": int(len(sep_p)),
        "Ybar_var": float(Y.var(ddof=1) / B),
    }


def var_sigma_mu2_paircov(counts, centers, mu, box_size, beam_volume, box_volume,
                          M=None, f_ng=1.4, sigma_cv=None, bin_edges=None):
    """Var[sigma_mu^2] from the all-unique-pairs K22(r), with the constant offset restored.

    It MEASURES the off-diagonal covariance as a function of separation instead of assuming
    the empirical rho^1.2 law, and returns K22(r) alongside the variance so you can see the
    correlation die out (for the Lumina geometry, by r ~ 250 cMpc).

    ACCURACY -- READ BEFORE PREFERRING THIS ROUTE
    ---------------------------------------------
    Averaged over many boxes this route is excellent (1.05e+02 vs a truth of 1.06e+02).
    On ONE box it is not better than var_sigma_mu2_replacement, because K22 in each
    separation bin is itself noisy and sigma_CV^4 in the offset amplifies that noise.
    Measured over 250 boxes at M_UV = -20.0, B = 500 positions, M = 100:

        route                              median   16-84%
        var_sigma_mu2_paircov              1.12     [0.76, 1.77]
        ... with sigma_CV supplied         1.11     [0.94, 1.39]
        var_sigma_mu2_replacement          0.96     [0.75, 1.38]

    So: use this route for the DIAGNOSTIC K22(r) -- it is the only route that shows you the
    shape -- and for a headline number either use var_sigma_mu2_replacement, or pass a
    sigma_cv you trust (from the full box, not from the B beams) to tighten this one.

    Var[sigma_hat_mu^2] = K_ii / M  +  (M-1)/M * <K22_corrected>

    where <K22_corrected> is the pair-weighted mean of K22_hat(r) + offset, and

        offset = f_ng * 2 mu^4 sigma_CV^4 / n_eff,     n_eff = V_box / V_beam

    is the box-to-box term a single box cannot measure.  See pair_covariance_profile for
    the validation of the constant-offset structure.

    ON f_ng: measured 1.62 by this route at M_UV = -20.0, versus 1.34 by the re-placement
    route on the same bin.  The two disagree because Ybar over a finite beam set still
    carries residual placement noise, which inflates the estimate here.  Default 1.4 is a
    compromise; the spread implies roughly +/-20% on this term, which is a third to a half
    of the total, so ~10% on the final answer.  Quote it accordingly.

    Returns
    -------
    dict with 'var', the diagonal and off-diagonal pieces, the applied 'offset', and the
    full 'profile' from pair_covariance_profile.
    """
    counts = np.asarray(counts, dtype=float)
    mu = float(mu)
    B = counts.size
    if M is None:
        M = B
    M = float(M)

    prof = pair_covariance_profile(counts, centers, mu, box_size, bin_edges=bin_edges)

    if sigma_cv is None:
        sigma_cv = np.sqrt(max((np.mean((counts - mu) ** 2) - mu) / mu ** 2, 0.0))
    n_eff = float(box_volume) / float(beam_volume)
    offset = float(f_ng) * 2.0 * mu ** 4 * float(sigma_cv) ** 4 / n_eff

    ok = np.isfinite(prof["K22"]) & (prof["n_pairs"] > 0)
    w = prof["n_pairs"][ok].astype(float)
    K_bar = float(np.average(prof["K22"][ok] + offset, weights=w))

    K_ii = float(np.var((counts - mu) ** 2, ddof=1) + np.mean((counts - mu) ** 2) ** 2)
    diag = K_ii / M
    off = (M - 1.0) / M * K_bar

    return {
        "var": diag + off,
        "diag": diag,
        "offdiag": off,
        "K22_mean_corrected": K_bar,
        "offset": offset,
        "sigma_cv": float(sigma_cv),
        "n_eff": n_eff,
        "f_ng": float(f_ng),
        "profile": prof,
    }


def var_sigma_mu2_jackknife(counts, centers, mu, box_size, n_slabs=8, axis=0):
    """Delete-one-slab spatial jackknife.  Assumption-free but biased LOW.

    Splits the box into ``n_slabs`` slabs along ``axis``, drops the beams whose centres
    fall in each slab in turn, and takes the usual jackknife variance of sigma_hat_mu^2.

    Calibration at M=100 on the real Lumina geometry: median 0.55x, mean 0.83x the truth.
    A sub-volume cannot contain modes larger than itself, so this MISSES the box-scale
    contribution.  Treat the result as a lower bound.

    Parameters
    ----------
    counts : (M,) array_like     counts in one M_UV bin
    centers : (M, 3) array_like  beam centres [cMpc]
    mu : float                   known mean count
    box_size : float             periodic box side [cMpc]
    """
    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    Y = (counts - float(mu)) ** 2
    lab = np.clip((centers[:, axis] % box_size) / box_size * n_slabs, 0,
                  n_slabs - 1e-9).astype(int)
    vals = [Y[lab != g].mean() for g in range(n_slabs) if np.count_nonzero(lab != g) > 2]
    vals = np.asarray(vals, dtype=float)
    nj = vals.size
    if nj < 2:
        return {"var": np.nan, "n_used": nj}
    var = (nj - 1.0) / nj * np.sum((vals - vals.mean()) ** 2)
    return {"var": float(var), "n_used": nj,
            "note": "biased low, ~0.55-0.83x truth; lower bound only"}


def var_sigma_mu2_ensemble(counts_list, mu):
    """Var[sigma_hat_mu^2] across INDEPENDENT realisations.  Unbiased, no PDF assumption.

    ``counts_list`` must be realisations that differ in their long-wavelength modes --
    different boxes, or well-separated snapshots.  Re-placing beams inside one fixed box
    does NOT qualify (it returns ~0.56x the truth) because the box modes are held fixed.

    Parameters
    ----------
    counts_list : sequence of (M,) arrays   one entry per independent realisation
    mu : float                              known mean count
    """
    ests = np.array([np.mean((np.asarray(c, dtype=float) - float(mu)) ** 2)
                     for c in counts_list], dtype=float)
    n = ests.size
    if n < 2:
        raise ValueError("need >= 2 independent realisations; got %d" % n)
    var = ests.var(ddof=1)
    return {"var": float(var), "n_realisations": int(n),
            "var_error": float(var * np.sqrt(2.0 / (n - 1.0)))}


def sigma_cv_squared_error(var_sigma_mu2, mu):
    """Propagate Var[sigma_hat_mu^2] to the 1-sigma error on sigma_CV^2.

    sigma_CV^2 = (sigma_hat_mu^2 - mu) / mu^2 with mu known, so the Jacobian is 1/mu^2.
    """
    return float(np.sqrt(var_sigma_mu2) / float(mu) ** 2)
