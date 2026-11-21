"""Linearity diagnostics for the Lumina pencil-beam cosmic-variance analysis.

Companion to `pencilbeam.py`. Answers three separate questions that are easy
to conflate under the single phrase "counts scale linearly":

  1. Is the *matter* field linear on the beam scale?      -> scale_ladder()
  2. Is the *galaxy* field linear in delta?               -> bias_correction()
  3. Is the *error bar* on sigma_mu^2 Gaussian (Wick)?    -> kappa4_counts()

Result summary (z=10, Lumina cosmology, see derivation.md):

  1. Yes, by 3 decades. R_NL(z=10) = 0.012 cMpc; the NIRCam beam is
     12.4 x 6.2 x 369 cMpc. Not the assumption that breaks.
  2. Marginal-to-broken. The expansion parameter is b1*sigma_delta, which
     is *identically* sigma_CV. At sigma_CV = 0.68 the second-order term
     is ~30% of sigma_CV^2.
  3. Broken, and this is the one that matters. At sigma_CV = 0.68 the
     connected term makes Var[sigma_mu^2] 6.3x larger than Wick predicts,
     i.e. the error bar from `sigma_mu_squared_error` is 2.5x too small.

Everything here is analytic or quadrature; no simulation input required.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = [
    "sigma_cuboid", "xi_beam_box", "beam_dims_from_survey",
    "bias_correction", "kappa4_counts", "var_sigma_mu2",
    "error_inflation", "scale_ladder",
]

DELTA_C = 1.686


# --------------------------------------------------------------------------
# window integrals
# --------------------------------------------------------------------------

def sigma_cuboid(dims, pk_fn, z=10.0, n=140, kmin=1e-4, kmax=60.0):
    """RMS density contrast in an axis-aligned cuboid of side lengths `dims`.

        sigma^2 = Int d3k/(2pi)^3 P(k) |W(k)|^2,  W = prod_d sinc(k_d L_d / 2)

    3D Gauss-Legendre in log k over the positive octant (x8 by symmetry).
    Converged to 5 significant figures at n=140 for the Lumina beams.

    Do NOT substitute an equal-volume sphere: for the NIRCam beam that
    overestimates sigma by 1.87x (equal-volume cube: 1.83x). The beam is
    12.4 x 6.2 x 369 cMpc -- a 60:1 axis ratio -- and the sphere
    approximation is only sane for quasi-cubic volumes.

    Parameters
    ----------
    dims : (3,) array_like -- comoving side lengths [cMpc]
    pk_fn : callable (k, z) -> P(k) [cMpc^3], k in 1/cMpc
    """
    L = np.asarray(dims, dtype=float)
    x, w = leggauss(n)
    lo, hi = np.log(kmin), np.log(kmax)
    lk = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    wk = 0.5 * (hi - lo) * w
    kg = np.exp(lk)
    KX, KY, KZ = np.meshgrid(kg, kg, kg, indexing="ij")
    WX, WY, WZ = np.meshgrid(wk * kg, wk * kg, wk * kg, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    W = (np.sinc(KX * L[0] / (2 * np.pi))
         * np.sinc(KY * L[1] / (2 * np.pi))
         * np.sinc(KZ * L[2] / (2 * np.pi)))
    integ = pk_fn(kmag.ravel(), z).reshape(kmag.shape) * W**2
    return float(np.sqrt(8.0 * np.sum(integ * WX * WY * WZ) / (2 * np.pi)**3))


def xi_beam_box(dims, rvecs, pk_fn, box_size, nmax=90, z=10.0):
    """Cross-covariance of the beam-smoothed density between two identical
    cuboids offset by `rvecs`, in a *periodic box* (exact discrete mode sum).

        xi_beam(r) = (1/V) sum_{k != 0} P(k) |W(k)|^2 cos(k.r)

    At r=0 this returns sigma_delta^2 including the box's missing-mode
    deficit. For a 500 cMpc box at z=10 the deficit is negligible for
    NIRCam (retains 100.8% -- discreteness slightly overshoots) and small
    for Roman (97.6%).

    Use this rather than a continuum integral when comparing against
    measurements made *in* the box: a naive per-axis k_min cutoff is wrong,
    because a periodic box does contain modes with individual components
    equal to zero (e.g. k = (0, 0, k_f)).
    """
    L = np.asarray(dims, float)
    kf = 2 * np.pi / box_size
    V = box_size**3
    R = np.atleast_2d(np.asarray(rvecs, float))
    n = np.arange(-nmax, nmax + 1)
    kx = n * kf
    Wx = np.sinc(kx * L[0] / (2 * np.pi))**2
    Wy = np.sinc(kx * L[1] / (2 * np.pi))**2
    acc = np.zeros(len(R))
    for nz in range(-nmax, nmax + 1):
        kz = nz * kf
        Wz = np.sinc(kz * L[2] / (2 * np.pi))**2
        kmag = np.sqrt(kx[:, None]**2 + kx[None, :]**2 + kz**2)
        w = Wx[:, None] * Wy[None, :] * Wz
        if nz == 0:                      # excise the k=0 mode
            kmag = kmag.copy(); kmag[nmax, nmax] = 1.0
            w = w.copy();       w[nmax, nmax] = 0.0
        pw = pk_fn(kmag.ravel(), z).reshape(kmag.shape) * w
        m = pw > pw.max() * 1e-10
        if not m.any():
            continue
        KX = np.broadcast_to(kx[:, None], pw.shape)[m]
        KY = np.broadcast_to(kx[None, :], pw.shape)[m]
        acc += (pw[m][None, :] * np.cos(KX[None, :] * R[:, 0:1]
                                        + KY[None, :] * R[:, 1:2]
                                        + kz * R[:, 2:3])).sum(1)
    return acc / V


def beam_dims_from_survey(cosmo, z_min, z_max, width, height):
    """Comoving cuboid dimensions of a pencil beam.

    `width`/`height` are astropy angular Quantities. Returns [Lx, Ly, Lz]
    in cMpc, matching the convention in `compute_UVLF_cv.py`.
    """
    import astropy.units as u
    z_c = 0.5 * (z_min + z_max)
    d_a = cosmo.comoving_transverse_distance(z_c).to_value(u.Mpc)
    depth = (cosmo.comoving_distance(z_max)
             - cosmo.comoving_distance(z_min)).to_value(u.Mpc)
    return np.array([d_a * width.to_value(u.rad),
                     d_a * height.to_value(u.rad), depth])


# --------------------------------------------------------------------------
# assumption 2: linear bias
# --------------------------------------------------------------------------

def b2_lazeyras(b1):
    """Second-order bias from the b2(b1) relation of Lazeyras et al. 2016.

    WARNING: calibrated over b1 ~ 1-10. High-z UV-selected galaxies sit at
    b1 ~ 10-50, so this is extrapolation and the returned correction should
    be read as an order-of-magnitude diagnostic, not a correction to apply.
    """
    return 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3


def bias_correction(b1, sigma_delta, n_eff):
    """Fractional second-order correction to sigma_CV^2.

    For delta_g = b1 delta + (b2/2)(delta^2 - sigma^2):

        Var[delta_g]/(b1 sigma)^2 - 1 = (b2/b1) S3 sigma^2
                                        + (b2^2/2 b1^2) sigma^2

    with the tree-level skewness S3 = 34/7 - (n_eff + 3).

    Returns (term1, term2, total). The expansion has *ceased to converge*
    once term2 > term1, which for the NIRCam beam happens at b1 = 10.3 --
    i.e. right where these galaxies live.
    """
    b2 = b2_lazeyras(b1)
    s3 = 34.0 / 7.0 - (n_eff + 3.0)
    t1 = (b2 / b1) * s3 * sigma_delta**2
    t2 = (b2**2 / (2 * b1**2)) * sigma_delta**2
    return t1, t2, t1 + t2


# --------------------------------------------------------------------------
# assumption 3: Wick / Gaussianity  -- the one that actually bites
# --------------------------------------------------------------------------

def kappa4_counts(mu, sigma_cv):
    """Connected 4th cumulant of beam counts N ~ Poisson(mu (1+delta)),
    for a lognormal delta with Var[delta] = sigma_cv^2.

    Derived by the law of total cumulance with a Poisson inner law.

    CORRECTED 2026-08-12. The previous version carried two spurious terms
    (+6 mu^2 m2 + 4 mu^3 m3) which inflated kappa_4 by 32-51% at low sigma_cv.
    The bug hid because the original check was run only at sigma_cv ~ 0.7-1.2,
    where the mu^4 k4d term dominates and the error is a mere 3%; the same
    expression is 51% high at sigma_cv = 0.28. Conclusions drawn at
    sigma_cv >= 0.68 are unaffected; anything at sigma_cv < 0.4 must be redone.
    Re-verified against 6e7 direct samples: agrees to 0.3% over
    sigma_cv = 0.185-0.681, mu = 8.4-68.

    Beware of validating a heavy-tailed 4th moment at one amplitude only --
    test across the range you intend to use.

    Returns (C, kappa4) where C = Var[N] = mu + mu^2 sigma_cv^2.
    """
    s = sigma_cv
    u = 1.0 + s**2                       # = exp(sigma_G^2)
    m2 = s**2
    m3 = u**3 - 3 * u + 2
    k4d = (u**6 - 4 * u**3 + 6 * u - 3) - 3 * m2**2
    C = mu + mu**2 * m2
    k4 = mu + 7 * mu**2 * m2 + 6 * mu**3 * m3 + mu**4 * k4d
    return C, k4


def var_sigma_mu2(mu, sigma_cv, M, wick_only=False):
    """Var[sigma_mu^2] for M independent beams, known mean.

        Wick:  2 C^2 / M
        full:  (2 C^2 + kappa_4) / M

    The `wick_only` branch is what `pencilbeam.sigma_mu_squared_error`
    currently implements. Compare the two with `error_inflation`.
    """
    C, k4 = kappa4_counts(mu, sigma_cv)
    return (2 * C**2 / M) if wick_only else ((2 * C**2 + k4) / M)


def error_inflation(mu, sigma_cv):
    """Factor by which the true error on sigma_mu^2 exceeds the Wick value.

        sqrt(1 + kappa_4 / (2 C^2))

    Independent of M. Validated against 4000 direct realisations of the
    estimator: the kappa_4 formula recovers the measured variance to
    better than 5% for sigma_cv <~ 0.7, degrading to 0.58 at sigma_cv=1.71
    where the estimator distribution is too heavy-tailed for a 4th-cumulant
    truncation.

    Reference values (mu = 68), analytic:
        sigma_cv   inflation      status
          0.10       1.03       Wick fine
          0.20       1.15
          0.40       1.58
          0.68       2.56       <- Lumina mock
          1.00       4.46
          1.71      14.09       <- PANORAMIC faintest bin

    Note: a 8e6-sample Monte Carlo gives ~13.0 at sigma_cv=1.71. That is
    MC under-convergence of a 4th moment with a very heavy tail, not a
    disagreement -- the analytic value is the reliable one. Sampling error
    on kappa_4 grows steeply with sigma_cv; do not "verify" this branch
    with a small MC and conclude the formula is wrong.
    """
    C, k4 = kappa4_counts(mu, sigma_cv)
    return float(np.sqrt(1.0 + k4 / (2 * C**2)))


# --------------------------------------------------------------------------
# assumption 1: linear growth
# --------------------------------------------------------------------------

def scale_ladder(dims, sigma_R_fn, lagrangian_radius_fn, masses=(1e9, 1e13)):
    """Compare beam size against R_NL and halo Lagrangian radii.

    Returns a dict of the quantities that make up `scale_ladder.png`.
    For the Lumina beams this test passes by 3 decades and is *not* the
    constraint -- it is included to document that fact rather than to gate
    anything.
    """
    from scipy.optimize import brentq
    r_nl = np.exp(brentq(lambda lr: sigma_R_fn(np.exp(lr)) - 1.0,
                         np.log(1e-4), np.log(50.0), xtol=1e-10))
    L = np.asarray(dims, float)
    r_lag = [float(lagrangian_radius_fn(m)) for m in masses]
    v_beam = float(np.prod(L))
    return {
        "R_NL": float(r_nl),
        "beam_dims": L.tolist(),
        "beam_volume": v_beam,
        "beam_min_side": float(L.min()),
        "R_lagrangian": r_lag,
        "min_side_over_R_NL": float(L.min() / r_nl),
        "volume_over_lagrangian": [v_beam / (4 / 3 * np.pi * r**3) for r in r_lag],
    }
