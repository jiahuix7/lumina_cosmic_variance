"""Tests for linearity.py. Run: pytest test_linearity.py -q"""

import numpy as np
import pytest

from linearity import (
    sigma_cuboid, xi_beam_box, kappa4_counts, var_sigma_mu2,
    error_inflation, bias_correction, b2_lazeyras,
)

# A scale-free P(k) = A k^n with an exponential cutoff: analytic-ish and cheap.
A_PL, N_PL, KC = 1.0e3, -2.0, 5.0


def pk_powerlaw(k, z=0.0):
    k = np.atleast_1d(np.asarray(k, float))
    return A_PL * np.where(k > 0, k, 1e-30)**N_PL * np.exp(-(k / KC)**2)


# ---------------------------------------------------------------- windows

def test_sigma_cuboid_converges():
    """The n^-2 test spectrum is far redder than CAMB, so it converges more
    slowly; the real Lumina P(k) reaches 5 significant figures by n=140."""
    dims = [12.4, 6.2, 369.0]
    vals = [sigma_cuboid(dims, pk_powerlaw, n=n) for n in (120, 200, 280)]
    assert abs(vals[2] - vals[1]) / vals[2] < 3e-3


def test_sigma_cuboid_matches_realspace_monte_carlo():
    """Independent route: sigma^2 = <xi(|x-y|)> over the beam volume."""
    dims = np.array([20.0, 20.0, 40.0])
    kg = np.logspace(-4, np.log10(60.0), 8000)
    pk = pk_powerlaw(kg)
    rng = np.random.default_rng(0)
    x = rng.random((400_000, 3)) * dims
    y = rng.random((400_000, 3)) * dims
    r = np.linalg.norm(x - y, axis=1)
    # xi(r) by direct quadrature, per sampled separation binned for speed
    rb = np.logspace(-3, np.log10(np.sqrt((dims**2).sum())), 200)
    xib = np.array([np.trapezoid(pk * kg**3 * np.sinc(kg * rr / np.pi)
                                 / (2 * np.pi**2), np.log(kg)) for rb_i, rr
                    in enumerate(rb)])
    xi_mc = np.interp(r, rb, xib).mean()
    assert np.isclose(np.sqrt(xi_mc), sigma_cuboid(dims, pk_powerlaw, n=140),
                      rtol=0.05)


def test_sphere_approximation_is_bad_for_pencil_beams():
    """The documented 1.8x failure of the equal-volume sphere."""
    beam = np.array([12.4, 6.2, 369.0])
    v = beam.prod()
    cube = np.full(3, v**(1 / 3))
    assert sigma_cuboid(cube, pk_powerlaw) / sigma_cuboid(beam, pk_powerlaw) > 1.5


def test_xi_beam_box_at_zero_is_the_variance():
    dims = [20.0, 20.0, 40.0]
    xi0 = xi_beam_box(dims, [[0, 0, 0]], pk_powerlaw, 500.0, nmax=60)[0]
    sig = sigma_cuboid(dims, pk_powerlaw, n=140)
    assert np.isclose(np.sqrt(xi0), sig, rtol=0.05)


def test_xi_beam_box_decays_with_separation():
    dims = [12.4, 6.2, 369.0]
    r = np.array([[0, 0, 0], [30.0, 0, 0], [200.0, 0, 0]])
    xi = xi_beam_box(dims, r, pk_powerlaw, 500.0, nmax=60)
    assert xi[0] > xi[1] > 0
    assert abs(xi[2]) < xi[0]


# ---------------------------------------------------------------- kappa_4

@pytest.mark.parametrize("sig,rtol", [(0.2, 0.35), (0.5, 0.25), (0.9, 0.25)])
def test_kappa4_matches_direct_sampling(sig, rtol):
    """MC estimation of a 4th cumulant is noisy -- the tolerance is set by
    sampling error, not by the accuracy of the expression. At small sigma
    kappa_4 is a small difference of large numbers, so it is noisiest there."""
    mu = 60.0
    rng = np.random.default_rng(4)
    sg = np.sqrt(np.log1p(sig**2))
    lam = mu * np.exp(rng.standard_normal(4_000_000) * sg - 0.5 * sg**2)
    n = rng.poisson(lam).astype(float)
    c_sim = n.var()
    k4_sim = np.mean((n - n.mean())**4) - 3 * c_sim**2
    c, k4 = kappa4_counts(mu, sig)
    assert np.isclose(c, c_sim, rtol=0.02)
    assert np.isclose(k4, k4_sim, rtol=rtol)


def test_kappa4_vanishes_in_the_poisson_limit():
    """With no clustering, kappa_4 of N is exactly mu (pure Poisson)."""
    c, k4 = kappa4_counts(80.0, 0.0)
    assert np.isclose(c, 80.0)
    assert np.isclose(k4, 80.0)


def test_error_inflation_is_monotonic_and_matches_reference():
    sig = np.array([0.1, 0.4, 0.68, 1.0, 1.71])
    infl = np.array([error_inflation(68.0, s) for s in sig])
    assert np.all(np.diff(infl) > 0)
    # values quoted in the module docstring and derivation.md
    assert np.isclose(infl[2], 2.56, rtol=0.02)
    assert np.isclose(infl[4], 14.09, rtol=0.02)


def test_wick_underpredicts_variance_by_the_documented_factor():
    """The headline number: 6.3x at the Lumina mock's sigma_CV."""
    vw = var_sigma_mu2(68.0, 0.68, 400, wick_only=True)
    vf = var_sigma_mu2(68.0, 0.68, 400, wick_only=False)
    assert np.isclose(vf / vw, 6.4, rtol=0.05)


def test_var_scales_as_one_over_M():
    a = var_sigma_mu2(68.0, 0.5, 100)
    b = var_sigma_mu2(68.0, 0.5, 400)
    assert np.isclose(a / b, 4.0, rtol=1e-10)


def test_direct_estimator_variance_matches_kappa4_formula():
    """End-to-end: simulate the estimator, compare to the corrected formula."""
    mu, sig, M = 68.0, 0.5, 400
    rng = np.random.default_rng(23)
    sg = np.sqrt(np.log1p(sig**2))
    lam = mu * np.exp(rng.standard_normal((4000, M)) * sg - 0.5 * sg**2)
    n = rng.poisson(lam).astype(float)
    measured = ((n - mu)**2).mean(1).var()
    assert np.isclose(measured, var_sigma_mu2(mu, sig, M), rtol=0.15)
    assert measured / var_sigma_mu2(mu, sig, M, wick_only=True) > 2.0


# ---------------------------------------------------------------- bias

def test_bias_expansion_breaks_down_near_b1_ten():
    """term2 overtakes term1 -- the series stops converging."""
    sig, neff = 0.0356, -2.31
    _, _, lo = bias_correction(5.0, sig, neff)
    t1_hi, t2_hi, _ = bias_correction(20.0, sig, neff)
    assert lo < 0.05
    assert t2_hi > t1_hi


def test_bias_correction_is_negligible_for_low_bias_tracers():
    """b2_lazeyras(1) = -0.79, not 0, so the correction is small but nonzero.
    What matters is that it is sub-percent at b1 ~ 1 and blows up by b1 ~ 20."""
    _, _, tot = bias_correction(1.0, 0.0356, -2.31)
    assert abs(tot) < 0.01
    assert abs(bias_correction(20.0, 0.0356, -2.31)[2]) > 30 * abs(tot)


def test_b2_relation_reference_values():
    assert b2_lazeyras(1.0) == pytest.approx(-0.794, abs=0.01)
    assert b2_lazeyras(2.0) < 0.5
    assert b2_lazeyras(10.0) > 50.0
