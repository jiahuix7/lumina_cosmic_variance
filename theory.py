"""Linear theory for the Lumina cosmic-variance analysis.

Everything here is analytic or quadrature -- no simulation input. Two
independent routes to P(k) so neither is trusted alone:

  * CAMB (Boltzmann, exact)
  * Eisenstein & Hu 1998 fitting formula (no network, no dependency)

Conventions
-----------
Distances are comoving Mpc (NOT Mpc/h) throughout, matching
compute_UVLF_cv.py where box_size_cmpc = 500.0 and positions are divided by
(1000*h). Wavenumbers are therefore 1/Mpc. Internally CAMB works in h/Mpc;
the conversion happens once, in `linear_power`.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

# --------------------------------------------------------------------------
# Lumina cosmology -- Planck, methods paper Table 2. Mirrors
# compute_UVLF_cv.lumina_cosmology exactly.
# --------------------------------------------------------------------------
H0 = 67.66
h = H0 / 100.0
Om0 = 0.3096
Ob0 = 4.897 / 100.0
Oc0 = Om0 - Ob0
ns = 0.9665
sigma8 = 0.8102
Tcmb0 = 2.7255

RHO_CRIT_0 = 2.77536627e11 * h ** 2   # Msun / Mpc^3 (comoving)
RHOBAR_M = Om0 * RHO_CRIT_0           # comoving mean matter density
DELTA_C = 1.686

BOX_SIZE = 500.0                      # cMpc


# --------------------------------------------------------------------------
# Growth factor
# --------------------------------------------------------------------------
def growth_factor(z, Om=Om0):
    """Linear growth D(z), normalised to D(0) = 1, for flat LCDM.

    D(a) propto H(a) * Int_0^a da' / (a' H(a'))^3, evaluated by quadrature.
    """
    Ol = 1.0 - Om

    def E(a):
        return np.sqrt(Om / a ** 3 + Ol)

    def integrand(a):
        return 1.0 / (a * E(a)) ** 3

    def D_unnorm(a):
        val, _ = quad(integrand, 1e-8, a, limit=200)
        return E(a) * val

    z = np.atleast_1d(np.asarray(z, dtype=float))
    d0 = D_unnorm(1.0)
    out = np.array([D_unnorm(1.0 / (1.0 + zi)) / d0 for zi in z])
    return out[0] if out.size == 1 else out


# --------------------------------------------------------------------------
# Eisenstein & Hu 1998 transfer function (with baryon suppression, no wiggles)
# --------------------------------------------------------------------------
def transfer_eh98(k):
    """EH98 'no-wiggle' shape-with-baryons transfer function.

    Parameters
    ----------
    k : array, comoving 1/Mpc

    Returns T(k), normalised to T -> 1 as k -> 0.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    kh = k / h                      # h/Mpc, the units of the fitting formula
    theta = Tcmb0 / 2.7
    om = Om0 * h ** 2
    ob = Ob0 * h ** 2
    fb = ob / om

    # sound horizon (Eq. 26)
    s = 44.5 * np.log(9.83 / om) / np.sqrt(1.0 + 10.0 * ob ** 0.75)   # Mpc/h
    # alpha_Gamma (Eq. 31)
    alpha = (1.0 - 0.328 * np.log(431.0 * om) * fb
             + 0.38 * np.log(22.3 * om) * fb ** 2)
    # effective shape (Eq. 30)
    ks = kh * s
    gamma_eff = Om0 * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * ks) ** 4))
    q = kh * theta ** 2 / gamma_eff

    # Eq. 29
    C = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L = np.log(2.0 * np.e + 1.8 * q)
    T = L / (L + C * q ** 2)
    return T


# --------------------------------------------------------------------------
# Power spectra
# --------------------------------------------------------------------------
def _sigma_R_from_pk(k, pk, R):
    """sigma(R) for a spherical top-hat, by log-k quadrature."""
    kR = np.outer(np.atleast_1d(R), k)
    W = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR ** 3
    integrand = pk[None, :] * W ** 2 * k[None, :] ** 3 / (2.0 * np.pi ** 2)
    val = np.trapezoid(integrand, np.log(k), axis=1)
    out = np.sqrt(val)
    return out[0] if out.size == 1 else out


def linear_power_eh(k, z=0.0, sigma8_norm=sigma8):
    """EH98 linear P(k) [Mpc^3] at redshift z, normalised to sigma8."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    shape = k ** ns * transfer_eh98(k) ** 2

    # normalise on a fixed, well-resolved grid at z=0
    kn = np.logspace(-5, 3, 4000)
    pn = kn ** ns * transfer_eh98(kn) ** 2
    s8_raw = _sigma_R_from_pk(kn, pn, 8.0 / h)
    A = (sigma8_norm / s8_raw) ** 2

    return A * shape * growth_factor(z) ** 2


def linear_power_camb(k, z=0.0):
    """CAMB linear P(k) [Mpc^3] at redshift z. k in comoving 1/Mpc."""
    import camb
    k = np.atleast_1d(np.asarray(k, dtype=float))
    pars = camb.set_params(
        H0=H0, ombh2=Ob0 * h ** 2, omch2=Oc0 * h ** 2,
        ns=ns, TCMB=Tcmb0, mnu=0.0, omk=0.0, nnu=3.046,
    )
    pars.set_matter_power(redshifts=[z], kmax=2000.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    # CAMB returns P in (Mpc/h)^3 vs k in h/Mpc
    kh, _, pkh = res.get_matter_power_spectrum(
        minkh=1e-5, maxkh=1e3, npoints=1000)
    s8 = res.get_sigma8()[-1]
    k_mpc = kh * h
    p_mpc = pkh[0] / h ** 3
    spl = InterpolatedUnivariateSpline(np.log(k_mpc), np.log(p_mpc), k=3)
    return np.exp(spl(np.log(k))), float(s8)


def sigma_R(R, z=0.0, pk_fn=None):
    """sigma(R,z) for a spherical top-hat of comoving radius R [Mpc]."""
    kn = np.logspace(-5, 3, 4000)
    pn = linear_power_eh(kn, z) if pk_fn is None else pk_fn(kn, z)
    return _sigma_R_from_pk(kn, pn, R)


def lagrangian_radius(M):
    """Comoving Lagrangian radius [Mpc] of a halo of mass M [Msun]."""
    M = np.asarray(M, dtype=float)
    return (3.0 * M / (4.0 * np.pi * RHOBAR_M)) ** (1.0 / 3.0)


def lagrangian_mass(R):
    """Inverse of lagrangian_radius: mass [Msun] enclosed in radius R [Mpc]."""
    R = np.asarray(R, dtype=float)
    return 4.0 / 3.0 * np.pi * R ** 3 * RHOBAR_M


def nonlinear_scale(z, pk_fn=None):
    """R_NL [Mpc] where sigma(R,z) = 1, by bisection."""
    from scipy.optimize import brentq
    f = lambda lr: sigma_R(np.exp(lr), z, pk_fn) - 1.0
    lo, hi = np.log(1e-3), np.log(50.0)
    if f(lo) < 0:
        return np.nan
    return float(np.exp(brentq(f, lo, hi, xtol=1e-6)))


def n_eff(k0, z=0.0, pk_fn=None):
    """Logarithmic slope dlnP/dlnk at k0 [1/Mpc]."""
    fn = (lambda kk: linear_power_eh(kk, z)) if pk_fn is None else (lambda kk: pk_fn(kk, z))
    dl = 0.01
    kp, km = k0 * np.exp(dl), k0 * np.exp(-dl)
    return float((np.log(fn(kp)) - np.log(fn(km))) / (2 * dl))
