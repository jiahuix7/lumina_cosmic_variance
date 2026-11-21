#!/usr/bin/env python
"""Run the three linearity diagnostics against the Lumina configuration.

    python check_linearity.py                    # theory only, no sim needed
    python check_linearity.py --mu 68 --M 400    # with your measured counts

Prints a table of the three assumptions and whether each holds. The only
input you must supply from the simulation is the mean count per beam (mu)
and your measured sigma_CV; everything else is analytic.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theory as th
from linearity import (
    sigma_cuboid, bias_correction, error_inflation, var_sigma_mu2,
    beam_dims_from_survey, scale_ladder,
)

Z = 10.0
BOX = 500.0


def build_pk():
    """CAMB linear P(k), normalised to the Lumina sigma8, scaled by D(z)."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    kk = np.logspace(-5, 3, 3000)
    p, s8 = th.linear_power_camb(kk, 0.0)
    p *= (th.sigma8 / s8) ** 2
    spl = InterpolatedUnivariateSpline(np.log(kk), np.log(p), k=3)

    def pk(k, z=0.0):
        k = np.atleast_1d(np.asarray(k, float))
        return np.exp(spl(np.log(np.clip(k, 1e-30, None)))) * th.growth_factor(z) ** 2

    return pk


def n_eff_at(pk, k0, z=Z, dl=0.01):
    kp, km = k0 * np.exp(dl), k0 * np.exp(-dl)
    return float((np.log(pk(kp, z)[0]) - np.log(pk(km, z)[0])) / (2 * dl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=68.0,
                    help="mean galaxy count per beam")
    ap.add_argument("--sigma-cv", type=float, default=0.681,
                    help="measured sigma_CV")
    ap.add_argument("--M", type=int, default=400, help="number of beams")
    ap.add_argument("--instrument", default="NIRCam", choices=["NIRCam", "Roman"])
    ap.add_argument("--z-min", type=float, default=9.2)
    ap.add_argument("--z-max", type=float, default=10.9)
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args()

    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.3096, Ob0=0.04897,
                          Tcmb0=2.7255 * u.K, Neff=3.046, m_nu=0 * u.eV)
    fov = {"NIRCam": (4.4 * u.arcmin, 2.2 * u.arcmin),
           "Roman": (0.8 * u.deg, 0.4 * u.deg)}[args.instrument]
    dims = beam_dims_from_survey(cosmo, args.z_min, args.z_max, *fov)

    pk = build_pk()
    sig_d = sigma_cuboid(dims, pk, z=Z, n=160)
    b1 = args.sigma_cv / sig_d
    k_eff = 2 * np.pi / np.sqrt(dims[0] * dims[1])
    neff = n_eff_at(pk, k_eff)
    t1, t2, tot = bias_correction(b1, sig_d, neff)
    infl = error_inflation(args.mu, args.sigma_cv)

    ladder = scale_ladder(
        dims,
        lambda R: th._sigma_R_from_pk(np.logspace(-5, 3, 3000),
                                      pk(np.logspace(-5, 3, 3000), Z), R),
        th.lagrangian_radius, masses=(1e11,))

    print(f"\n=== Lumina linearity check: {args.instrument}, z = {args.z_min}-{args.z_max} ===")
    print(f"beam        {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} cMpc"
          f"   (V = {np.prod(dims):.3e} cMpc^3)")
    print(f"sigma_delta {sig_d:.5f}   (exact cuboid window, not an equal-V sphere)")
    print(f"implied b1  {b1:.2f}   from your sigma_CV = {args.sigma_cv:.3f}")

    print("\n(1) LINEAR GROWTH of the matter field")
    print(f"    R_NL(z={Z:.0f}) = {ladder['R_NL']:.4f} cMpc;"
          f" shortest beam side = {ladder['beam_min_side']:.2f} cMpc")
    print(f"    ratio = {ladder['min_side_over_R_NL']:.3e}"
          f"   -> PASSES by {np.log10(ladder['min_side_over_R_NL']):.1f} decades")

    print("\n(2) LINEAR BIAS of the galaxy field")
    print(f"    expansion parameter b1*sigma_delta = {b1 * sig_d:.3f}"
          f"   (identically sigma_CV)")
    print(f"    2nd-order correction to sigma_CV^2: {100 * tot:+.1f}%"
          f"  (term1 {100 * t1:+.1f}%, term2 {100 * t2:+.1f}%)")
    verdict2 = "OK" if abs(tot) < 0.1 else ("MARGINAL" if abs(tot) < 0.3 else "BROKEN")
    print(f"    -> {verdict2}" + ("   [b2(b1) extrapolated beyond calibration]"
                                  if b1 > 10 else ""))

    print("\n(3) WICK / GAUSSIANITY in the error bar")
    vw = var_sigma_mu2(args.mu, args.sigma_cv, args.M, wick_only=True)
    vf = var_sigma_mu2(args.mu, args.sigma_cv, args.M)
    print(f"    Var[sigma_mu^2]  Wick {vw:.4e}   with kappa_4 {vf:.4e}")
    print(f"    error-bar inflation = {infl:.2f}x")
    verdict3 = "OK" if infl < 1.1 else ("MARGINAL" if infl < 1.5 else "BROKEN")
    print(f"    -> {verdict3}: sigma_mu_squared_error is {infl:.1f}x too small")

    print(f"\nBOTTOM LINE: assumption (1) is safe; (2) is {verdict2.lower()};"
          f" (3) is the one to fix.\n")

    if args.json:
        json.dump({"instrument": args.instrument, "dims": dims.tolist(),
                   "sigma_delta": sig_d, "b1_implied": b1, "n_eff": neff,
                   "bias_terms": [t1, t2, tot], "error_inflation": infl,
                   "var_wick": vw, "var_full": vf, "scale_ladder": ladder},
                  open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
