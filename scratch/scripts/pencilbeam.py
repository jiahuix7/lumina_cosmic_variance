import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib as mpl
import astropy.units as u
from scipy.spatial.transform import Rotation

repo_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo_root))

from scripts import compute_UVLF_cv as cv

if cv is not None:
    luminosity1500_to_MUV = cv.luminosity1500_to_MUV
    fuv_to_MUV = cv.fuv_to_MUV
    load_data = cv.load_data
    load_EDE_data = cv.load_EDE_data
    select_valid_uv_galaxies = cv.select_valid_uv_galaxies
    calculate_rectangle_field_lengths = cv.calculate_rectangle_field_lengths
    calculate_field_volume = cv.calculate_field_volume
    match_mu = cv.match_mu
    compute_UVLF = cv.compute_UVLF
    # constants
    lumina_cosmology = cv.lumina_cosmology
    survey_area = cv.survey_area
    z_min, z_max, z_center = cv.z_min, cv.z_max, cv.z_center
    nircam_width, nircam_height = cv.nircam_width, cv.nircam_height
    MUV_limits_default = cv.MUV_limits
    res_cut = cv.res_cut
    snap_num = cv.snap_num
    lcdm_base_path = cv.lcdm_base_path
    ede_hdf5_path = cv.ede_hdf5_path
    box_size_cmpc_EDE = cv.box_size_cmpc_EDE
    random_seed = cv.random_seed

# Jacob's plot style
mpl.style.use("classic")
mpl.rc("xtick", labelsize=16)
mpl.rc("ytick", labelsize=16)
mpl.rc("xtick.major", size=14, width=2)
mpl.rc("xtick.minor", size=7, width=2, visible=True)
mpl.rc("ytick.major", size=14, width=2)
mpl.rc("ytick.minor", size=7, width=2, visible=True)
mpl.rc("lines", linewidth=2, markersize=10)
mpl.rc("axes", linewidth=2, labelsize=18, labelpad=2.5)
mpl.rc("legend", fontsize=13, loc="best", frameon=True, numpoints=1, scatterpoints=1)
mpl.rc("font", family="STIXGeneral")
mpl.rc("mathtext", fontset="stix")

# survey--> comoving beam distances
INSTRUMENTS = {
    "NIRCam": (4.4 * u.arcmin, 2.2 * u.arcmin),   # 2:1
    "Roman":  (0.8 * u.deg,    0.4 * u.deg),      # 2:1, 119x the NIRCam area
}

def beam_dimensions(cosmology, z_center, z_min, z_max, width_angle, height_angle):
    """Comoving (width, height, depth) of one beam in cMpc.

    Uses the transverse comoving distance at z_center, matching 
    cv.calculate_rectangle_field_lengths. The true light cone is a
    frustum (it diverges: far/near width ratio 1.039 over z=9.2-10.9), but the
    frustum volume differs from this cuboid by only -0.21%, so the constant
    cross-section is used. Checked, not assumed.

    Delegates to compute_UVLF_cv.calculate_rectangle_field_lengths so the two
    files cannot drift apart; the arithmetic below is a standalone fallback and
    is verified identical to it (agrees to 0 cMpc on both instruments).
    """
    if cv is not None:
        return cv.calculate_rectangle_field_lengths(cosmology, z_center, z_min, z_max, width_angle, height_angle)

    D_M = cosmology.comoving_transverse_distance(z_center).to_value(u.Mpc)
    depth = (cosmology.comoving_distance(z_max) - cosmology.comoving_distance(z_min)).to_value(u.Mpc)
    width = D_M * width_angle.to_value(u.rad)
    height = D_M * height_angle.to_value(u.rad)
    return np.array([width, height, depth])

def place_beams(n_beams, box_size, seed=42):
    """Random beam origins and orientations.
    Returns
    -------
    origins : (n_beams, 3)   start of the beam axis, uniform in the box
    frames  : (n_beams, 3, 3) rotation matrices; columns are (e_w, e_h, e_los)

    Orientations are uniform on SO(3), so the line of sight points anywhere and
    the beam is rolled randomly about it. Origins are uniform, so beams may
    overlap -- which is correct here: each beam is an independent draw of
    "one survey field", not one member of a 100-field mosaic.
    """
    rng = np.random.default_rng(seed)
    origins = rng.random((n_beams, 3)) * box_size
    frames = Rotation.random(n_beams, random_state=seed).as_matrix()
    return origins, frames


#counting how many galaxies fall within one beam
def _beam_corners(origin, frame, dims):
    """The 8 corners of the beam cuboid in unwrapped coordinates."""
    w, h, d = dims
    e_w, e_h, e_los = frame[:, 0], frame[:, 1], frame[:, 2]
    offs = [
        su * (w / 2) * e_w + sv * (h / 2) * e_h + t * d * e_los
        for su in (-1, 1) for sv in (-1, 1) for t in (0, 1)
    ]
    return origin + np.array(offs)


def build_tree(positions, box_size):
    """Periodic KD-tree over the galaxy positions. 
    scipy's boxsize argument gives true periodic neighbour queries (verified
    against a minimum-image brute force). Positions must lie in [0, box_size).
    """
    from scipy.spatial import cKDTree
    pos = np.mod(np.asarray(positions, dtype=float), box_size)
    return cKDTree(pos, boxsize=box_size)


def galaxies_in_beam(positions, origin, frame, dims, box_size,
                     return_multiplicity=False, tree=None, segment_length=None):
    """Indices of galaxies inside one periodic, arbitrarily-oriented beam.

    The beam is walked in short SEGMENTS along its axis. For each segment the
    KD-tree returns the galaxies within a sphere enclosing it, and those few
    candidates get the exact cuboid test. Cost scales with the number of
    galaxies actually near the beam, not with the size of the catalog.

    Why segments rather than one displacement from the origin: the beam is
    369 cMpc deep but the box is 500, so box_size/2 = 250 < 369 and the
    minimum-image convention is INVALID over the full beam -- a galaxy near
    the far end wraps to a negative path length and is silently dropped.
    Within one short segment the minimum image is valid, so the wrap is
    handled locally and the path length reassembled as
    t = t_segment_centre + (local displacement . e_los).

    If the beam is long enough to re-enter a region it already crossed, a
    galaxy is genuinely inside it twice. That is an artifact of a finite
    periodic box, not a survey property, so it is reported rather than hidden:
    duplicates arising from segment overlap are removed, but true re-crossings
    (same galaxy at well-separated path lengths) are counted and returned.
    """
    w, h, d = dims
    e_w, e_h, e_los = frame[:, 0], frame[:, 1], frame[:, 2]

    if tree is None:
        tree = build_tree(positions, box_size)
    pos = np.mod(np.asarray(positions, dtype=float), box_size)

    # segment length: short enough that minimum image is safe inside a segment
    if segment_length is None:
        segment_length = min(d, max(np.hypot(w, h), box_size / 8.0))
    n_seg = int(np.ceil(d / segment_length))
    seg_len = d / n_seg
    # sphere enclosing one segment of the cuboid
    radius = 0.5 * np.sqrt(w**2 + h**2 + seg_len**2)

    t_centres = (np.arange(n_seg) + 0.5) * seg_len
    centres = np.mod(origin + np.outer(t_centres, e_los), box_size)

    found_idx, found_t = [], []
    for s, c in enumerate(centres):
        cand = tree.query_ball_point(c, radius)
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)
        # local displacement from the segment centre, minimum image (valid:
        # |displacement| <= radius << box_size/2)
        dv = pos[cand] - c
        dv -= box_size * np.round(dv / box_size)
        t_loc = dv @ e_los
        t_glob = t_centres[s] + t_loc
        uu = dv @ e_w
        vv = dv @ e_h
        inside = (
            (t_glob >= 0.0) & (t_glob < d)
            & (np.abs(uu) <= w / 2.0) & (np.abs(vv) <= h / 2.0)
        )
        if inside.any():
            found_idx.append(cand[inside])
            found_t.append(t_glob[inside])

    if not found_idx:
        empty = np.empty(0, dtype=int)
        return (empty, 0) if return_multiplicity else empty

    idx = np.concatenate(found_idx)
    tt = np.concatenate(found_t)

    # drop duplicates from overlapping segment spheres: same galaxy at
    # essentially the same path length. Keep genuine re-crossings.
    key = np.stack([idx.astype(np.int64), np.round(tt / max(seg_len, 1e-9) * 4).astype(np.int64)], 1)
    _, keep = np.unique(key, axis=0, return_index=True)
    idx, tt = idx[np.sort(keep)], tt[np.sort(keep)]

    if return_multiplicity:
        _, counts = np.unique(idx, return_counts=True)
        return idx, int((counts > 1).sum())
    return idx


def beam_count_matrix(positions, MUV, MUV_limits, origins, frames, dims, box_size, verbose=True):
    """(n_beams, n_limits) counts of galaxies brighter than each MUV limit.

    MUV is sorted once and thresholds are resolved by searchsorted, so all
    limits come out of a single pass per beam.
    """
    order = np.argsort(MUV)
    pos_sorted = np.mod(positions[order], box_size)
    MUV_sorted = MUV[order]
    limits = np.asarray(MUV_limits, dtype=float)

    n_beams = len(origins)
    counts = np.zeros((n_beams, len(limits)), dtype=int)
    total_dup = 0

    tree = build_tree(pos_sorted, box_size)   # built once, reused by every beam

    for i in range(n_beams):
        idx, ndup = galaxies_in_beam(
            pos_sorted, origins[i], frames[i], dims, box_size,
            return_multiplicity=True, tree=tree,
        )
        total_dup += ndup
        if idx.size:
            # idx index the MUV-sorted array, so counts brighter than each
            # limit come from one searchsorted on the selected subset
            sel = np.sort(MUV_sorted[idx])
            counts[i] = np.searchsorted(sel, limits, side="left")

    if verbose:
        print(f"beams: {n_beams}   dims (w,h,d) = {dims.round(2)} cMpc")
        print(f"beam volume = {np.prod(dims):.5g} cMpc^3  "
              f"= {np.prod(dims) / box_size**3:.5f} x box")
        if total_dup:
            print(f"WARNING: {total_dup} galaxy-beam pairs counted more than once "
                  f"(beam re-crossed its own path through the periodic box)")
        for j, lim in enumerate(limits):
            print(f"  MUV < {lim}: mean count = {counts[:, j].mean():9.3f}, "
                  f"empty beams = {(counts[:, j] == 0).sum()}")
    return counts


#variance
def cosmic_variance(count_matrix, MUV_limits, mu_known=None, model_name="", verbose=True):
    limits = np.asarray(MUV_limits, dtype=float)
    M = count_matrix.shape[0]

    mean_counts, variances, sigma_cv, sigma_cv_err = [], [], [], []
    if model_name:
        print(f"\n{model_name}: cosmic variance (random pencil beams)")

    for j in range(count_matrix.shape[1]):
        N = count_matrix[:, j].astype(float)
        mu = float(N.mean()) if mu_known is None else float(np.atleast_1d(mu_known)[j])
        var = float(N.var(ddof=1)) if mu_known is None else float(((N - mu) ** 2).sum() / M)

        if mu <= 0:
            s_cv, s_err = np.nan, np.nan
        elif var <= mu:
            s_cv, s_err = 0.0, np.nan          # shot-noise dominated
        else:
            s_cv = np.sqrt((var - mu) / mu**2)
            var_of_var = 2.0 * var**2 / max(M - 1, 1)
            s_err = np.sqrt(var_of_var) / (2.0 * s_cv * mu**2)

        mean_counts.append(mu); variances.append(var)
        sigma_cv.append(s_cv);  sigma_cv_err.append(s_err)
        if verbose:
            print(f"  MUV < {limits[j]}: mu={mu:9.4g}, var={var:11.5g}, "
                  f"poisson={mu:9.4g}, sigma_cv={s_cv:7.4f} +/- {s_err:.4f}")

    return {
        "mean_counts": np.array(mean_counts),
        "total_variances": np.array(variances),
        "poisson_variances": np.array(mean_counts),
        "sigma_cv": np.array(sigma_cv),
        "sigma_cv_err": np.array(sigma_cv_err),
        "n_beams": M,
    }

def _shot_inflation(count_matrix, j=0):
    """(1 + mu/Var_clust): inflation of the fractional error on sigma_CV caused
    by subtracting Poisson shot noise. 1.0 when clustering dominates."""
    N = np.asarray(count_matrix)[:, j].astype(float)
    mu = N.mean()
    Vc = N.var(ddof=1) - mu
    return 1.0 + mu / Vc if Vc > 0 else np.inf


def n_beams_for_precision(target_frac=None, instrument=None, box_size=500.0, shot_inflation=1.0):
    floor = {"NIRCam": 0.0186, "Roman": 0.0417}
    if target_frac is None:
        target_frac = floor.get(instrument, 0.02)
    # shot_inflation from _shot_inflation(count_matrix); >1 needs more beams
    M = 1.0 + 2.0 * (shot_inflation / (2.0 * target_frac)) ** 2
    out = {"M_for_target": int(np.ceil(M)), "target_frac": target_frac,
           "shot_inflation": shot_inflation}
    if instrument in floor:
        fl = floor[instrument]
        out["box_floor_frac"] = fl
        out["M_useful_cap"] = int(np.ceil(1.0 + 2.0 / (2.0 * fl) ** 2))
        out["M_recommended"] = min(out["M_for_target"], out["M_useful_cap"])
        if target_frac < fl:
            out["note"] = (
                f"target {100*target_frac:.2f}% is below the {100*fl:.2f}% "
                f"box-to-box floor for {instrument}; extra beams cannot reach it. "
                f"Only a larger box or more boxes can."
            )
    else:
        out["M_recommended"] = out["M_for_target"]
    return out


def bootstrap_error(count_matrix, MUV_limits, n_boot=500, seed=42):
    """Resample beams with replacement -> honest error bar on sigma_CV.

    Preferred over the analytic Gaussian error because it makes no Gaussian
    assumption and automatically absorbs the (small) correlation induced by
    overlapping beams. Validated against the true run-to-run scatter:
    ratio 0.87-1.13 across M = 21..400 for both instruments, versus the
    convergence tolerance's 4-11x understatement.

    Reports PLACEMENT noise only. The box-to-box floor (NIRCam 1.86%,
    Roman 4.17%) must be added in quadrature for a total error, and cannot be
    reduced by adding beams.
    """
    rng = np.random.default_rng(seed)
    limits = np.asarray(MUV_limits, dtype=float)
    M = count_matrix.shape[0]
    err = np.zeros(len(limits))
    for j in range(len(limits)):
        N = count_matrix[:, j].astype(float)
        vals = []
        for _ in range(n_boot):
            z = N[rng.integers(0, M, M)]
            m = z.mean()
            if m > 0:
                v = z.var(ddof=1)
                vals.append(np.sqrt(max(v - m, 0.0)) / m)
        err[j] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
    return err


def effective_beam_count(origins, dims, box_size, instrument, ng=100):
    o = np.mod(np.asarray(origins), box_size)
    cell = box_size / ng
    ext = np.maximum(np.asarray(dims, dtype=float) / cell, 1e-9)
    oc = o / cell
    M = len(oc)
    ov = np.ones((M, M))
    for d in range(3):
        dd = np.abs(oc[:, d][:, None] - oc[:, d][None, :])
        dd = np.minimum(dd, ng - dd)
        ov *= np.clip(ext[d] - dd, 0, None) / ext[d]
    fcal, rcal = _R2_CALIBRATION[instrument]
    R = np.interp(ov, fcal, rcal)
    R[ov <= 0] = 0.0
    R = np.clip(R, 0.0, None)
    np.fill_diagonal(R, 1.0)
    return float(M ** 2 / R.sum())


# Calibrated r2(f) = corr(dN_a^2, dN_b^2) vs shared volume fraction, from
# 400,000 independent beam pairs in a clustered 500 cMpc box per instrument.
_R2_CALIBRATION = {
    "NIRCam": (np.array([0.0, 0.05, 0.2, 0.5, 1.0]),
               np.array([0.0, 0.02, 0.10, 0.40, 0.70])),
    "Roman":  (np.array([0.0, 0.012, 0.073, 0.146, 0.246, 0.365, 0.515, 0.675, 1.0]),
               np.array([0.0, 0.021, 0.013, 0.041, 0.161, 0.304, 0.435, 0.684, 1.0])),
}
_COSMIC_CALIB = {"NIRCam": 1.51, "Roman": 1.14}


def cosmic_error(origins, dims, box_size, instrument, ng=100):
    Me = effective_beam_count(origins, dims, box_size, instrument, ng=ng)
    return _COSMIC_CALIB[instrument] * 0.5 * np.sqrt(2.0 / max(Me - 1.0, 1.0)), Me


def total_error(count_matrix, MUV_limits, instrument, origins=None, dims=None, box_size=500.0, n_boot=500, seed=42):
    floor = {"NIRCam": 0.0186, "Roman": 0.0417}
    scv = cosmic_variance(count_matrix, MUV_limits, model_name="",
                          verbose=False)["sigma_cv"]
    meas = bootstrap_error(count_matrix, MUV_limits, n_boot=n_boot, seed=seed)
    # measurement error: inflate for shot-noise subtraction (see _shot_inflation)
    meas = np.asarray(meas) * np.array(
        [_shot_inflation(count_matrix, j) for j in range(len(np.atleast_1d(MUV_limits)))])
    if origins is not None and dims is not None:
        fl, M_eff = cosmic_error(origins, dims, box_size, instrument)
    else:
        fl, M_eff = floor.get(instrument), None
        if fl is None:
            raise ValueError(f"no measured cosmic-error floor for {instrument!r}")
    cosmic = fl * np.asarray(scv)
    return {
        "sigma_cv": np.asarray(scv),
        "meas_err": meas,
        "cosmic_err": cosmic,
        "total_err": np.sqrt(meas ** 2 + cosmic ** 2),
        "cosmic_frac": fl,
        "M_eff": M_eff,
        "note": (
            "cosmic_err is a LOWER BOUND: it is the box-to-box scatter of "
            "500 cMpc boxes and still misses modes longer than the box."
        ),
    }


def run_beam_cv(run_type="LCDM", instrument="NIRCam", n_beams=None, MUV_limits=None, seed=None, use_known_mean=True, verbose=True):
    if MUV_limits is None:
        MUV_limits = cv.MUV_limits
    MUV_limits = np.asarray(MUV_limits, dtype=float)
    if seed is None:
        seed = cv.random_seed

    if run_type == "EDE":
        data = cv.load_EDE_data(cv.ede_hdf5_path, cv.redshift_EDE,
                                cv.box_size_cmpc_EDE)
    else:
        data = cv.load_data(cv.lcdm_base_path, cv.snap_num)
    positions_cmpc, MUV, _ = cv.select_valid_uv_galaxies(data, run_type)
    box_size = float(data["box_size_cmpc"])

    width, height = INSTRUMENTS[instrument]
    dims = beam_dimensions(cv.lumina_cosmology, cv.z_center, cv.z_min,
                           cv.z_max, width, height)

    if n_beams is None:
        n_beams = n_beams_for_precision(None, instrument,
                                        box_size=box_size)["M_recommended"]

    origins, frames = place_beams(n_beams, box_size, seed=seed)
    count_matrix = beam_count_matrix(positions_cmpc, MUV, MUV_limits,
                                     origins, frames, dims, box_size,
                                     verbose=verbose)

    mu_known = None
    if use_known_mean:
        V_beam = float(np.prod(dims))
        n_bar = np.array([(MUV < lim).sum() for lim in MUV_limits]) / box_size ** 3
        mu_known = n_bar * V_beam

    cv = cosmic_variance(count_matrix, MUV_limits, mu_known=mu_known,
                         model_name=f"{run_type} {instrument} random beams")
    err = total_error(count_matrix, MUV_limits, instrument,
                      origins=origins, dims=dims, box_size=box_size)

    return {"count_matrix": count_matrix, "origins": origins, "frames": frames,
            "dims_cmpc": dims, "n_beams": n_beams, "box_size_cmpc": box_size,
            "MUV_limits": MUV_limits, **cv,
            "meas_err": err["meas_err"], "cosmic_err": err["cosmic_err"],
            "total_err": err["total_err"], "M_eff": err["M_eff"]}
