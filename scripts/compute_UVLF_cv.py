import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as con
import h5py
from scipy.integrate import quad

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))
from scripts import groupcat
from scripts import observation_data as obs

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

#Lumina planck cosmology parameters (methods paper Table 2)
lumina_cosmology=FlatLambdaCDM(
    H0=67.66*u.km/u.s/u.Mpc,
    Om0=0.3096,
    Ob0=4.897/100,
    Tcmb0=2.7255*u.K,
    Neff=3.046,
    m_nu=0.0*u.eV,
)

lcdm_base_path = "/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/group_files"
#EDE run path below
ede_hdf5_path = "/home/jxie2/Projects/lumina_cosmic_variance/data/EDE_group_files/fof_subhalo_tab_048.hdf5"
snap_num = 48
redshift_EDE = 10.0
box_size_cmpc_EDE = 500.0

#experimental design
survey_area=9.7*u.arcmin**2
z_min=9.2
z_max=10.9
z_center = 0.5 * (z_min + z_max)
nircam_width=4.4*u.arcmin
nircam_height=2.2*u.arcmin

res_cut=0
geometry="rectangle"

MUV_bin_width=0.5
MUV_bin_edges = np.arange(-24, -16.0, MUV_bin_width)
MUV_limits = np.array([-19.5, -20.0, -20.5])
best_delta_MUV=1
best_sigma_UV=1
random_seed=42

obs_MUV_limits = np.array([-19.5, -20.0, -20.5])
obs_mu = np.array([3.5588749404099795, 1.3235294117647056, 0.6146341463414631])
obs_sigma_cv = np.array([0.96, 1.46, 1.71]) #from digitizer
obs_lower_error = [0.18,0.44,0.59]
obs_upper_error=[0.20, 0.54, 0.72]

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

def luminosity1500_to_MUV(luminosity1500_raw):
    # Convert simulation luminosity unit to erg/s/Angstrom units
    unit_conversion = (1e10* con.M_sun.value* 1e7* (1e3)**3/ (1e3 * con.pc.value))
    L_lambda = luminosity1500_raw * unit_conversion
    # Convert luminosity density to flux density at 10 pc.
    distance_cm = 10 * con.pc.cgs.value
    area_10pc = 4 * np.pi * distance_cm**2
    f_lambda = L_lambda / area_10pc  # erg/s/cm^2/Angstrom
    # Convert f_lambda to f_nu for AB magnitude.
    f_lambda_per_cm = f_lambda * 1e8  # per Angstrom -> per cm
    lambda_cm = 1500e-8
    c_cm_s = con.c.cgs.value
    f_nu = f_lambda_per_cm * lambda_cm**2 / c_cm_s
    MUV = -2.5 * np.log10(f_nu) - 48.6

    return MUV

def fuv_to_MUV(fuv):
    # EDE fuv is already L_lambda in erg/s/Angstrom.
    distance_cm = 10 * con.pc.cgs.value
    area_10pc = 4 * np.pi * distance_cm**2
    f_lambda = fuv / area_10pc
    f_lambda_per_cm = f_lambda * 1e8
    lambda_cm = 1500e-8
    c_cm_s = con.c.cgs.value
    f_nu = f_lambda_per_cm * lambda_cm**2 / c_cm_s

    MUV = -2.5 * np.log10(f_nu) - 48.6
    return MUV

def load_EDE_data(hdf5_path, redshift, box_size_cmpc):
    print("loading EDE catalog...", flush=True)

    with h5py.File(hdf5_path, "r") as f:
        positions_ckpc = f["pos"][:]
        fuv = f["fuv"][:] # in erg/s/Angstrom units

    print("finished loading EDE catalog", flush=True)
    print("redshift:", redshift)
    print("box size:", box_size_cmpc, "cMpc")

    return {
        "positions_raw": positions_ckpc,
        "luminosity1500_raw": fuv,
        "redshift": redshift,
        "box_size_cmpc": box_size_cmpc,
        "catalog_path": hdf5_path,
    }

def load_data(base_path, snap_num):
    catalog_path = groupcat.gcPath(base_path, snap_num)
    print("loading subhalo catalog...", flush=True)
    subhalos = groupcat.loadSubhalos(base_path,snap_num,fields=["SubhaloPos", "Subhalo_Luminosity1500"],)
    print("finished loading subhalo catalog", flush=True)
    with h5py.File(catalog_path, "r") as f:
        h = f["Parameters"].attrs["HubbleParam"]
        box_size_raw = f["Header"].attrs["BoxSize"]
        redshift = f["Header"].attrs["Redshift"]
    box_size_cmpc = box_size_raw / (1000 * h)
    print("h:", h)
    print("redshift:", redshift)
    print("box size:", box_size_cmpc, "cMpc")

    return {
        "positions_raw": subhalos["SubhaloPos"],
        "luminosity1500_raw": subhalos["Subhalo_Luminosity1500"],
        "h": h,
        "redshift": redshift,
        "box_size_cmpc": box_size_cmpc,
        "catalog_path": catalog_path,
    }

def select_valid_uv_galaxies(data, run_type):
    positions = data["positions_raw"]
    luminosity1500 = data["luminosity1500_raw"]
    MUV_all = np.full(len(luminosity1500), np.nan)
    positive_luminosity = ((luminosity1500 > 0) & np.isfinite(luminosity1500))

    if run_type == "EDE":
        MUV_all[positive_luminosity] = fuv_to_MUV(luminosity1500[positive_luminosity])
    else:
        MUV_all[positive_luminosity] = luminosity1500_to_MUV(luminosity1500[positive_luminosity])

    valid = (positive_luminosity& np.isfinite(MUV_all)& np.all(np.isfinite(positions), axis=1))

    if run_type == "EDE":
        positions_cmpc = positions[valid] / 1000
    else:
        h = data["h"]
        positions_cmpc = positions[valid] / (1000 * h)

    MUV = MUV_all[valid]

    print("All subhalos:", len(luminosity1500))
    print("Selected UV galaxies:", valid.sum())
    print("Selected fraction:", valid.mean())
    print("M_UV range:", np.min(MUV), np.max(MUV))
    print("M_UV percentiles:", np.percentile(MUV, [1, 50, 99]))

    return positions_cmpc, MUV, valid

def assign_galaxies_to_MUV_bins(MUV, inside, MUV_bin_width):
    MUV_inside = MUV[inside]

    MUV_min = np.floor(MUV_inside.min() / MUV_bin_width) * MUV_bin_width
    MUV_max = np.ceil(MUV_inside.max() / MUV_bin_width) * MUV_bin_width

    MUV_bin_edges = np.arange(MUV_min,MUV_max + MUV_bin_width,MUV_bin_width,)

    n_MUV_bins = len(MUV_bin_edges) - 1
    MUV_bin_ids = np.full(len(MUV_inside), -1, dtype=int)

    for bin_id in range(n_MUV_bins):
        lower_edge = MUV_bin_edges[bin_id]
        upper_edge = MUV_bin_edges[bin_id + 1]

        in_this_bin = ((MUV_inside >= lower_edge)& (MUV_inside < upper_edge))
        MUV_bin_ids[in_this_bin] = bin_id

    if np.any(MUV_inside == MUV_bin_edges[-1]):
        MUV_bin_ids[MUV_inside == MUV_bin_edges[-1]] = n_MUV_bins - 1

    in_MUV_range = MUV_bin_ids >= 0

    print("M_UV range:", MUV_min, MUV_max)
    print("M_UV bin edges:", MUV_bin_edges)
    print("number of M_UV bins:", n_MUV_bins)
    print("galaxies with assigned M_UV bin:", in_MUV_range.sum())

    return MUV_bin_ids, in_MUV_range, MUV_bin_edges, n_MUV_bins

def match_mu(MUV_inside, field_ids, target_number, seed=42):
    rng = np.random.default_rng(seed)

    if len(MUV_inside) < target_number:
        raise ValueError("cannot downsample to a larger catalog")

    keep_indices = np.sort(rng.choice(len(MUV_inside),size=target_number,replace=False,))
    MUV_removed = MUV_inside[keep_indices]
    field_ids_removed = field_ids[keep_indices]

    return MUV_removed, field_ids_removed

def calculate_field_volume(cosmology, z_min, z_max, survey_area):
    def differential_volume(z):
        return cosmology.differential_comoving_volume(z).value

    volume, error = quad(differential_volume, z_min, z_max)
    field_solid_angle = survey_area.to(u.sr)
    field_volume = volume * u.Mpc**3 / u.sr * field_solid_angle
    field_volume_error = error * u.Mpc**3 / u.sr * field_solid_angle
    los_depth = cosmology.comoving_distance(z_max) - cosmology.comoving_distance(z_min)

    return field_volume, field_volume_error, los_depth

def calculate_rectangle_field_lengths(cosmology, z_center, z_min, z_max, width_angle, height_angle):
    transverse_distance = cosmology.comoving_transverse_distance(z_center).to_value(u.Mpc)
    los_depth = (cosmology.comoving_distance(z_max) - cosmology.comoving_distance(z_min)).to_value(u.Mpc)

    width_cmpc = transverse_distance * width_angle.to_value(u.rad)
    height_cmpc = transverse_distance * height_angle.to_value(u.rad)

    field_lengths_cmpc = np.array([width_cmpc, height_cmpc, los_depth])

    print("rectangle field lengths [cMpc]:", field_lengths_cmpc)
    print("rectangle field volume [cMpc^3]:", np.prod(field_lengths_cmpc))

    return field_lengths_cmpc


def assign_subhalos_to_fields(positions_cmpc,box_size_cmpc,field_volume,geometry="cube",field_lengths_cmpc=None):
    if geometry == "cube":
        cube_side_cmpc = np.cbrt(field_volume.to_value(u.Mpc**3))
        n_side = int(np.floor(box_size_cmpc / cube_side_cmpc))
        covered_length = n_side * cube_side_cmpc
        n_fields = n_side**3
        inside = np.all(positions_cmpc < covered_length, axis=1)
        field_coord = np.floor(positions_cmpc[inside] / cube_side_cmpc).astype(int)
        field_ids = (field_coord[:, 0] * n_side**2+ field_coord[:, 1] * n_side+ field_coord[:, 2])

        print("cube side:", cube_side_cmpc, "cMpc")
        print("cubes per dimension:", n_side)
        print("total fields:", n_fields)
        print("fields with at least one UV galaxy:", len(np.unique(field_ids)))
        print("empty fields:", n_fields - len(np.unique(field_ids)))

        return field_ids, inside, cube_side_cmpc, n_fields

    if geometry=="rectangle":
        n_side = np.floor(box_size_cmpc / field_lengths_cmpc).astype(int)
        covered_lengths = n_side * field_lengths_cmpc
        n_fields = int(np.prod(n_side))

        inside = np.all(positions_cmpc < covered_lengths, axis=1)
        field_coord = np.floor(positions_cmpc[inside] / field_lengths_cmpc).astype(int)
        field_ids = (field_coord[:, 0] * n_side[1] * n_side[2]+ field_coord[:, 1] * n_side[2]+ field_coord[:, 2])

        print("rectangle side lengths:", field_lengths_cmpc, "cMpc")
        print("rectangles per dimension:", n_side)
        print("total fields:", n_fields)

        return field_ids, inside, field_lengths_cmpc, n_fields
    
    raise NotImplementedError("geometry must be either cube or rectangle")
    

def build_MUV_count_matrix(field_ids, MUV_inside, n_fields, MUV_limits, model_name=""):
    count_matrix = np.zeros((n_fields, len(MUV_limits)), dtype=int)
    if model_name != "":
        print(f"\n{model_name}: count matrix")
    for bin_id, MUV_limit in zip(range(len(MUV_limits)), MUV_limits):
        bright = MUV_inside < MUV_limit

        for field_id in field_ids[bright]:
            count_matrix[field_id, bin_id] += 1

        print(
            f"MUV < {MUV_limit}: "
            f"total galaxies={count_matrix[:, bin_id].sum()}, "
            f"fields with galaxies={np.count_nonzero(count_matrix[:, bin_id])}"
        )

    return count_matrix

def compute_cosmic_variance(count_matrix, MUV_limits, model_name=""):
    mean_counts = []
    total_variances = []
    poisson_variances = []
    sigma_cv_values = []

    if model_name != "":
        print(f"\n{model_name}: cosmic variance")

    for bin_id in range(count_matrix.shape[1]):
        counts = count_matrix[:, bin_id]

        mean_count = np.mean(counts)
        total_variance = np.var(counts)
        poisson_variance = mean_count

        if mean_count == 0:
            sigma_cv = np.nan
        elif total_variance <= poisson_variance:
            sigma_cv = 0.0
        else:
            sigma_cv = np.sqrt((total_variance - poisson_variance) / mean_count**2)

        mean_counts.append(mean_count)
        total_variances.append(total_variance)
        poisson_variances.append(poisson_variance)
        sigma_cv_values.append(sigma_cv)

        print(
            f"M_UV < {MUV_limits[bin_id]}: "
            f"mu={mean_count:.4g}, "
            f"variance={total_variance:.4g}, "
            f"poisson={poisson_variance:.4g}, "
            f"sigma_cv={sigma_cv:.4g}, "
            f"total galaxies={counts.sum()}"
        )

    return {
        "mean_counts": np.array(mean_counts),
        "total_variances": np.array(total_variances),
        "poisson_variances": np.array(poisson_variances),
        "sigma_cv": np.array(sigma_cv_values),
    }

def compute_UVLF(MUV_values, field_volume, n_fields, MUV_bin_edges, MUV_bin_width):
    counts, _ = np.histogram(MUV_values, bins=MUV_bin_edges)
    volume = n_fields * field_volume.to_value(u.Mpc**3)
    phi = counts / (volume * MUV_bin_width)
    MUV_bin_centers = 0.5 * (MUV_bin_edges[:-1] + MUV_bin_edges[1:])
    nonzero = phi > 0

    return MUV_bin_centers[nonzero], np.log10(phi[nonzero])

def plot_uvlf_overlay(lcdm_MUV_inside,lcdm_MUV_A1,lcdm_MUV_C1,ede_MUV_inside,field_volume,n_fields):
    # lcdm_raw_bin, lcdm_raw_log_phi = compute_UVLF(lcdm_MUV_inside,field_volume,n_fields,MUV_bin_edges,MUV_bin_width,)
    lcdm_A1_bin, lcdm_A1_log_phi = compute_UVLF(lcdm_MUV_A1,field_volume,n_fields,MUV_bin_edges,MUV_bin_width,)
    lcdm_C1_bin, lcdm_C1_log_phi = compute_UVLF(lcdm_MUV_C1,field_volume,n_fields,MUV_bin_edges,MUV_bin_width,)
    ede_bin, ede_log_phi = compute_UVLF(ede_MUV_inside,field_volume,n_fields,MUV_bin_edges,MUV_bin_width,)

    fig, ax = plt.subplots(figsize=(9, 7))

    # ax.plot(lcdm_raw_bin,lcdm_raw_log_phi,marker="s",linestyle="-",color="gray",label="LCDM",)
    ax.plot(lcdm_A1_bin,lcdm_A1_log_phi,marker="D",linestyle="-",color="tab:blue",label=rf"LCDM A1: $\Delta M_{{UV}}={best_delta_MUV}$",)
    ax.plot(lcdm_C1_bin,lcdm_C1_log_phi,marker="o",linestyle="-",color="tab:orange",label=rf"LCDM C1: $\sigma_{{UV}}={best_sigma_UV}$",)
    ax.plot(ede_bin,ede_log_phi,marker="^",linestyle="-",color="tab:green",label="EDE",)

    obs.plot_photoz_constraints(redshift=10, ax=ax, marker="o", ms=5, linestyle="none", color="black", capsize=4)
    obs.plot_specz_constraints(redshift=10, ax=ax, marker="s", ms=5, linestyle="none", color="red", capsize=4)

    ax.set_xlabel(r"$M_{\rm UV}$")
    ax.set_ylabel(r"$\log_{10}(\Phi / {\rm Mpc}^{-3}\,{\rm mag}^{-1})$")
    ax.text(0.9, 0.9, r"$z \simeq 10$", transform=ax.transAxes, ha="right", va="top", fontsize=20)

    # ax.set_title(r"UVLF at $z \simeq 10$", fontsize=24)
    ax.set_xlim(-24, -16)
    ax.set_ylim(-7.5, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=22, markerscale=0.6, handlelength=1.5, labelspacing=0.4, borderpad=0.4, frameon=True)

    plt.savefig(output_dir / "LCDM_EDE_UVLF_vs_obs_z=10.png", dpi=200, bbox_inches="tight")

def A1_delta_MUV(MUV_inside, field_ids, n_fields, MUV_limits, delta_MUV_list):
    results = []
    
    for delta_MUV in delta_MUV_list:
        model_name = f"LCDM A1, delta_MUV={delta_MUV}"
        MUV_model_inside = MUV_inside - delta_MUV
        count_matrix = build_MUV_count_matrix(field_ids=field_ids,MUV_inside=MUV_model_inside,n_fields=n_fields,MUV_limits=MUV_limits,model_name=model_name,)
        cv_results = compute_cosmic_variance(count_matrix=count_matrix,MUV_limits=MUV_limits,model_name=model_name)

        for bin_id in range(len(MUV_limits)):
            results.append(
                {
                    "model": "A1",
                    "parameter": delta_MUV,
                    "MUV_limit": MUV_limits[bin_id],
                    "mu": cv_results["mean_counts"][bin_id],
                    "sigma_cv": cv_results["sigma_cv"][bin_id],
                    "total_galaxies": count_matrix[:, bin_id].sum(),
                }
            )

    return results

def C1_sigma_UV(MUV_inside, field_ids, n_fields, MUV_limits, sigma_UV_list, seed=42):
    results = []
    
    for sigma_UV in sigma_UV_list:
        model_name = f"LCDM C1, sigma_UV={sigma_UV}"
        rng = np.random.default_rng(seed)
        MUV_model_inside = MUV_inside + rng.normal(0.0,sigma_UV,size=len(MUV_inside),)
        count_matrix = build_MUV_count_matrix(field_ids=field_ids,MUV_inside=MUV_model_inside,n_fields=n_fields,MUV_limits=MUV_limits,model_name=model_name,)
        cv_results = compute_cosmic_variance(count_matrix=count_matrix,MUV_limits=MUV_limits,model_name=model_name,)
        for bin_id in range(len(MUV_limits)):
            results.append(
                {
                    "model": "C1",
                    "parameter": sigma_UV,
                    "MUV_limit": MUV_limits[bin_id],
                    "mu": cv_results["mean_counts"][bin_id],
                    "sigma_cv": cv_results["sigma_cv"][bin_id],
                    "total_galaxies": count_matrix[:, bin_id].sum(),
                }
            )

    return results

def og_UV_model(MUV_inside, field_ids, n_fields, MUV_limits, model_name):
    count_matrix = build_MUV_count_matrix(field_ids=field_ids,MUV_inside=MUV_inside,n_fields=n_fields,MUV_limits=MUV_limits,model_name=model_name)
    cv_results = compute_cosmic_variance(count_matrix=count_matrix,MUV_limits=MUV_limits,model_name=model_name)
    results = []

    for bin_id in range(len(MUV_limits)):
        results.append(
            {
                "model": model_name,
                "parameter": 0.0,
                "MUV_limit": MUV_limits[bin_id],
                "mu": cv_results["mean_counts"][bin_id],
                "sigma_cv": cv_results["sigma_cv"][bin_id],
                "total_galaxies": count_matrix[:, bin_id].sum(),
            }
        )

    return results

def plot_cv_vs_MUV_limit(lcdm_results,lcdm_A1_results,lcdm_C1_results,ede_results):
    lcdm_sigma_cv = []
    lcdm_A1_sigma_cv = []
    lcdm_C1_sigma_cv = []
    ede_sigma_cv = []

    for MUV_limit in MUV_limits:
        for result in lcdm_results:
            if result["MUV_limit"] == MUV_limit:
                lcdm_sigma_cv.append(result["sigma_cv"])

        for result in lcdm_A1_results:
            if np.isclose(result["parameter"], best_delta_MUV) and result["MUV_limit"] == MUV_limit:
                lcdm_A1_sigma_cv.append(result["sigma_cv"])

        for result in lcdm_C1_results:
            if np.isclose(result["parameter"], best_sigma_UV) and result["MUV_limit"] == MUV_limit:
                lcdm_C1_sigma_cv.append(result["sigma_cv"])

        for result in ede_results:
            if result["MUV_limit"] == MUV_limit:
                ede_sigma_cv.append(result["sigma_cv"])

    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(MUV_limits,lcdm_sigma_cv,marker="s",color="gray",label="LCDM",)
    ax.plot(MUV_limits,lcdm_A1_sigma_cv,marker="D",color="tab:blue",label=rf"LCDM A1: $\Delta M_{{UV}}={best_delta_MUV}$",)
    ax.plot(MUV_limits,lcdm_C1_sigma_cv,marker="o",color="tab:orange",label=rf"LCDM C1: $\sigma_{{UV}}={best_sigma_UV}$",)
    ax.plot(MUV_limits,ede_sigma_cv,marker="^",color="tab:green",label="EDE",)
    ax.scatter(MUV_limits,obs_sigma_cv,marker="*",s=220,color="red",label="PANORAMIC obs",zorder=10,)
    ax.errorbar(MUV_limits,obs_sigma_cv,yerr=[obs_lower_error, obs_upper_error], ecolor='#fc2b26',fmt='none',capsize=6,elinewidth=0.8)

    ax.set_xlabel(r"$M_{\rm UV}$ limit")
    ax.set_ylabel(r"$\sigma_{\rm CV}$")
    ax.text(0.9, 0.9, r"$z \simeq 10$", transform=ax.transAxes, ha="right", va="top", fontsize=20)
    # ax.set_title(r"Cosmic variance vs UV magnitude limit at $z \simeq 10$", fontsize=20)
    ax.set_xlim(-19.3, -20.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, markerscale=0.6, handlelength=1.5, labelspacing=0.4, borderpad=0.4, frameon=True)

    plt.savefig(output_dir / "LCDM_EDE_sigma_cv_vs_MUV_limit_z=10.png",dpi=200,bbox_inches="tight",)

def plot_mu_sigma_cv(lcdm_results,lcdm_A1_results,lcdm_C1_results,ede_results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for plot_id, (ax, MUV_limit) in enumerate(zip(axes, MUV_limits)):
        lcdm_mu = []
        lcdm_sigma_cv = []

        lcdm_A1_mu = []
        lcdm_A1_sigma_cv = []

        lcdm_C1_mu = []
        lcdm_C1_sigma_cv = []

        ede_mu = []
        ede_sigma_cv = []

        for result in lcdm_results:
            if result["MUV_limit"] == MUV_limit:
                lcdm_mu.append(result["mu"])
                lcdm_sigma_cv.append(result["sigma_cv"])

        for result in lcdm_A1_results:
            if np.isclose(result["parameter"], best_delta_MUV) and result["MUV_limit"] == MUV_limit:
                lcdm_A1_mu.append(result["mu"])
                lcdm_A1_sigma_cv.append(result["sigma_cv"])

        for result in lcdm_C1_results:
            if np.isclose(result["parameter"], best_sigma_UV) and result["MUV_limit"] == MUV_limit:
                lcdm_C1_mu.append(result["mu"])
                lcdm_C1_sigma_cv.append(result["sigma_cv"])

        for result in ede_results:
            if result["MUV_limit"] == MUV_limit:
                ede_mu.append(result["mu"])
                ede_sigma_cv.append(result["sigma_cv"])

        ax.scatter(lcdm_mu,lcdm_sigma_cv,marker="s",color="gray",label="LCDM",)
        ax.scatter(lcdm_A1_mu,lcdm_A1_sigma_cv,marker="D",color="tab:blue",label=rf"LCDM A1: $\Delta M_{{UV}}={best_delta_MUV}$",)
        ax.scatter(lcdm_C1_mu,lcdm_C1_sigma_cv,marker="o",color="tab:orange",label=rf"LCDM C1: $\sigma_{{UV}}={best_sigma_UV}$",)
        ax.scatter(ede_mu,ede_sigma_cv,marker="^",color="tab:green",label="EDE",)
        ax.scatter(obs_mu[plot_id],obs_sigma_cv[plot_id],marker="*",s=220,color="red",label="PANORAMIC obs",zorder=10,)

        ax.set_title(rf"$M_{{UV}} < {MUV_limit}$", fontsize=16)
        ax.set_xlabel(r"mean galaxies per field $\mu$")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

    axes[0].set_ylabel(r"$\sigma_{\rm CV}$")
    axes[0].legend(fontsize=11, markerscale=0.8, handlelength=1.2, labelspacing=0.35, borderpad=0.35, frameon=True)

    fig.suptitle(r"Cosmic variance vs mean counts at $z \simeq 10$", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_dir / "LCDM_EDE_mu_sigma_cv_z=10.png",dpi=200,bbox_inches="tight",)

def compute_mu_from_UVLF(MUV_bin_centers, log_phi, MUV_bin_width, field_volume, MUV_limits):
    phi = 10**log_phi
    field_volume_value = field_volume.to_value(u.Mpc**3)

    mu_values = []

    for MUV_limit in MUV_limits:
        bright = MUV_bin_centers < MUV_limit

        number_density = np.sum(phi[bright] * MUV_bin_width)
        mu = number_density * field_volume_value
        mu_values.append(mu)

        print(
            f"from UVLF, MUV < {MUV_limit}: "
            f"number density={number_density:.4g} Mpc^-3, "
            f"mu={mu:.4g}"
        )

    return np.array(mu_values)

def main():
    lcdm_data = load_data(lcdm_base_path, snap_num)
    lcdm_positions_cmpc,lcdm_MUV, valid=select_valid_uv_galaxies(lcdm_data,run_type="LCDM")
    
    ede_data = load_EDE_data(hdf5_path=ede_hdf5_path,redshift=redshift_EDE,box_size_cmpc=box_size_cmpc_EDE,)
    ede_positions_cmpc, ede_MUV, valid = select_valid_uv_galaxies(ede_data, run_type="EDE")

    field_volume, field_volume_error, los_depth = calculate_field_volume(cosmology=lumina_cosmology,z_min=z_min, z_max=z_max,survey_area=survey_area,)
    field_lengths_cmpc = calculate_rectangle_field_lengths(cosmology=lumina_cosmology,z_center=z_center,z_min=z_min,z_max=z_max,width_angle=nircam_width,height_angle=nircam_height,)
    field_volume = np.prod(field_lengths_cmpc) * u.Mpc**3

    print("field volume:", field_volume)
    print("field volume error:", field_volume_error)
    print("los depth:", los_depth)

    # lcdm_field_ids, lcdm_inside, cube_side_cmpc, lcdm_n_fields = assign_subhalos_to_fields(positions_cmpc=lcdm_positions_cmpc,box_size_cmpc=lcdm_data["box_size_cmpc"],field_volume=field_volume,geometry=geometry,)
    lcdm_field_ids, lcdm_inside, field_lengths_cmpc, lcdm_n_fields = assign_subhalos_to_fields(positions_cmpc=lcdm_positions_cmpc,box_size_cmpc=lcdm_data["box_size_cmpc"],field_volume=field_volume,geometry=geometry,field_lengths_cmpc=field_lengths_cmpc,)
    lcdm_MUV_inside = lcdm_MUV[lcdm_inside]

    # ede_field_ids, ede_inside, cube_side_cmpc, ede_n_fields = assign_subhalos_to_fields(positions_cmpc=ede_positions_cmpc,box_size_cmpc=ede_data["box_size_cmpc"],field_volume=field_volume,geometry=geometry,)
    ede_field_ids, ede_inside, field_lengths_cmpc, ede_n_fields = assign_subhalos_to_fields(positions_cmpc=ede_positions_cmpc,box_size_cmpc=ede_data["box_size_cmpc"],field_volume=field_volume,geometry=geometry,field_lengths_cmpc=field_lengths_cmpc,)
    ede_MUV_inside = ede_MUV[ede_inside]
    # ede_MUV_matched, ede_field_ids_matched = match_mu(MUV_inside=ede_MUV_inside,field_ids=ede_field_ids,target_number=len(lcdm_MUV_inside),seed=random_seed,)

    #lcdm modified (A1 and C1)
    lcdm_MUV_A1 = lcdm_MUV_inside - best_delta_MUV
    rng = np.random.default_rng(random_seed)

    lcdm_MUV_C1 = lcdm_MUV_inside + rng.normal(0.0,best_sigma_UV,size=len(lcdm_MUV_inside),)
    lcdm_A1_results = A1_delta_MUV(MUV_inside=lcdm_MUV_inside,field_ids=lcdm_field_ids,n_fields=lcdm_n_fields,MUV_limits=MUV_limits,delta_MUV_list=[best_delta_MUV],)
    lcdm_C1_results = C1_sigma_UV(MUV_inside=lcdm_MUV_inside,field_ids=lcdm_field_ids,n_fields=lcdm_n_fields,MUV_limits=MUV_limits,sigma_UV_list=[best_sigma_UV],seed=random_seed,)

    lcdm_results = og_UV_model(MUV_inside=lcdm_MUV_inside,field_ids=lcdm_field_ids,n_fields=lcdm_n_fields,MUV_limits=MUV_limits,model_name="LCDM",)
    ede_results = og_UV_model(MUV_inside=ede_MUV_inside,field_ids=ede_field_ids,n_fields=ede_n_fields,MUV_limits=MUV_limits,model_name="EDE",)

    plot_uvlf_overlay(lcdm_MUV_inside=lcdm_MUV_inside,lcdm_MUV_A1=lcdm_MUV_A1,lcdm_MUV_C1=lcdm_MUV_C1,ede_MUV_inside=ede_MUV_inside,field_volume=field_volume,n_fields=lcdm_n_fields,)
    plot_cv_vs_MUV_limit(lcdm_results=lcdm_results,lcdm_A1_results=lcdm_A1_results,lcdm_C1_results=lcdm_C1_results,ede_results=ede_results,)
    plot_mu_sigma_cv(lcdm_results=lcdm_results,lcdm_A1_results=lcdm_A1_results,lcdm_C1_results=lcdm_C1_results,ede_results=ede_results,)
    

#parameter tracks (optional)
def optimize_parameter(results, MUV_limits, obs_mu, obs_sigma_cv):
    best_parameter = 0.0
    best_score = np.inf

    parameters = sorted(set(result["parameter"] for result in results))

    for parameter in parameters:
        score = 0.0

        for limit_id, MUV_limit in enumerate(MUV_limits):
            for result in results:
                if result["parameter"] == parameter and result["MUV_limit"] == MUV_limit:
                    mu_error = np.log10(result["mu"]) - np.log10(obs_mu[limit_id])
                    sigma_error = np.log10(result["sigma_cv"]) - np.log10(obs_sigma_cv[limit_id])

                    score += mu_error**2 + sigma_error**2

        if score < best_score:
            best_score = score
            best_parameter = parameter

    return best_parameter, best_score

# fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
# delta_MUV = np.linspace(0.0, 2.5, 51)
# sigma_UV = np.linspace(0.0, 1.8, 37)
# for plot_id, (ax, MUV_limit) in enumerate(zip(axes, MUV_limits)):
#     A1_mu = []
#     A1_sigma_cv = []
#     A1_delta_MUV = []

#     C1_mu = []
#     C1_sigma_cv = []
#     C1_sigma_UV = []

#     for result in A1_results:
#         if result["MUV_limit"] == MUV_limit:
#             A1_mu.append(result["mu"])
#             A1_sigma_cv.append(result["sigma_cv"])
#             A1_delta_MUV.append(result["parameter"])

#     for result in C1_results:
#         if result["MUV_limit"] == MUV_limit:
#             C1_mu.append(result["mu"])
#             C1_sigma_cv.append(result["sigma_cv"])
#             C1_sigma_UV.append(result["parameter"])

#     ax.scatter(A1_mu, A1_sigma_cv, marker="D", color="tab:blue", label="A1: delta_MUV")
#     ax.scatter(C1_mu, C1_sigma_cv, marker="o", color="tab:orange", label="C1: sigma_UV")
#     ax.scatter(obs_mu[plot_id],obs_sigma_cv[plot_id],marker="*",s=200,color="red",label="PANORAMIC obs",zorder=10,)

#     for i in range(len(A1_mu)):
#         ax.text(A1_mu[i], A1_sigma_cv[i], f"{A1_delta_MUV[i]:.1f}", color="tab:blue", fontsize=8)

#     for i in range(len(C1_mu)):
#         ax.text(C1_mu[i], C1_sigma_cv[i], f"{C1_sigma_UV[i]:.1f}", color="tab:orange", fontsize=8)

#     ax.set_title(f"MUV < {MUV_limit}")
#     ax.set_xlabel(r"mean number of galaxies per field $\mu$")
#     ax.grid(alpha=0.3)
#     ax.set_xlim(left=0)

# axes[0].set_ylabel(r"cosmic variance $\sigma_{\rm CV}$")
# axes[0].legend()

# for ax in axes:
#     ax.set_ylim(0, 2.5)

# fig.suptitle(r"UV cosmic variance tracks at $z \simeq 10$")
# plt.savefig(output_dir / "A1_C1_uv_cv_tracks_z=10.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()