# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import h5py
from scipy.integrate import quad

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))

from scripts import groupcat

# %%
#Lumina planck cosmology parameters (methods paper Table 2)
lumina_cosmology=FlatLambdaCDM(
    H0=67.66*u.km/u.s/u.Mpc,
    Om0=0.3096,
    Ob0=4.897/100,
    Tcmb0=2.7255*u.K,
    Neff=3.046,
    m_nu=0.0*u.eV,
)

#experimental design
survey_area=9.7*u.arcmin**2
delta_z=1.0
mass_bin_width=0.5

lumina_cosmology

# %%
#find volume integration
def calculate_field_volume(cosmology, z_center, delta_z, survey_area):
    z_min=z_center-delta_z/2
    z_max=z_center+delta_z/2

    def differential_volume(z):
        return cosmology.differential_comoving_volume(z).value
    
    volume,error= quad(differential_volume,z_min,z_max) #numerical values
    field_solid_angle=survey_area.to(u.sr)

    field_volume=volume*u.Mpc**3/u.sr*field_solid_angle #Mpc^3 units
    field_volume_error=error*u.Mpc**3/u.sr*field_solid_angle #Mpc^3 units

    los_depth=cosmology.comoving_distance(z_max)-cosmology.comoving_distance(z_min)

    return field_volume,field_volume_error,los_depth

# %%
def load_data(base_path, snap_num):
    catalog_path = groupcat.gcPath(base_path, snap_num)
    halos = groupcat.loadHalos(base_path,snap_num,fields=["GroupPos", "Group_M_Crit200", "GroupLen"],)
    with h5py.File(catalog_path, "r") as f:
        h = f["Parameters"].attrs["HubbleParam"]
        box_size_raw = f["Header"].attrs["BoxSize"]
        redshift = f["Header"].attrs["Redshift"]
    box_size_cmpc = box_size_raw / (1000 * h)

    print("h:", h)
    print("redshift:", redshift)
    print("box size:", box_size_cmpc, "cMpc")

    return {
        "positions_raw": halos["GroupPos"],
        "m200c_raw": halos["Group_M_Crit200"],
        "particle_counts": halos["GroupLen"],
        "h": h,
        "redshift": redshift,
        "box_size_cmpc": box_size_cmpc,
        "catalog_path": catalog_path,
    }

# %%
base_path = "data/Lumina_above_z_4p75/group_files"
snap_num = 36
data = load_data(base_path, snap_num)
positions = data["positions_raw"]
m200c = data["m200c_raw"]
particle_counts = data["particle_counts"]
h=data["h"]
res_cut = 20 #minimum dm particle, ask later

valid = ((particle_counts >= res_cut) & (m200c > 0) & np.isfinite(m200c) & np.all(np.isfinite(positions), axis=1))

print("All FoF groups:", len(m200c))
print("Selected halos:", valid.sum())
print("Selected fraction:", valid.mean())

positions_cmpc = positions[valid] / (1000 * h)
m200c_msun = m200c[valid] * 1e10 / h

# %%
#box splitting
field_volume, field_volume_error, los_depth = calculate_field_volume(cosmology=lumina_cosmology,z_center=data["redshift"],delta_z=delta_z,survey_area=survey_area,)
print("field volume:", field_volume)
print("field volume error",field_volume_error)
print("los depth", los_depth)

#size of each box

#rectangle (pencil beam)
cross_sectional_area=field_volume/los_depth
length=np.sqrt(cross_sectional_area)
rectangle_dim=(length,length,los_depth)

#cube
cube_length=np.cbrt(field_volume)
cube_dim=(cube_length,cube_length,cube_length)
cube_side_cmpc = np.cbrt(field_volume.to_value(u.Mpc**3))
n_side = int(np.floor(data["box_size_cmpc"]/cube_side_cmpc))

print("cube side:", cube_side_cmpc, "cMpc")
print("cubes per dimension:", n_side)
print("field volume:", field_volume)
print("rectangle (pencil beam) dim:", rectangle_dim)
print("cube dim:", cube_dim)

""" 
How we assign halos to each field:
1. The simulation box is divided into non-overlapping cubes. Each cube is one field. The cube side length is chosen such that each cube has approximately the same volume as the target survey volume
2. Find all halos that are within the cube covered total volume (cubes don't fit in the total simulation volume perfectly)
3. For each halo position, divide by cube side length, and take the floor to define a cube index which this halo belongs to
4. Halos with the same cube are in the same field. A field can have zero, one or many halos
5. Flatten each cube index into a field id and count each repeated id within each mass bin (will do next) to build the count matrix
6. Fields with no halos are kept as zero rows for computation

"""

#assign halos to fields
covered_length = n_side * cube_side_cmpc
inside = np.all(positions_cmpc < covered_length, axis=1)
field_coord = np.floor(positions_cmpc[inside] / cube_side_cmpc).astype(int) 

# Flatten the 3D cube index into a single field ID based on the spatial geometry of the cube
# through row-major flattening: field _id = x*n_side^2 + y*n_side + z
field_ids = (field_coord[:,0]*n_side**2+field_coord[:,1]*n_side+field_coord[:,2])
assert field_ids.min()>=0
assert field_ids.max() < n_side**3
print("total fields:", n_side**3)
print("fields with at least one halo:", len(np.unique(field_ids)))
print("empty fields:", n_side**3 - len(np.unique(field_ids)))

# %%
"""
How we assign halos to mass bins:
1. For each halo inside the covered volume, we use its M200c mass in solar masses and bin them in log10 mass
2. A bin width of 0.5 dex means each bin is 0.5 wide in log10(M/Msun). This mean each bin is wider than the previous one by a factor of 10**0.5 ≈ 3.16
3. Each halo is assigned to exactly one mass bin. We use half-open bins: [lower edge, upper edge). A halo exactly on an upper edge is assigned to the next bin
4. A single mass bin can have multiple halos
5. We combine field ids and mass-bin ids to build the count matrix: count_matrix[field_id, mass_bin_id] gives the number of halos in that field and mass bin

"""

mass_bin_width = 0.5  # dex
masses_inside = m200c_msun[inside]
log_mass = np.log10(masses_inside)

log_mass_min = np.floor(log_mass.min() / mass_bin_width) * mass_bin_width #cleaner, integer edges
log_mass_max = np.ceil(log_mass.max() / mass_bin_width) * mass_bin_width
mass_bin_edges = np.arange(log_mass_min,log_mass_max + mass_bin_width,mass_bin_width,)
n_mass_bins = len(mass_bin_edges) - 1 #total bins. Must be 1 less than the number of edges
mass_bin_ids = np.full(len(log_mass), -1, dtype=int) #all mass bin ids of halos start with -1, meaning unassigned

#goes through each bin and find all halo masses in this bin instead of the opposite more better efficiency
for bin_id in range(n_mass_bins): 
    lower_edge = mass_bin_edges[bin_id]
    upper_edge = mass_bin_edges[bin_id + 1]

    in_this_bin = (
        (log_mass >= lower_edge)
        & (log_mass < upper_edge)
    )

    mass_bin_ids[in_this_bin] = bin_id

in_mass_range = mass_bin_ids >= 0 #makes sure every halo we work with is assigned a bin

print("log mass range:", log_mass_min, log_mass_max)
print("mass bin edges:", mass_bin_edges)
print("number of mass bins:", n_mass_bins)
print("halos with assigned mass bin:", in_mass_range.sum())

# %%
# now we build the field x mass-bin count 2d array

n_fields = n_side**3
n_mass_bins = len(mass_bin_edges) - 1
count_matrix = np.zeros((n_fields, n_mass_bins), dtype=int)

#same mask must be applied to both arrays to ensure alignment, then we add value 1 for each repeating (field id, mass bin id) pairs
for field_id, mass_bin_id in zip(field_ids[in_mass_range], mass_bin_ids[in_mass_range]):
    count_matrix[field_id, mass_bin_id] += 1
print("count matrix shape:", count_matrix.shape)
print("total counted halos:", count_matrix.sum())
print("halos used:", in_mass_range.sum())

assert count_matrix.sum() == in_mass_range.sum()


