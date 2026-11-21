# Lumina Cosmic Variance

Cosmic variance of UV-bright galaxies at z ≈ 10 in the LUMINA and LUMINA-EDE simulations, measured by counting galaxies in randomly placed and oriented NIRCam- and Roman-sized pencil beams.

## Layout

| Path | What it is |
| --- | --- |
| `notebooks/pencilbeam_v3.ipynb` | Main pipeline: UV-luminosity models, pencil-beam counts, σ_CV with trial errors, and the paper figures |
| `scripts/compute_UVLF_cv.py` | Catalogue loading, UV-luminosity model prescriptions (A1, B2, C1, C2, DMSFE), σ_CV and linear-bias helpers |
| `scripts/observation_data.py` | Observational UVLF compilations used for comparison |
| `scripts/compute_HMF_cv.py`, `scripts/compute_SMF_cv.py`, `scripts/plot_obs_smf.py` | Earlier halo- and stellar-mass-function cosmic variance analyses |
| `scripts/groupcat.py`, `scripts/snapshot.py`, `scripts/util.py` | Readers for AREPO/IllustrisTNG-style group catalogues |
| `theory.py`, `linearity.py`, `variance_error.py`, `check_linearity.py` | Linear-theory σ_m, beam-averaged correlation function, and analytic error estimates |
| `tests/` | `pytest` checks for the linear-theory code |
| `docs/derivation.md` | Derivation notes for the variance and error estimators |
| `slurm/` | Example cluster job scripts |

## Setup

```bash
conda env create -f environment.yml
conda activate lumina
```

## Data

Lumina catalogues and raw data from the DMSFE model are not shared here.

## Running

Run `notebooks/pencilbeam_v3.ipynb` once per instrument (`LUMINA_INSTRUMENT=NIRCam` and `LUMINA_INSTRUMENT=Roman`). Each run uses 40 trials of 2,000 beams; σ²_CV is averaged over trials before taking the square root.

```bash
pytest tests
```
