# temp-tune

This repository contains the code and materials for the paper “Understanding temperature tuning in energy-based models” by Peter William Fields, Vudtiwat Ngampruetikorn, David J. Schwab, and Stephanie E. Palmer. arXiv ID: 2512.09152, ( https://www.arxiv.org/pdf/2512.09152 )

---

This code base uses the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to maintain a fully reproducible scientific project named

> **temp-tune**

The project also relies on **PyCall + PyPlot + a project-local Conda environment**
for all Python/Matplotlib functionality.

---

## Reproducing this project locally

The Jupyter notebooks use IJulia. This package is not in the project environment, so ensure that IJulia is installed in your global Julia environment if it is not already (e.g. via Pkg.add("IJulia")) before running the notebooks or activating the project. 

To (locally) reproduce this project, do the following.

#### 0. Obtain the code

Clone or download this repository.

---

#### 1. Open a Julia console and do:

```julia
using Pkg
Pkg.add("DrWatson")          # install globally, for using `quickactivate`
Pkg.activate("path/to/temp-tune")
Pkg.instantiate()

cd("path/to/temp-tune")
include(scriptsdir("setup_python.jl"))
```

This will:

- activate the `temp-tune` project
- instantiate all Julia dependencies
- set up the project-local Conda environment at `./.conda`
- set up the project-local Matplotlib config at `./.mplconfig`
- rebuild `PyCall` against the project-local Conda Python
- install `matplotlib` into that same Conda environment

**Important:**

- `path/to/temp-tune` must be the root of this repository.
- This setup step is **required once per clone**.
- After this completes, **restart Julia** before doing any analysis.
- The directory .conda/pkgs is a cache and may be safely deleted at any time to reclaim disk space.

---

## Data availability

The simulation data for both figures is hosted on Hugging Face. From the **root** of your cloned `temp-tune` repository, run:

```bash
# install git-lfs if not already installed
git lfs install

# clone the dataset into ./data (creates both required subfolders)
git clone https://huggingface.co/datasets/peter-fields/temp-tune-data data
```
This produces the required directory structure:
```
temp-tune/
└── data/
    ├── ising_sweeps/         # nearest-neighbor Ising sweeps  (Fig. 3)
    │   └── constant_M_*/ ...
    └── simple_model_sweeps/  # two-level toy-model sweeps     (Fig. 2)
        └── Nstates=*_nground=*.jld2
```
> **Note:** run this from the repository root. If a `data/` directory already exists, clone into a temporary location and move `ising_sweeps/` and `simple_model_sweeps/` into `data/`. (The dataset also brings its own `README.md` and `.gitattributes` into `data/`; these are harmless.)

---

## Minimum working examples

Can be found for both nearest-neighbor Ising distribution and toy model experiments at:
```
temp-tune/notebooks/min_working_examples.ipynb
```

---

## Supplemental (appendix) figures

Notebooks reproducing the appendix figures live in `notebooks/supplemental/`, driven by the small
summary dicts committed alongside them (no separate download needed):

- **`toy_model_appendix.ipynb`** — `newfig-working-1.pdf` (Appendix C, toy model: raising vs.
  lowering τ, κ vs. C). Data: `newfigdict.jld2`, `kappa_C_plot_dict.jld2`.
- **`nn_ising_kappa.ipynb`** — the full `nn_ising_kappa_C.pdf` (Appendix C), **panels a–e**:
  per-level reversed D_KL and its τ-derivative at low/high T (a–d, from `hld_dict_for_last_figs.jld2`)
  and C(τ=1)/κ(τ=1) vs ground-truth T (e). **Panel e recomputes from the fitted 4×4 models, so this
  notebook needs the Ising sweeps in `data/ising_sweeps/constant_M_4by4/`** (see Data availability
  above) and loads the heavier Ising `src`.
- **`ising_appendix.ipynb`** — the Appendix-D figures: **`supp_fig_working-new.pdf`**
  (mean-over-replicates D_KL vs M for several T) and the **data panels of `fig4-working-latest.pdf`**
  (per-state probabilities, per-state reversed-D_KL contribution, per-level D_KL decomposition).
  Data: `all_temp_level_dict.jld2`, `hld_dict_for_last_figs.jld2`. Light (plots from dicts only).

These reproduce the underlying panels; the published PDFs were composited in Inkscape. Reproduced
SVGs are written to `plots/supplemental/`.

The per-level D_KL data (`all_temp_level_dict.jld2`) came from an **independent re-sampling +
re-fitting sweep** (not a decomposition of the raw sweeps on Hugging Face), so it is committed here
directly. The only omission from the published figures is the faint energy-level gridlines in fig4
Panel A (from `make_hist_edges`), left out as cosmetic.

---

## Using this project

Start each script/notebook with:

```julia
using DrWatson
@quickactivate "temp-tune"
```

to activate the project and ensure local path handling is correct. 

---

## Using Python / Matplotlib in this project

After the one-time setup is complete, initialize Python/Matplotlib in any script
or notebook with:

```julia
using DrWatson
@quickactivate "temp-tune"
include(srcdir("matplotlib_helpers.jl"))

setup_notebook_for_paper()
using_Py(true)
```

This ensures that:

- PyCall is bound to the project-local Conda environment
- Matplotlib uses the project-local `.mplconfig`
- all plotting settings are consistent and reproducible

A project local install matplotlib will be used via the Julia Conda package. 

---

## Notes on reproducibility

- The use of a **project-local Conda environment** is intentional and required for
  reproducibility.
- Do **not** use your system Python, a global Conda install, or another project’s
  Conda environment with this project.
- All Python dependencies must be installed into `temp-tune/.conda`.
- If you encounter Python/Matplotlib issues, re-running

  ```julia
  include(scriptsdir("setup_python.jl"))
  ```

  in a **fresh Julia session** usually resolves them.

---

