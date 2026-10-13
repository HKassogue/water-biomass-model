# Nonautonomous interaction model: water–biomass dynamics

Reproducible numerical experiments accompanying the research article
**“Bistability in a water–biomass model with
truncated inhibition and a nonautonomous
crop-cycle illustration.”**

The repository includes the original Jupyter notebooks and corresponding standalone Python scripts.

## Contents

| File | Description |
|---|---|
| `simulation-autonomous.ipynb` | Autonomous model, equilibria, phase portraits, and initial-condition sensitivity |
| `simulation-nonautonomous.ipynb` | Nonautonomous water–biomass model, crop-stage coefficients, irrigation sensitivity, and initial-condition sensitivity |
| `simulation-autonomous.py` | Terminal version of the autonomous experiments |
| `simulation-nonautonomous.py` | Terminal version of the nonautonomous experiments |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |

## Installation

Python **3.10 or later** is recommended.

```bash
git clone https://github.com/HKassogue/water-biomass-model.git
cd water-biomass-model
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running in a terminal

```bash
python simulation-autonomous.py
python simulation-nonautonomous.py
```

Both scripts run headlessly and automatically save figures as numbered PNG files under `figs/`, e.g. `figs/autonomous_001.png` and `figs/nonautonomous_001.png`. The folder is created automatically.

Use a different output directory if needed:

```bash
python simulation-autonomous.py --fig-dir results/autonomous
python simulation-nonautonomous.py --fig-dir results/nonautonomous
```

**Note:** When both scripts use the same `figs/` folder, their distinct filename prefixes prevent collisions. Re-running a script overwrites that script's previously numbered images.

## Running the notebooks

```bash
jupyter lab
```

Open either `.ipynb` file and select **Run All**. The notebooks retain their original interactive plotting behavior; the `.py` scripts redirect plots to PNG files.

## Models

The applied water–biomass model is
```math
\begin{cases}
\dot e(t)&=-\alpha(t)e(t)-\beta(t)\mathcal I(t,e(t),z(t))+u(t),\\
\dot z(t)&=-\gamma(t)z(t)+\delta(t)\mathcal I(t,e(t),z(t)),
\end{cases}
```

with

```math
\mathcal I(t,e,z)=e\left(1-\frac{e}{e_{\max}(t)}\right)_+z.
```

The autonomous experiments use constant coefficients. The nonautonomous experiments use time-varying crop-stage coefficients and examine the effect of irrigation deficit compensation, parameterized by $\kappa$.

The numerical values are research-model assumptions and should not be interpreted as calibrated agronomic recommendations.

## Reproducibility notes

- The scripts are derived from the executable code cells of the supplied final notebooks, in notebook order.
- Figures are exported with Matplotlib using a noninteractive backend.
- Figures are numbered in execution order rather than assigned the original article's figure identifiers.
- All executable notebook cells were syntax-checked, and both scripts were run successfully.
- Results can vary slightly across dependency versions and numerical solvers.

## Citation

Please cite the associated article when using or adapting this code. 

## License

GNU GENERAL PUBLIC LICENSE.
