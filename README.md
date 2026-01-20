# Dark Sector Memory Test (DSMT)

### Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-pilot%20analysis%20in%20progress-green.svg)]()

**A model-agnostic observational framework for testing kinematic-history dependence in gravitational lensing**

---

## 🔭 Overview

The **Dark Sector Memory Test** is a model-agnostic observational framework for testing whether gravitational lensing encodes kinematic history beyond the instantaneous matter distribution.

Multiple theoretical frameworks predict "memory" effects in the dark sector:

| Theory | Mechanism | Prediction |
|--------|-----------|------------|
| **Superfluid Dark Matter** | Phase transitions at critical velocity | Turbulent wakes (Sivakumar+ 2025) |
| **Non-Markovian EFT** | Memory kernels from heavy fields | Smooth power-law response (Chaudhuri+ 2025) |
| **Nonlocal Gravity** | Retarded stress-energy coupling | Logarithmic response (Maggiore+ 2014) |
| **ΛCDM** | No history dependence | No correlation (null hypothesis) |

**The key question:** Do lensing convergence residuals correlate with merger infall velocity?

Despite theoretical predictions, **no systematic observational test has been performed** — until now.

---

## 📊 Pilot Analysis Results (January 2026)

### Current Sample

We have analyzed **three merging galaxy clusters** with Hubble Frontier Fields convergence maps and published kinematic constraints:

| Cluster | z | v_infall (km/s) | Mach | Geometry | Kinematic Source |
|---------|---|-----------------|------|----------|------------------|
| **MACSJ0416** | 0.396 | 1600 ± 400 | — | Plane-of-sky | Jauzac+ 2015 |
| **Abell 2744** | 0.308 | 2000 ± 200 | 1.2 | Plane-of-sky | Chadayammuri+ 2024 |
| **Abell 370** | 0.375 | ~3000 (LOS) | — | **Line-of-sight** | Bimodal redshifts |

### Key Finding: Geometry Dependence

**Plane-of-sky mergers show a positive correlation between infall velocity and residual dipole moment:**

| Cluster | v (km/s) | Dipole | Asymmetry | Residual RMS |
|---------|----------|--------|-----------|--------------|
| MACSJ0416 | 1600 | 0.037 | 0.886 | 0.066 |
| Abell 2744 | 2000 | 0.044 | 0.934 | 0.087 |

**Abell 370 deviates from this trend** — but this is physically expected: its merger is largely along the line of sight, so any wake signature would be foreshortened in projection.

### Preliminary Interpretation

- For **plane-of-sky mergers**, higher infall velocity → larger dipole moment and asymmetry
- This is **consistent with wake signature predictions** from superfluid DM and nonlocal gravity
- **Line-of-sight mergers** require projection corrections before inclusion

### Next Steps

- **Abell 2146** (v = 2700 km/s, Mach = 2.3 ± 0.2) — best-constrained kinematics from shock measurements (Russell+ 2012). Convergence map requested from Coleman/King.
- With 3+ plane-of-sky mergers, we can compute statistically meaningful correlations

---

## 🎯 What This Project Does

1. **Reconstructs** lensing convergence maps from HST weak+strong lensing data
2. **Constructs** ΛCDM baseline models via parametric fitting (Gaussian smoothing)
3. **Computes** residual maps: Δκ = κ_obs − κ_baseline
4. **Measures** six morphological metrics quantifying residual structure
5. **Tests** for correlations between metrics and published merger kinematics

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/LBCplus/Dark-Sector-Memory-Test.git
cd Dark-Sector-Memory-Test
pip install -r requirements.txt
```

### Download Data

```bash
# Download Frontier Fields convergence maps from MAST
python code/download_and_analyze.py --download --data-dir ./data
```

### Run Analysis

```bash
python code/download_and_analyze.py --analyze --data-dir ./data
```

---

## 📁 Repository Structure

```
Dark-Sector-Memory-Test/
├── README.md                       # You are here
├── LICENSE                         # MIT License
├── CITATION.cff                    # Citation metadata
├── requirements.txt                # Python dependencies
│
├── paper/
│   ├── dsmt_paper_draft.md         # Manuscript draft
│   └── figures/                    # Publication figures
│
├── code/
│   ├── dsmt_analysis.py            # Main analysis module (~700 lines)
│   └── download_and_analyze.py     # Data pipeline with published kinematics
│
├── docs/
│   ├── methodology.md              # Detailed methods
│   └── statistical_analysis_plan.md # Pre-specified analysis
│
├── configs/
│   └── pilot_study.yaml            # Analysis configuration with kinematic params
│
└── data/                           # Data directory (not tracked)
    └── .gitkeep
```

---

## 📐 Morphological Metrics

We quantify lensing residual structure using six metrics:

| Metric | Symbol | Definition | Interpretation |
|--------|--------|------------|----------------|
| **Dipole moment** | \|d\| | ∫ Δκ(x) **x** d²x | Preferred direction of excess mass |
| **Quadrupole** | Q | Eigenvalue ratio of Q_ij | Elongation of residuals |
| **Tail-alignment** | T | cos(2(θ_res − θ_merger)) | Alignment with merger axis |
| **Asymmetry** | A | Σ\|Δκ − Δκ_180°\| / 2Σ\|Δκ\| | Departure from point symmetry |
| **Centroid offset** | \|Δx_c\| | \|x_obs − x_baseline\| | Mass center displacement |
| **Power spectrum** | P_tot | ∫ P(k) dk | Total residual structure |

---

## 📊 Pilot Sample

| Cluster | z | v_infall (km/s) | Data Source | Status |
|---------|---|-----------------|-------------|--------|
| **MACSJ0416** | 0.396 | 1600 ± 400 | HFF CATS v4 | ✅ Analyzed |
| **Abell 2744** | 0.308 | 2000 ± 200 | HFF CATS v4.1 | ✅ Analyzed |
| **Abell 370** | 0.375 | ~3000 (LOS) | HFF CATS v4 | ✅ Analyzed (LOS geometry) |
| **Abell 2146** | 0.232 | 2700 (+400/−300) | Coleman+ 2017 | ⏳ Data requested |

### Kinematic Sources

- **MACSJ0416**: Jauzac et al. 2015, MNRAS 446, 4132
- **Abell 2744**: Chadayammuri et al. 2024, arXiv:2407.03142 (2.1 Ms Chandra + JWST)
- **Abell 370**: Bimodal galaxy redshift distribution (~3000 km/s separation)
- **Abell 2146**: Russell et al. 2012, MNRAS 423, 236 (shock Mach numbers)

---

## 🎓 Theoretical Background

### Why "Memory"?

Standard ΛCDM predicts that lensing depends only on the **current** matter distribution. Several beyond-ΛCDM theories predict dependence on **kinematic history**:

**Superfluid Dark Matter** (Berezhiani & Khoury 2015)
> "Merger dynamics depend on the infall velocity versus phonon sound speed; distinct mass peaks in bullet-like cluster mergers correspond to superfluid and normal components."

**Sivakumar et al. (2025)** — Most direct prediction:
> "Merger-induced turbulence should produce asymmetric, fine-structure residuals in lensing maps, **correlated with infall velocity**."

**Non-Markovian EFT** (Chaudhuri et al. 2025)
> Memory kernels from integrated-out heavy fields produce history-dependent gravitational response.

**Nonlocal Gravity** (Maggiore & Mancarella 2014)
> Past stress-energy contributes to present gravitational dynamics via retarded Green's functions.

### The Gap We Address

Cognola et al. (2022) tested nonlocal gravity against cluster lensing and found it **indistinguishable from GR** using standard methods, concluding that "a different discriminator is needed."

**DSMT is that discriminator.**

---

## 📚 Key References

### Theoretical Foundations
- Berezhiani & Khoury (2015) — Superfluid DM framework — [PRD 92, 103510](https://doi.org/10.1103/PhysRevD.92.103510)
- Sivakumar et al. (2025) — Turbulent mergers — [PRD 111, 083511](https://doi.org/10.1103/PhysRevD.111.083511)
- Chaudhuri et al. (2025) — Non-Markovian EFT — [arXiv:2509.22293](https://arxiv.org/abs/2509.22293)
- Maggiore & Mancarella (2014) — Nonlocal gravity — [PRD 90, 023005](https://doi.org/10.1103/PhysRevD.90.023005)

### Observational Data
- Grayson et al. (2024) — MACSJ0416 BUFFALO model — [MNRAS 536, 2690](https://doi.org/10.1093/mnras/stae2123)
- Russell et al. (2012) — Abell 2146 kinematics — [MNRAS 423, 236](https://doi.org/10.1111/j.1365-2966.2012.20808.x)
- Chadayammuri et al. (2024) — Abell 2744 multiwavelength — [arXiv:2407.03142](https://arxiv.org/abs/2407.03142)
- Coleman et al. (2017) — Abell 2146 strong lensing — [MNRAS 464, 2469](https://doi.org/10.1093/mnras/stw2493)

### Gap Identification
- Cognola et al. (2022) — Nonlocal gravity vs. GR degeneracy — [arXiv:2205.03216](https://arxiv.org/abs/2205.03216)

---

## 🤝 Contributing

Contributions welcome! Particularly interested in:
- **Additional cluster convergence maps** with published kinematic constraints
- Simulation comparisons (TNG-Cluster, BAHAMAS)
- Statistical methodology improvements
- Theoretical predictions from other frameworks

Please open an issue or submit a pull request.

---

## 📄 Citation

If you use this code or methodology, please cite:

```bibtex
@software{dsmt2026,
  author = {[Author]},
  title = {Dark Sector Memory Test: Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers},
  year = {2026},
  url = {https://github.com/LBCplus/Dark-Sector-Memory-Test}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- HST Frontier Fields and BUFFALO teams for public lensing data
- MAST archive for data hosting
- Chandra X-ray Observatory for archival data
- The lensing community for published convergence maps

---

<p align="center">
  <i>Testing whether gravity remembers</i>
</p>
