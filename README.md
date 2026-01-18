# Dark Sector Memory Test (DSMT)

### Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-coming%20soon-blue.svg)](https://zenodo.org/)

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

## 🎯 What This Project Does

1. **Reconstructs** lensing convergence maps from HST weak+strong lensing data
2. **Constructs** ΛCDM baseline models via parametric fitting and simulation matching
3. **Computes** residual maps: Δκ = κ_obs − κ_baseline
4. **Measures** six morphological metrics quantifying residual structure
5. **Infers** merger kinematic parameters via Bayesian forward-modeling
6. **Tests** for correlations that exceed ΛCDM expectations

---

## 📊 Key Results

*[Results will be added as analysis progresses]*

**Pilot Sample:** 5 merging clusters including MACSJ0416 (237 spectroscopic multiple images — best-constrained lens ever)

**Statistical Power:** 
- N=5: Can detect very large effects (|r| > 0.85) at 70% power
- N=30: Can detect moderate effects (|r| > 0.5) at 80% power

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/[username]/dark-sector-memory-test.git
cd dark-sector-memory-test
pip install -r requirements.txt
```

### Run Demonstration

```bash
python code/dsmt_analysis.py --demo
```

This runs the complete pipeline on synthetic data with an injected wake signal.

### Analyze Real Data

```bash
# Download Frontier Fields convergence maps
python code/download_data.py --cluster macs0416

# Run full analysis
python code/dsmt_analysis.py --cluster macs0416 --config configs/pilot_study.yaml
```

---

## 📁 Repository Structure

```
dark-sector-memory-test/
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
│   ├── dsmt_analysis.py            # Main analysis module
│   ├── download_data.py            # MAST/archive data fetching
│   ├── morphology_metrics.py       # Metric computation
│   ├── kinematic_inference.py      # Bayesian parameter estimation
│   └── statistical_tests.py        # Correlation & bootstrap analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore convergence maps
│   ├── 02_metric_validation.ipynb  # Validate metrics on simulations
│   └── 03_full_analysis.ipynb      # Complete pipeline walkthrough
│
├── docs/
│   ├── methodology.md              # Detailed methods
│   ├── statistical_analysis_plan.md # Pre-specified analysis
│   └── literature_review.md        # Theoretical background
│
├── configs/
│   └── pilot_study.yaml            # Analysis configuration
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

## 📊 Pilot Sample

| Cluster | z | Data Quality | Status |
|---------|---|--------------|--------|
| **MACSJ0416** | 0.396 | 237 spectroscopic images (BUFFALO) | Primary target |
| **Abell 2146** | 0.232 | 400 ks Chandra, Mach number measured | Kinematic benchmark |
| **JKCS041** | 1.95 | High-z merger, eROSITA | High-z test |
| **Abell 2744** | 0.308 | Complex multi-merger, HFF | Complexity test |
| **RX J2129** | 0.235 | CLASH + MUSE | Additional sample |

---

## 📚 Key References

### Theoretical Foundations
- Berezhiani & Khoury (2015) — Superfluid DM framework — [PRD 92, 103510](https://doi.org/10.1103/PhysRevD.92.103510)
- Sivakumar et al. (2025) — Turbulent mergers — [PRD 111, 083511](https://doi.org/10.1103/PhysRevD.111.083511)
- Chaudhuri et al. (2025) — Non-Markovian EFT — [arXiv:2509.22293](https://arxiv.org/abs/2509.22293)
- Maggiore & Mancarella (2014) — Nonlocal gravity — [PRD 90, 023005](https://doi.org/10.1103/PhysRevD.90.023005)

### Observational Context
- Grayson et al. (2024) — MACSJ0416 BUFFALO model — [MNRAS 536, 2690](https://doi.org/10.1093/mnras/stae2123)
- Russell et al. (2012) — Abell 2146 kinematics — [MNRAS 423, 236](https://doi.org/10.1111/j.1365-2966.2012.20808.x)
- ZuHone et al. (2018) — Galaxy Cluster Merger Catalog — [ApJS 234, 4](https://doi.org/10.3847/1538-4365/aa99dc)

### Gap Identification
- Cognola et al. (2022) — Nonlocal gravity vs. GR degeneracy — [arXiv:2205.03216](https://arxiv.org/abs/2205.03216)

---

## 🤝 Contributing

Contributions welcome! Particularly interested in:
- Additional cluster data reduction
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
  url = {https://github.com/[username]/dark-sector-memory-test}
}
```

Paper citation (when available):
```bibtex
@article{dsmt_paper2026,
  author = {[Authors]},
  title = {Dark Sector Memory Test: Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers},
  journal = {[Journal]},
  year = {2026}
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
- ESO/VLT MUSE team for spectroscopic data
- Galaxy Cluster Merger Catalog team (ZuHone et al.)

---

<p align="center">
  <i>Testing whether gravity remembers</i>
</p>
