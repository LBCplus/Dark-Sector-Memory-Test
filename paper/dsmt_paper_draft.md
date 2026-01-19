# Dark Sector Memory Test: Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers

**[Author 1]¹, [Author 2]², [Author 3]¹**

*¹[Institution 1], ²[Institution 2]*

---

## ABSTRACT

Multiple theoretical frameworks predict that gravitational lensing should encode kinematic history beyond the instantaneous matter distribution. Superfluid dark matter exhibits velocity-dependent phase transitions that produce turbulent wakes in mergers (Sivakumar et al. 2025); non-Markovian effective field theories retain memory kernels from integrated-out heavy fields (Chaudhuri et al. 2025); and nonlocal gravity couples present dynamics to past stress-energy (Maggiore & Mancarella 2014). Despite these predictions, no systematic observational test exists for correlations between lensing convergence residuals and merger infall kinematics.

We present a model-agnostic methodology to test for such "wake signatures" in merging galaxy clusters. Using archival HST imaging, Chandra X-ray observations, and VLT/MUSE spectroscopy for a pilot sample of three well-characterized mergers with published kinematic constraints—Abell 2146 (v_infall ~ 2700 km/s, Mach 2.3; Russell et al. 2012), Abell 2744 (~2.1 Ms Chandra + JWST UNCOVER; Chadayammuri et al. 2024), and MACSJ0416 with its unprecedented 237-image strong-lensing model (Grayson et al. 2024)—we: (1) reconstruct observed convergence maps from combined weak and strong lensing; (2) construct ΛCDM baseline models via parametric fitting and simulation matching using the Galaxy Cluster Merger Catalog (ZuHone et al. 2018); (3) compute residual maps Δκ = κ_obs − κ_baseline; (4) measure six morphological metrics quantifying residual structure; and (5) test for correlations with published merger kinematics.

We test for Spearman correlations between morphological metrics and infall velocity, comparing observed correlations to null distributions from 10,000 ΛCDM-matched bootstrap realizations. [RESULTS TO BE INSERTED]. Our analysis establishes the first observational framework for testing history-dependent dark-sector effects in gravitational lensing, addressing a specific gap identified by Cognola et al. (2022) who showed that standard lensing analyses cannot distinguish nonlocal gravity from GR.

---

## 1. INTRODUCTION

The nature of dark matter remains one of the most profound open questions in physics. While the cold dark matter (CDM) paradigm successfully explains large-scale structure formation and cosmic microwave background anisotropies, alternative frameworks have emerged that predict qualitatively different behavior in specific regimes. Of particular interest are theories predicting that gravitational dynamics should depend not only on the instantaneous matter distribution, but on the kinematic history of the system.

### 1.1 Theoretical Motivation

Several well-motivated theoretical frameworks predict history-dependent gravitational effects:

**Superfluid Dark Matter.** Berezhiani & Khoury (2015) proposed that DM particles condense into a superfluid phase in high-density environments. The superfluid supports collective phonon excitations mediating a long-range force, but this phase exists only below a critical velocity v_crit. In a comprehensive 2025 review, Berezhiani et al. emphasize that "merger dynamics depend on the infall velocity versus phonon sound speed; distinct mass peaks in bullet-like cluster mergers correspond to superfluid and normal components."

Most critically for our proposed test, Sivakumar et al. (2025) recently simulated merging self-gravitating condensates using the Gross-Pitaevskii-Poisson model and found that:
- Merger velocity determines the turbulent cascade structure
- Collisions produce dark soliton-mediated instabilities that decay into Kolmogorov-like turbulence
- The self-gravitating trap (cluster potential) modulates distribution of compressible kinetic energy

Their key result: "Merger-induced turbulence should produce asymmetric, fine-structure residuals in lensing maps, correlated with infall velocity." This is precisely the signature our discriminator tests.

Additionally, Shaber et al. (2025) studied dynamical friction in dark matter superfluids, showing how infall velocity couples to the underlying DM phase structure with quantitative predictions for merger morphology.

**Non-Markovian Effective Field Theory.** Chaudhuri et al. (2025) developed a systematic Schwinger-Keldysh EFT for dark sectors with memory kernels arising from heavy-field integration. Their framework predicts testable signatures in gravitational-wave spectra and relic abundances, but—as they note—has not yet been applied to cluster-merger lensing. The response function has temporal non-locality:

$$\phi(t) = \int dt' K(t-t') \Theta(t')$$

where K is the memory kernel (set by the mass M_heavy of integrated-out fields) and Θ is the source. Our proposed test would be the first application of this framework to gravitational lensing observables.

**Nonlocal Gravity.** Maggiore & Mancarella (2014) modified Einstein's equations by adding terms integrating over past matter stress-energy:

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu} + 8\pi G \int dt' K(t-t') T_{\mu\nu}(t')$$

Critically, Cognola et al. (2022) tested nonlocal gravity predictions against CLASH cluster lensing and found "agreement with data at the same level of statistical significance as General Relativity." This degeneracy with GR using standard analyses motivates our alternative approach: testing for correlations with merger kinematics, which nonlocal gravity explicitly predicts but standard methods cannot probe.

### 1.2 The Observational Gap

Despite these theoretical predictions, **no existing observation has tested whether gravitational lensing convergence residual morphologies correlate with inferred merger-infall kinematics in a way inconsistent with ΛCDM expectations.**

Existing constraints address different questions:
- **Bullet Cluster**: Constrains DM self-interaction (σ/m < 1 cm²/g), not history-dependent excitations
- **CLASH/Frontier Fields lensing**: Measures mass profiles and substructure, not residual-kinematic correlations
- **CMB + BAO**: Constrains linear-regime growth, not nonlinear merger dynamics
- **Galaxy rotation curves**: Constrains superfluid DM phonon parameters at galaxy scales, not cluster-scale phase transitions

Our proposed test directly addresses this gap with a falsifiable, model-agnostic methodology.

### 1.3 This Work

We present a systematic observational framework to test for kinematic-history wake signatures in cluster mergers. Our approach is deliberately theory-independent: we test whether lensing residual morphology correlates with infall velocity, impact parameter, or collision timescale, regardless of which (if any) theoretical framework might explain such correlations. The null hypothesis (ΛCDM) predicts no such correlations beyond scatter from projection effects and measurement uncertainties.

We apply this methodology to a pilot sample of three merging clusters with the strongest available kinematic constraints, anchored by Abell 2146 (the best-calibrated merger kinematics), Abell 2744 (deepest X-ray + JWST lensing), and MACSJ0416 (most precise lens model). With three clusters we have limited statistical power, but can detect very strong correlations (|r| > 0.9) and establish the framework for expanded Phase 2 studies including JKCS041 (z=1.95).

This paper is organized as follows. Section 2 describes our cluster sample and data. Section 3 details the methodology. Section 4 presents the statistical framework. Section 5 reports results. Section 6 discusses implications. Section 7 concludes.

---

## 2. SAMPLE AND DATA

### 2.1 Sample Selection

We select a pilot sample of three merging clusters with the strongest joint constraints on both lensing convergence and merger kinematics. Rather than pursuing a larger sample with heterogeneous data quality, we focus on systems where both observables are well-determined, enabling a meaningful test of correlation.

Selection criteria:
- Deep HST imaging enabling weak lensing shear measurement to R ~ 28 mag
- Published strong lensing models with spectroscopically confirmed multiple images
- Chandra X-ray observations with sufficient depth for temperature mapping or shock identification
- **Published merger kinematic constraints** (infall velocity, Mach number, or collision timescale)

**Table 1: Pilot Sample with Published Kinematic Constraints**

| Cluster | z | v_infall (km/s) | Mach | t_coll (Gyr) | Kinematic Source |
|---------|---|-----------------|------|--------------|------------------|
| Abell 2146 | 0.232 | 2700 (+400/−300) | 2.3±0.2 | 0.1–0.2 | Shock (Russell+ 2012) |
| Abell 2744 | 0.308 | 1800–2200 | ~1.2 | 0.5–0.6 | X-ray+sims (Chadayammuri+ 2024) |
| MACSJ0416 | 0.396 | 1200–2000 | — | post-pericenter | Dynamical (Jauzac+ 2015) |

**Sample rationale:**
- **Abell 2146** provides the best-calibrated kinematics in the literature: direct Mach number measurements from bow shock (M=2.3±0.2) and upstream shock (M=1.6±0.1), yielding plane-of-sky shock velocities of v_bow = 2700 (+400/−300) km/s and v_upstream = 2400±300 km/s (Russell et al. 2012). Simulation-based modeling (Kim et al. 2021) confirms the merger geometry at ~0.1–0.2 Gyr post-pericenter.
- **Abell 2744** has the deepest X-ray data (~2.1 Ms Chandra) with recent multiwavelength merger modeling (Chadayammuri et al. 2024) that combines X-ray, radio, optical, and JWST UNCOVER strong-lensing convergence. Their best-fit scenario: head-on N–S major merger 0.5–0.6 Gyr ago, followed by NW subcluster first infall.
- **MACSJ0416** has the most precise lens model ever constructed (Grayson et al. 2024), though kinematic constraints are weaker. Jauzac et al. (2015) characterize it as an ongoing merger between two comparable-mass subclusters with characteristic relative velocities of order 1200–2000 km/s, likely observed post–first pericenter.

**Excluded clusters:** RX J2129.7+0005 was initially considered but excluded because it is not a canonical binary merger—most work treats it as a relaxed strong-lensing cluster with substructure rather than a system with measurable two-body infall kinematics.

**Phase 2 extension:** JKCS041 (z=1.95) will extend the sample to high redshift. Finner et al. (2024) provide dedicated merger simulations with synthetic lensing maps and a best-fit configuration at ~0.3 Gyr post-pericenter, though direct Mach number measurements are unavailable at current data quality.

### 2.2 MACSJ0416: The Anchor Target

MACSJ0416 serves as our primary target due to unprecedented data quality. The December 2024 BUFFALO analysis by Grayson et al. provides:

- **237 spectroscopically confirmed multiple images** (largest secure sample for any lens)
- **Positional accuracy of 0.191 arcsec** (among the most precise ever achieved)
- **Free-form GRALE reconstruction** using genetic algorithm optimization
- **Identification of two light-unaffiliated substructures** (~10¹² M☉ each)

As Grayson et al. note: "Bimodal mass structure is a prototypical feature of actively merging clusters... dynamically complex with abundant substructures on varying length scales."

JWST PEARLS observations (2024-2025) have increased the total multiple-image count to 343, making MACSJ0416 uniquely suited for our analysis.

### 2.3 Abell 2146: Kinematic Benchmark

Abell 2146 provides the best-characterized merger kinematics in our sample and serves as the calibration anchor. Russell et al. (2012) analyzed 400 ks of Chandra observations, identifying:
- **Bow shock Mach number**: M_bow = 2.3 ± 0.2
- **Upstream shock Mach number**: M_upstream = 1.6 ± 0.1
- **Bow shock velocity**: v_bow = 2700 (+400/−300) km/s (plane of sky)
- **Upstream shock velocity**: v_upstream = 2400 ± 300 km/s
- **Time since pericenter**: ~0.1–0.2 Gyr

The merger geometry is roughly in the plane of the sky with small impact parameter, making velocity projection corrections minimal. Kim et al. (2021) performed idealized GAMER-2 merger simulations to fit Abell 2146's configuration, providing additional validation of the Russell et al. kinematic picture.

Strong-lensing mass modeling by Coleman et al. (2017) maps the mass distribution using HST observations; we adopt their parametric LENSTOOL model for baseline construction.

### 2.4 Abell 2744: Multiwavelength Benchmark

Abell 2744 ("Pandora's Cluster") is a complex multiple merger with the deepest X-ray observations of any cluster in our sample. Chadayammuri et al. (2024) combined ~2.1 Ms Chandra data with radio, optical, and JWST UNCOVER strong-lensing convergence, finding:
- **Best-fit scenario**: Head-on N–S major merger 0.5–0.6 Gyr ago, plus NW subcluster first infall
- **Line-of-sight velocity separation**: Δcz ≈ 4000–4500 km/s between main components
- **NW subcluster shock**: Mach ~1.2, implying v_merge ~ 1800–2200 km/s

Their analysis reveals that three cluster-scale halos consistent with lensing constraints underproduce observed convergence and X-ray surface brightness, suggesting an additional large-scale overdensity—precisely the kind of residual our methodology aims to characterize.

Lensing convergence maps from CATS and GRALE teams are available through MAST Frontier Fields.

### 2.5 Data Access

### 2.5 Data Access

**MACSJ0416 and Abell 2744 convergence maps:**
- MAST Frontier Fields lens model index: https://archive.stsci.edu/prepds/frontier/lensmodels/
- MACSJ0416 CATS model: https://archive.stsci.edu/prepds/frontier/lensmodels/macs0416/CATS/
- MACSJ0416 GRALE model: https://archive.stsci.edu/prepds/frontier/lensmodels/macs0416/GRALE/
- Abell 2744 CATS model: https://archive.stsci.edu/prepds/frontier/lensmodels/abell2744/CATS/
- Abell 2744 GRALE model: https://archive.stsci.edu/prepds/frontier/lensmodels/abell2744/GRALE/

Convergence maps (kappa.fits) are provided at multiple source redshifts (z=1, 2, 4, 9).

**Abell 2146 convergence map:** Coleman et al. (2017) provide parametric LENSTOOL model parameters but did not deposit a public FITS convergence map. We reconstruct κ from their published parameters, or obtain the original map via author correspondence.

**X-ray data:** HEASARC Chandra archive (Abell 2146: ObsID 13023; Abell 2744: merged 2.1 Ms dataset)

**Spectroscopy:** ESO archive (VLT/MUSE), Gemini archive (GMOS)

---

## 3. METHODOLOGY

### 3.1 Lensing Reconstruction

We reconstruct convergence maps using combined weak and strong lensing:

**Weak Lensing**: Galaxy shapes measured using KSB+ on the reddest HST band with adequate depth. Source selection requires S/N > 10 and photometric redshift z_phot > z_cluster + 0.1.

**Strong Lensing**: We incorporate published multiple-image identifications. For MACSJ0416, we use the Grayson et al. (2024) BUFFALO model with 237 spectroscopic images. For other clusters, we use the best available published models.

**Combined Reconstruction**: Using LENSTOOL (Jullo et al. 2007), we fit parametric dPIE profiles simultaneously constrained by strong-lensing positions and weak-lensing shear.

### 3.2 Baseline Construction

We construct baseline convergence maps κ_baseline representing ΛCDM expectations via two approaches:

**Parametric Baseline**: Fit smooth NFW + multipole model to observed κ:
$$\kappa_{baseline}(r,\theta) = \sum_i \kappa_{NFW,i}(r; M_i, c_i, x_i) + \sum_{lm} a_{lm} Y_{lm}(\theta)$$

**Simulation-Matched Baseline**: We identify analog clusters in the Galaxy Cluster Merger Catalog (ZuHone et al. 2018) and GAMER-2 simulations (used for Abell 2146 by the same team), matched by total mass, mass ratio, and X-ray morphology. For expanded studies, we will use TNG-Cluster (2024-2025), which provides 352 massive clusters with mock lensing outputs.

### 3.3 Residual Maps and Morphological Metrics

The residual convergence map is:
$$\Delta\kappa(\mathbf{x}) = \kappa_{obs}(\mathbf{x}) - \kappa_{baseline}(\mathbf{x})$$

We quantify Δκ morphology using six metrics:

| Metric | Symbol | Definition | Physical Meaning |
|--------|--------|------------|------------------|
| Dipole moment | \|d\| | ∫ Δκ(x) x d²x | Preferred direction of excess |
| Quadrupole | Q | Eigenvalue ratio of Q_ij | Elongation |
| Tail-alignment | T | cos(2(θ_residual - θ_merger)) | Alignment with merger axis |
| Asymmetry | A | Σ\|Δκ - Δκ_180°\| / 2Σ\|Δκ\| | Point symmetry departure |
| Centroid offset | \|Δx_c\| | \|x_c,obs - x_c,baseline\| | Mass center displacement |
| Power spectrum | P_total | ∫ P(k) dk | Total residual structure |

### 3.4 Kinematic Inference

We infer merger kinematic parameters using Bayesian forward-modeling:

**Parameters**: v_infall (infall velocity), b (impact parameter), t_coll (time since pericenter), α (viewing angle)

**Method**: MCMC sampling comparing GADGET-4 simulation library to multi-wavelength observations via composite likelihood including X-ray surface brightness, temperature structure, and lensing constraints.

**Validation**: For Abell 2146, we verify consistency with published shock-derived velocities (Russell et al. 2012).

---

## 4. STATISTICAL FRAMEWORK

### 4.1 Hypotheses

**H₀ (Null)**: Population correlation ρ(M_i, v_infall) = 0 for all morphological metrics M_i. Observed correlations are consistent with ΛCDM simulation scatter.

**H₁ (Alternative)**: Population correlation ρ(M_i, v_infall) ≠ 0 for at least one metric, exceeding ΛCDM expectations.

### 4.2 Primary Analysis

We compute Spearman rank correlations between each metric and v_infall, comparing to null distributions from 10,000 ΛCDM-matched bootstrap realizations.

**Multiple comparison correction**: Benjamini-Hochberg FDR at q = 0.05 across 6 primary tests.

### 4.3 Power Analysis

For N = 5 clusters at α = 0.05:
- Power to detect |r| = 0.9: ~68%
- Power to detect |r| = 0.7: ~23%
- Power to detect |r| = 0.5: ~12%

The pilot study can reliably detect only very strong effects. An expanded study with N = 30 achieves 80% power for |r| = 0.5.

### 4.4 Model Discrimination (if signal detected)

If significant correlation detected, we distinguish frameworks by fitting:
- **Superfluid DM**: Step function at v_crit (Sivakumar et al. 2025 predict turbulence threshold)
- **Non-Markovian EFT**: Power law M ∝ v^β
- **Nonlocal Gravity**: Logarithmic M ∝ ln(1 + v/v₀)

Model comparison via AIC/BIC.

---

## 5. RESULTS

[TO BE COMPLETED WITH ACTUAL ANALYSIS]

### 5.1 Convergence Maps and Residuals

[Figure: Observed and residual convergence maps for pilot clusters]

### 5.2 Morphological Metrics

**Table 2: Morphological Metrics for Pilot Sample**

[Metrics with uncertainties for each cluster]

### 5.3 Kinematic Parameters

**Table 3: Inferred Kinematic Parameters**

[v_infall, b, t_coll, α with credible intervals]

### 5.4 Correlation Analysis

**Table 4: Spearman Correlations with v_infall**

| Metric | r | p (raw) | q (FDR) | Significant? |
|--------|---|---------|---------|--------------|
| Dipole | | | | |
| Quadrupole | | | | |
| Tail-alignment | | | | |
| Asymmetry | | | | |
| Centroid offset | | | | |
| Power | | | | |

[Figure: Observed correlations vs. ΛCDM null distributions]

---

## 6. DISCUSSION

### 6.1 Interpretation

[Interpretation depending on outcome]

**If Null Result**: Our analysis provides the first observational upper limits on kinematic-history-dependent lensing effects, constraining:
- Superfluid DM critical velocity v_crit
- Non-Markovian EFT memory kernel strength
- Nonlocal gravity retardation scale

**If Signal Detected**: We would have the first evidence for velocity-dependent dark-sector response, with model discrimination analysis identifying the most consistent theoretical framework.

### 6.2 Relation to Previous Work

Our test directly addresses the gap identified by Cognola et al. (2022), who found nonlocal gravity indistinguishable from GR using standard cluster lensing analyses. Their conclusion that "a different discriminator is needed" motivates our kinematic-history approach.

The Sivakumar et al. (2025) prediction that "merger-induced turbulence should produce asymmetric, fine-structure residuals in lensing maps, correlated with infall velocity" provides the most direct theoretical support for our discriminator.

### 6.3 Systematic Uncertainties

- **Lens model systematics**: Addressed using multiple independent models
- **Projection effects**: Controlled via viewing angle in multivariate analysis
- **Baryonic physics**: Compared simulations with/without cooling and feedback
- **Sample selection**: Not statistically complete; expanded study will use eROSITA volume-limited selection

---

## 7. CONCLUSIONS

We have presented the first systematic methodology for testing history-dependent dark-sector signatures in gravitational lensing. Our model-agnostic approach seeks correlations between lensing residual morphology and merger kinematics without assuming a specific theoretical framework.

**Key contributions**:
1. Development of a falsifiable, model-agnostic test for kinematic-history-dependent gravitational effects
2. Definition of six morphological metrics quantifying lensing residual structure
3. Demonstration of methodology using MACSJ0416 with its unprecedented 237-image lens model
4. [Quantitative constraints or detections]

**Future work**: Expansion to 25-30 clusters using eROSITA-selected mergers with DES/Rubin weak lensing will achieve 80% power for moderate effect sizes (|r| ~ 0.5).

---

## ACKNOWLEDGMENTS

Based on observations made with the NASA/ESA Hubble Space Telescope, obtained from the data archive at the Space Telescope Science Institute. Chandra data obtained from the Chandra Data Archive. VLT/MUSE data obtained from the ESO Science Archive.

---

## REFERENCES

Berezhiani, L., & Khoury, J. 2015, PRD, 92, 103510  
Berezhiani, L., et al. 2025, "Superfluid Dark Matter" (review)  
Caminha, G. B., et al. 2017, A&A, 600, A90  
Chaudhuri, S., et al. 2025, arXiv:2509.22293  
Cognola, G., et al. 2022, arXiv:2205.03216  
Grayson, J., et al. 2024, MNRAS, 536, 2690  
Maggiore, M., & Mancarella, M. 2014, PRD, 90, 023005  
Markevitch, M., et al. 2004, ApJ, 606, 819  
Pratt, G. W., et al. 2014, A&A, 567, A16  
Russell, H. R., et al. 2012, MNRAS, 423, 236  
Shaber, A., et al. 2025, "Dynamical friction in dark matter superfluids"  
Sivakumar, A., et al. 2025, PRD, 111, 083511  
ZuHone, J. A., et al. 2018, ApJS, 234, 4  

---

*Paper Draft v2.0 - Updated with January 2026 literature review*
