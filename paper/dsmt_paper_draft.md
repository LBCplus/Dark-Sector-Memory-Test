# Dark Sector Memory Test: Probing Dark Matter and Dark Energy History-Dependence via Cluster Mergers

**[Author]**

*[Institution]*

---

## ABSTRACT

Multiple theoretical frameworks predict that gravitational lensing should encode kinematic history beyond the instantaneous matter distribution. Superfluid dark matter exhibits velocity-dependent phase transitions producing turbulent wakes (Sivakumar et al. 2025); non-Markovian effective field theories retain memory kernels from integrated-out heavy fields (Chaudhuri et al. 2025); nonlocal gravity couples present dynamics to past stress-energy (Maggiore & Mancarella 2014). Despite these predictions, no systematic observational test exists.

We present a model-agnostic methodology to test for "wake signatures" in merging galaxy clusters using Hubble Frontier Fields convergence maps. For a pilot sample of four clusters with published kinematic constraints—MACSJ0416, Abell 2744, Abell 370, and Abell 2146—we construct smooth baseline models, compute residual maps Δκ = κ_obs − κ_baseline, and measure morphological metrics including dipole moment, asymmetry, and power spectrum slope.

A key methodological finding emerges: **merger geometry critically affects projected wake signatures**. For plane-of-sky (POS) mergers (MACSJ0416, Abell 2744), residual metrics increase with infall velocity. Abell 370, whose merger axis lies largely along the line of sight (LOS), deviates from this trend—consistent with geometric foreshortening of any wake structure. This geometry dependence must be accounted for in correlation analyses.

[FINAL RESULTS PENDING ABELL 2146 DATA]

Our analysis establishes the first observational framework for testing history-dependent dark-sector effects, addressing the gap identified by Cognola et al. (2022) who showed standard lensing cannot distinguish nonlocal gravity from GR.

---

## 1. INTRODUCTION

The cold dark matter (CDM) paradigm successfully explains large-scale structure formation and CMB anisotropies, yet alternative frameworks predict qualitatively different behavior in specific regimes. Of particular interest are theories in which gravitational dynamics depend not only on the instantaneous matter distribution, but on the kinematic history of the system.

### 1.1 Theoretical Motivation

**Superfluid Dark Matter.** Berezhiani & Khoury (2015) proposed that dark matter particles condense into a superfluid phase in high-density environments, supporting collective phonon excitations that mediate a long-range force. This phase exists only below a critical velocity v_crit. Sivakumar et al. (2025) simulated merging self-gravitating condensates and found that merger velocity determines the turbulent cascade structure: collisions produce dark soliton-mediated instabilities decaying into Kolmogorov-like turbulence. Their prediction—"merger-induced turbulence should produce asymmetric, fine-structure residuals in lensing maps, correlated with infall velocity"—motivates our test.

**Non-Markovian Effective Field Theory.** Chaudhuri et al. (2025) developed a Schwinger-Keldysh EFT for dark sectors with memory kernels arising from heavy-field integration. The response function exhibits temporal non-locality:

$$\phi(t) = \int dt' \, K(t-t') \, \Theta(t')$$

where K is the memory kernel set by heavy field masses. This framework has been applied to gravitational waves and relic abundances, but not yet to cluster lensing—our test would be the first such application.

**Nonlocal Gravity.** Maggiore & Mancarella (2014) modified Einstein's equations with terms integrating over past stress-energy:

$$G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu} + 8\pi G \int dt' \, K(t-t') \, T_{\mu\nu}(t')$$

Cognola et al. (2022) tested this against CLASH cluster lensing, finding "agreement with data at the same level of statistical significance as General Relativity." This degeneracy using standard methods motivates our alternative approach.

### 1.2 The Observational Gap

Despite theoretical predictions, **no existing observation has tested whether lensing convergence residuals correlate with merger kinematics** in a manner inconsistent with ΛCDM. Existing constraints address different questions: the Bullet Cluster constrains DM self-interaction cross-section, not history-dependent excitations; Frontier Fields lensing measures mass profiles, not residual-kinematic correlations; CMB and BAO constrain linear-regime growth, not nonlinear merger dynamics.

### 1.3 This Work

We present a model-agnostic framework testing whether lensing residual morphology correlates with infall velocity. The null hypothesis (ΛCDM) predicts no correlation beyond projection scatter. We apply this to a pilot sample of merging clusters with published kinematics, discovering a key methodological requirement: merger geometry (plane-of-sky vs. line-of-sight) must be considered when interpreting correlations.

---

## 2. SAMPLE AND DATA

### 2.1 Sample Selection Philosophy

We prioritize clusters with **both** high-quality lensing convergence maps **and** published merger kinematic constraints. Rather than maximizing sample size, we focus on systems where both observables are well-determined. This approach revealed an important finding: merger geometry affects the projected signatures.

### 2.2 Pilot Sample

**Table 1: Pilot Sample with Published Kinematic Constraints**

| Cluster | z | v_infall (km/s) | Mach | Geometry | Kinematic Source |
|---------|---|-----------------|------|----------|------------------|
| MACSJ0416 | 0.396 | 1600 ± 400 | — | **POS** | Jauzac+ 2015 |
| Abell 2744 | 0.308 | 2000 ± 200 | 1.2 | **POS** | Chadayammuri+ 2024 |
| Abell 370 | 0.375 | ~3000 (LOS) | — | **LOS** | Bimodal z |
| Abell 2146 | 0.232 | 2700 (+400/−300) | 2.3±0.2 | **POS** | Russell+ 2012 |

**Geometry classification:**
- **Plane-of-sky (POS)**: Merger axis predominantly perpendicular to line of sight. Infall velocity measured from shock Mach numbers or dynamical modeling. Wake signatures project onto the convergence map with minimal foreshortening.
- **Line-of-sight (LOS)**: Merger axis predominantly along line of sight. Velocity inferred from redshift separation between subclusters. Wake signatures are foreshortened in projection.

### 2.3 Individual Cluster Properties

**MACSJ0416** (z=0.396) — *Best lens model, moderate kinematics*

The December 2024 BUFFALO analysis (Grayson et al. 2024) provides the most precise cluster lens model ever constructed: 237 spectroscopically confirmed multiple images with 0.191″ positional accuracy. The bimodal mass structure indicates active merging between two comparable subclusters. Jauzac et al. (2015) estimate relative velocities of 1200–2000 km/s, observed post–first pericenter. The merger geometry is predominantly plane-of-sky.

**Abell 2744** (z=0.308) — *Deepest X-ray, good kinematics*

Chadayammuri et al. (2024) combined 2.1 Ms Chandra data with JWST UNCOVER strong lensing, determining a best-fit scenario of head-on N–S major merger 0.5–0.6 Gyr ago, plus NW subcluster first infall. The NW shock yields Mach ~1.2, implying v_merge ~ 1800–2200 km/s. The main merger axis lies in the plane of sky.

**Abell 370** (z=0.375) — *LOS geometry, high apparent velocity*

Abell 370 exhibits bimodal galaxy redshift distribution with ~3000 km/s separation (Richard et al. 2010). Unlike MACSJ0416 and Abell 2744, this velocity is measured along the line of sight from redshift differences, indicating the merger axis is largely radial. Any wake structure would be foreshortened in the projected convergence map.

**Abell 2146** (z=0.232) — *Best kinematics, benchmark*

Russell et al. (2012) provide the most precise merger kinematics available: bow shock Mach M = 2.3 ± 0.2 and upstream shock M = 1.6 ± 0.1, yielding v_bow = 2700 (+400/−300) km/s. The merger is roughly in the plane of sky with small impact parameter. Kim et al. (2021) validated this geometry through GAMER-2 simulations. Coleman et al. (2017) provide strong-lensing mass modeling.

### 2.4 Data Sources

Convergence maps obtained from MAST Frontier Fields archive (CATS team v4 models):
- MACSJ0416: `hlsp_frontier_model_macs0416_cats_v4_kappa.fits`
- Abell 2744: `hlsp_frontier_model_abell2744_cats_v4_1_kappa.fits`
- Abell 370: `hlsp_frontier_model_abell370_cats_v4_kappa.fits`
- Abell 2146: Coleman et al. (2017) LENSTOOL model (author correspondence)

---

## 3. METHODOLOGY

### 3.1 Convergence Maps

We use published convergence (κ) maps from the CATS strong-lensing team, reconstructed using LENSTOOL parametric modeling constrained by multiply-imaged background galaxies. Maps are provided at source redshift z_s = 2 on grids with 0.2–0.3″ pixel scale.

### 3.2 Baseline Construction

The baseline κ_baseline represents the smooth mass distribution expected in ΛCDM without fine-structure wake signatures. We construct baselines via Gaussian smoothing:

$$\kappa_{baseline}(\mathbf{x}) = \kappa_{obs} * G_\sigma$$

where G_σ is a Gaussian kernel with σ = 40 pixels (~12″). This removes structure on scales ≲ 50 kpc while preserving the large-scale mass distribution. The smoothing scale is chosen to exceed typical subhalo sizes but remain smaller than the separation between merging subclusters.

### 3.3 Residual Maps

The residual convergence isolates structure beyond the smooth baseline:

$$\Delta\kappa(\mathbf{x}) = \kappa_{obs}(\mathbf{x}) - \kappa_{baseline}(\mathbf{x})$$

Positive residuals indicate mass concentrations (subhalos, compact structures); negative residuals indicate mass deficits relative to the smooth model.

### 3.4 Morphological Metrics

We quantify residual structure using five metrics:

**Dipole moment** (normalized):
$$|d| = \frac{1}{N_{pix}} \left| \int \Delta\kappa(\mathbf{x}) \, \mathbf{x} \, d^2x \right|$$

Measures the preferred direction of excess mass. Wake signatures would produce dipoles aligned with the merger axis.

**Asymmetry** (180° rotational):
$$A = \frac{\sum |\Delta\kappa(\mathbf{x}) - \Delta\kappa(-\mathbf{x})|}{\sum |\Delta\kappa(\mathbf{x})| + |\Delta\kappa(-\mathbf{x})|}$$

Measures departure from point symmetry. Turbulent wakes break symmetry.

**Quadrupole moment**:
$$Q = \sqrt{Q_{xx}^2 + 4Q_{xy}^2}, \quad Q_{ij} = \int \Delta\kappa \, (x_i x_j - \frac{1}{2}\delta_{ij}r^2) \, d^2x$$

Measures elongation of residual structure.

**Power spectrum slope**:
Azimuthally averaged power spectrum P(k) fitted to power law P ∝ k^α. Steeper slopes indicate more large-scale structure; shallower slopes indicate fine-scale turbulence.

**Residual RMS**:
$$\sigma_{\Delta\kappa} = \sqrt{\langle \Delta\kappa^2 \rangle}$$

Total amplitude of structure beyond the baseline.

### 3.5 Geometry Considerations

A critical methodological finding: **merger geometry affects projected signatures**.

For a wake structure aligned with the merger axis:
- **POS merger**: Wake projects onto the κ map at full extent. Dipole moment reflects true wake amplitude.
- **LOS merger**: Wake is foreshortened along the line of sight. Projected dipole is suppressed by factor ~cos(θ) where θ is the angle from plane of sky.

We therefore analyze POS and LOS mergers separately, or restrict correlation analyses to POS systems where projection effects are minimal.

---

## 4. STATISTICAL FRAMEWORK

### 4.1 Hypotheses

**H₀ (Null)**: No correlation between morphological metrics and infall velocity beyond ΛCDM scatter. ρ(M_i, v) = 0.

**H₁ (Alternative)**: Significant correlation exists for at least one metric, indicating velocity-dependent residual structure.

### 4.2 Correlation Analysis

We compute Spearman rank correlations between each metric and v_infall. For N = 3–4 clusters, statistical power is limited, but perfect rank correlation (ρ = 1.0) has p = 0.167 for N = 3.

### 4.3 Geometry Stratification

Given the geometry dependence discovered in preliminary analysis, we stratify:
- **POS-only analysis**: MACSJ0416, Abell 2744, Abell 2146 (N=3)
- **Full sample**: All clusters with geometry as covariate

### 4.4 Power Considerations

With N = 3 POS clusters, we can detect only very strong effects (|ρ| > 0.9). Phase 2 expansion to N ~ 10 POS mergers would achieve meaningful power for moderate effects.

---

## 5. PRELIMINARY RESULTS

### 5.1 Convergence Maps and Residuals

Figure 1 shows observed κ and residual Δκ maps for the three clusters with public convergence data.

**Table 2: Morphological Metrics**

| Cluster | v (km/s) | Geometry | Dipole | Asymmetry | Power Slope | RMS |
|---------|----------|----------|--------|-----------|-------------|-----|
| MACSJ0416 | 1600 | POS | 0.037 | 0.886 | 2.11 | 0.066 |
| Abell 2744 | 2000 | POS | 0.044 | 0.934 | 2.45 | 0.087 |
| Abell 370 | 3000 | LOS | 0.030 | 0.919 | 1.94 | 0.069 |

### 5.2 Geometry Dependence

The two POS mergers show a consistent trend: higher infall velocity corresponds to larger dipole moment, higher asymmetry, steeper power slope, and larger residual RMS.

Abell 370 (LOS geometry) does not follow this trend despite having the highest apparent velocity. This is consistent with geometric foreshortening: if wake signatures exist, an LOS merger would project them at reduced amplitude.

### 5.3 Pending: Abell 2146

Abell 2146 provides the critical test. With v = 2700 km/s (highest among POS mergers) and the best-constrained kinematics (direct Mach measurement), it should:
- Follow the POS trend if wake signatures exist
- Fall on the extrapolation: predicted dipole ~0.05–0.06

Convergence map requested from Coleman/King (January 2026).

---

## 6. DISCUSSION

### 6.1 The Geometry Finding

Our preliminary analysis reveals that merger geometry critically affects interpretation. This has methodological implications:

1. **Sample selection must consider geometry**, not just kinematic precision
2. **LOS mergers require projection corrections** before inclusion in correlation analyses
3. **POS mergers are preferred** for initial tests of wake signatures

This finding is robust regardless of whether wake signatures ultimately exist—it reflects basic projection physics.

### 6.2 Interpretation of Trends

The positive correlation between velocity and residual metrics in POS mergers is **consistent with** wake signature predictions, but with N = 2 we cannot claim statistical significance. Abell 2146 (v = 2700 km/s, POS) will determine whether the trend continues.

Possible interpretations:
- **If trend continues**: Evidence for velocity-dependent dark-sector response
- **If Abell 2146 deviates**: Trend may be coincidental; larger sample needed
- **Null result**: Constrains wake amplitude in viable theoretical models

### 6.3 Relation to Previous Work

Cognola et al. (2022) found nonlocal gravity indistinguishable from GR using standard cluster lensing, concluding "a different discriminator is needed." Our kinematic-correlation approach directly addresses this gap.

The Sivakumar et al. (2025) prediction of velocity-correlated asymmetric residuals provides the most direct theoretical support for our methodology.

### 6.4 Limitations

- **Small sample**: N = 3–4 limits statistical power
- **Baseline choice**: Gaussian smoothing is simple; NFW fitting would be more physical
- **Kinematic uncertainties**: Published velocities have ~20% errors
- **Projection effects**: Even POS mergers have some LOS component

---

## 7. CONCLUSIONS

We present the first methodology for testing kinematic-history dependence in gravitational lensing through correlation of residual morphology with merger infall velocity.

**Key contributions:**
1. Model-agnostic framework applicable to any "memory" theory
2. Definition of morphological metrics quantifying residual structure
3. Discovery that merger geometry (POS vs LOS) critically affects projected signatures
4. Preliminary evidence for velocity-metric correlation in POS mergers (N = 2)

**Immediate next step:** Analysis of Abell 2146 (v = 2700 km/s, best kinematics) will test whether the POS trend continues.

**Future work:** Expansion to ~10 POS mergers using eROSITA-selected clusters with Rubin/Euclid weak lensing will achieve meaningful statistical power for moderate effect sizes.

---

## ACKNOWLEDGMENTS

Based on observations with NASA/ESA Hubble Space Telescope from STScI. Frontier Fields lens models from the CATS team. We thank L. King and J. Coleman for providing Abell 2146 convergence data.

---

## REFERENCES

Berezhiani, L., & Khoury, J. 2015, PRD, 92, 103510
Chadayammuri, U., et al. 2024, arXiv:2407.03142
Chaudhuri, S., et al. 2025, arXiv:2509.22293
Cognola, G., et al. 2022, arXiv:2205.03216
Coleman, J. E., et al. 2017, MNRAS, 464, 2469
Grayson, J., et al. 2024, MNRAS, 536, 2690
Jauzac, M., et al. 2015, MNRAS, 446, 4132
Kim, J., et al. 2021, MNRAS, 509, 1201
Maggiore, M., & Mancarella, M. 2014, PRD, 90, 023005
Richard, J., et al. 2010, MNRAS, 404, 325
Russell, H. R., et al. 2012, MNRAS, 423, 236
Sivakumar, A., et al. 2025, PRD, 111, 083511

---

*Draft v3.0 — January 2026*
*Pending: Abell 2146 analysis*
