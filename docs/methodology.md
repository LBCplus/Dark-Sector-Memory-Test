# Dark Sector Memory Test: Detailed Methodology

## 1. Lensing Reconstruction Pipeline

### 1.1 Weak Lensing Shear Measurement

The observed ellipticity of background galaxies provides a noisy estimate of the reduced shear g = γ/(1-κ). In the weak-lensing regime (κ << 1), g ≈ γ.

**Data Sources:**
- MACSJ0416: CLASH + HFF + BUFFALO (F435W-F160W, depth ~28.5-29 mag)
- Abell 2146: GO-12871 (F606W, F814W, depth ~27 mag)
- Abell 2744: CLASH + HFF (F435W-F160W, depth ~28.5-29 mag)

**Shape Measurement Pipeline:**
1. Source detection: SExtractor on stacked images; S/N > 10
2. PSF modeling: TinyTim or PSFEx on stellar sources
3. Shape measurement: KSB+ or lensfit; correct for PSF anisotropy
4. Photo-z estimation: BPZ or EAZY; select z_phot > z_cluster + 0.1

### 1.2 Strong Lensing Constraints

Multiple-image systems provide high-S/N constraints on inner mass distribution.

**Key Sources:**
- MACSJ0416: Grayson et al. (2024) - 237 spectroscopic images
- Abell 2744: Mahler et al. (2018) - 181 images
- RX J2129: Caminha et al. (2019) - MUSE spectroscopy

### 1.3 Convergence Reconstruction

**Method A: Kaiser-Squires Inversion**
Direct inversion in Fourier space: κ̃(k) = (k₁² - k₂² + 2ik₁k₂) / |k|² × γ̃(k)

**Method B: Parametric Modeling (LENSTOOL)**
Fit dPIE or NFW profiles to shear + strong-lensing constraints via MCMC.

**Method C: Free-Form (GRALE)**
Genetic algorithm optimization on adaptive grid.

---

## 2. Baseline Construction

### 2.1 Parametric Baseline (NFW + Multipole)

Model form:
κ_baseline(r,θ) = Σᵢ κ_NFW,i(r; Mᵢ, cᵢ, xᵢ) + Σₗₘ aₗₘ Yₗₘ(θ)

Procedure:
1. Identify mass peaks from κ_obs (local maxima above 3σ)
2. Initialize NFW parameters from ΛCDM mass-concentration relation
3. Fit via Levenberg-Marquardt
4. Compute residual: Δκ = κ_obs - κ_baseline

### 2.2 Simulation-Matched Baseline

Sources:
- Galaxy Cluster Merger Catalog (ZuHone et al. 2018)
- TNG-Cluster (2024-2025)
- BAHAMAS-MACSIS

Matching procedure:
1. Characterize cluster: M_200, mass ratio μ, morphology
2. Query simulation library for analogs
3. Project to observed viewing geometry
4. Extract simulated κ as baseline

---

## 3. Morphological Metrics

### 3.1 Dipole Moment
**d** = ∫ Δκ(x) **x** d²x

Measures preferred direction of excess mass.

### 3.2 Quadrupole Anisotropy
Q_ij = ∫ Δκ(x) (xᵢxⱼ - δᵢⱼ|x|²/2) d²x

Q = (λ₁ - λ₂)/(λ₁ + λ₂) where λ are eigenvalues.

### 3.3 Tail-Alignment Index
T = cos(2(θ_residual - θ_merger))

Range [-1, +1]: +1 = aligned, -1 = perpendicular.

### 3.4 Asymmetry Score
A = Σ|Δκ(x) - Δκ(-x)| / (2 Σ|Δκ(x)|)

Range [0, 1]: 0 = symmetric.

### 3.5 Centroid Displacement
|Δx_c| = |x_c,obs - x_c,baseline|

### 3.6 Power Spectrum
P(k) = |Δκ̃(k)|²; P_total = ∫ P(k) dk

---

## 4. Kinematic Inference

### 4.1 X-ray Shock Analysis
From Rankine-Hugoniot: T₂/T₁ = (5M² - 1)(M² + 3)/(16M²)
v_shock = M × c_s where c_s = √(γkT/μm_p)

### 4.2 Spectroscopic Velocities
Δv = c(z₁ - z₂)/(1 + z_cluster) between subclusters.

### 4.3 Bayesian Forward Modeling
Parameters: M_total, μ, v_infall, b, t_coll, α
Method: MCMC sampling comparing GADGET simulations to observations.
