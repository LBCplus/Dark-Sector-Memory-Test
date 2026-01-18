# Dark Sector Memory Test: Statistical Analysis Plan

## 1. Hypotheses

**H₀ (Null):** Population correlation ρ(Mᵢ, v_infall) = 0 for all morphological metrics Mᵢ.

**H₁ (Alternative):** ρ(Mᵢ, v_infall) ≠ 0 for at least one metric, exceeding ΛCDM expectations.

## 2. Primary Analysis

### 2.1 Correlation Testing
- Compute Spearman rank correlations for all metric-kinematic pairs
- Primary focus: correlations with v_infall (6 tests)
- Report with 95% confidence intervals via Fisher z-transformation

### 2.2 Multiple Comparison Correction
Benjamini-Hochberg FDR at q = 0.05:
1. Rank p-values from smallest to largest
2. Threshold: αᵢ = (i/6) × 0.05
3. Reject all H₀ with p ≤ threshold

### 2.3 Bootstrap Significance
1. Generate 10,000 ΛCDM mock catalogs
2. Compute correlations for each mock
3. p-value = fraction with |r| ≥ |r_observed|

## 3. Power Analysis

| N | Power (r=0.9) | Power (r=0.7) | Power (r=0.5) |
|---|---------------|---------------|---------------|
| 5 | 68% | 23% | 12% |
| 15 | 98% | 85% | 48% |
| 30 | >99% | 99% | 81% |

## 4. Decision Rules

| Outcome | Criterion | Action |
|---------|-----------|--------|
| Null | All q > 0.05; \|r\| < 0.5 | Report upper limits |
| Suggestive | 0.01 < q < 0.05 | Propose expanded study |
| Detection | q < 0.01; \|r\| > 0.7 | Model discrimination |

## 5. Model Discrimination (if signal)

Fit functional forms:
- Superfluid DM: M = a + b × sigmoid((v - v_crit)/σ)
- Non-Markovian EFT: M = a × v^β
- Nonlocal Gravity: M = a × ln(1 + v/v₀)

Compare via AIC/BIC.
