#!/usr/bin/env python3
"""
Wake Signatures Analysis Pipeline
==================================
Comprehensive toolkit for testing kinematic-history-dependent 
gravitational signatures in merging galaxy clusters.

This module provides:
1. Data download utilities for MAST Frontier Fields
2. Convergence map processing
3. Morphological metric computation
4. Kinematic parameter handling
5. Correlation analysis with bootstrap significance testing
6. Visualization tools

Author: [Your Name]
Date: January 2026
"""

import numpy as np
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import os

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClusterData:
    """Container for cluster observational data."""
    name: str
    redshift: float
    kappa_obs: np.ndarray          # Observed convergence map
    kappa_baseline: np.ndarray     # Baseline model (NFW or simulation-matched)
    kappa_error: np.ndarray        # Uncertainty map
    pixel_scale: float             # arcsec/pixel
    merger_axis_angle: float       # degrees, measured from x-axis
    mask: Optional[np.ndarray] = None  # Boolean mask for valid pixels
    
    @property
    def delta_kappa(self) -> np.ndarray:
        """Residual convergence map."""
        return self.kappa_obs - self.kappa_baseline
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.kappa_obs.shape


@dataclass
class KinematicParams:
    """Merger kinematic parameters with uncertainties."""
    v_infall: float      # km/s - infall velocity
    v_infall_err: float  # km/s - uncertainty
    impact_param: float  # kpc - impact parameter
    impact_param_err: float
    t_collision: float   # Gyr - time since pericenter
    t_collision_err: float
    viewing_angle: float  # degrees - angle from plane of sky
    viewing_angle_err: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'v_infall': self.v_infall,
            'impact_param': self.impact_param,
            't_collision': self.t_collision,
            'viewing_angle': self.viewing_angle
        }


@dataclass
class MorphologyMetrics:
    """Container for all morphological metrics."""
    dipole_magnitude: float
    dipole_angle: float  # degrees
    quadrupole_strength: float
    quadrupole_angle: float  # degrees
    tail_alignment: float  # -1 to +1
    asymmetry: float  # 0 to 1
    centroid_offset: float  # arcsec
    power_total: float
    power_slope: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'dipole_mag': self.dipole_magnitude,
            'dipole_angle': self.dipole_angle,
            'quadrupole': self.quadrupole_strength,
            'quadrupole_angle': self.quadrupole_angle,
            'tail_alignment': self.tail_alignment,
            'asymmetry': self.asymmetry,
            'centroid_offset': self.centroid_offset,
            'power_total': self.power_total,
            'power_slope': self.power_slope
        }


# =============================================================================
# DATA ACCESS UTILITIES
# =============================================================================

def get_mast_frontier_fields_url(cluster: str, model: str = 'cats') -> str:
    """
    Generate URL for Frontier Fields lensing models on MAST.
    
    Parameters
    ----------
    cluster : str
        Cluster name: 'macs0416', 'abell2744', 'macs0717', etc.
    model : str
        Lens model team: 'cats', 'sharon', 'williams', 'grale', etc.
    
    Returns
    -------
    str
        URL to FITS file on MAST
    """
    base_url = "https://archive.stsci.edu/pub/hlsp/frontier/lensmodels"
    
    cluster_map = {
        'macs0416': 'macs0416',
        'macsj0416': 'macs0416',
        'abell2744': 'abell2744',
        'a2744': 'abell2744',
        'macs0717': 'macs0717',
        'macs1149': 'macs1149',
        'abells1063': 'abells1063',
        'abell370': 'abell370'
    }
    
    cluster_id = cluster_map.get(cluster.lower(), cluster.lower())
    
    return f"{base_url}/{cluster_id}/{model}/"


def list_available_models() -> Dict[str, List[str]]:
    """Return dictionary of available clusters and their lens model teams."""
    return {
        'macs0416': ['cats', 'sharon', 'williams', 'grale', 'keeton', 'glafic'],
        'abell2744': ['cats', 'sharon', 'williams', 'grale', 'keeton'],
        'macs0717': ['cats', 'sharon', 'williams', 'grale'],
        'macs1149': ['cats', 'sharon', 'williams', 'grale'],
        'abells1063': ['cats', 'sharon', 'williams'],
        'abell370': ['cats', 'sharon', 'williams']
    }


def download_convergence_map(cluster: str, model: str = 'cats', 
                            source_redshift: float = 9.0,
                            output_dir: str = './data') -> str:
    """
    Download convergence map from MAST Frontier Fields archive.
    
    Parameters
    ----------
    cluster : str
        Cluster name
    model : str
        Lens model team
    source_redshift : float
        Source redshift for κ map (common: 1.0, 2.0, 4.0, 9.0)
    output_dir : str
        Directory to save downloaded file
    
    Returns
    -------
    str
        Path to downloaded FITS file
        
    Notes
    -----
    Requires internet access. Run this on your local machine if 
    network is restricted in the analysis environment.
    """
    import requests
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct filename (MAST naming convention)
    z_str = f"z{source_redshift:.1f}".replace('.', '')
    filename = f"hlsp_frontier_{cluster}_{model}_kappa_{z_str}.fits"
    
    url = get_mast_frontier_fields_url(cluster, model) + filename
    output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to: {output_path}")
        return output_path
    else:
        raise RuntimeError(f"Download failed: HTTP {response.status_code}")


def load_fits_convergence(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load convergence map from FITS file.
    
    Returns
    -------
    data : np.ndarray
        2D convergence map
    header : dict
        FITS header as dictionary
    """
    from astropy.io import fits
    
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = dict(hdul[0].header)
    
    return data, header


# =============================================================================
# MORPHOLOGICAL METRICS
# =============================================================================

def compute_centroid(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute intensity-weighted centroid of image.
    
    Returns (x_centroid, y_centroid) in pixel coordinates.
    """
    ny, nx = image.shape
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    
    if mask is not None:
        image = np.where(mask, image, 0)
    
    total = np.sum(np.abs(image))
    if total == 0:
        return nx/2, ny/2
    
    x_c = np.sum(np.abs(image) * x_coords) / total
    y_c = np.sum(np.abs(image) * y_coords) / total
    
    return x_c, y_c


def compute_dipole_moment(delta_kappa: np.ndarray, 
                          pixel_scale: float = 1.0,
                          mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    Compute dipole moment of residual convergence map.
    
    d = ∫ Δκ(x) x d²x
    
    Parameters
    ----------
    delta_kappa : np.ndarray
        Residual convergence map
    pixel_scale : float
        Pixel scale in arcsec/pixel
    mask : np.ndarray, optional
        Boolean mask (True = valid)
    
    Returns
    -------
    d_magnitude : float
        |d| in arcsec
    d_x : float
        x-component
    d_y : float
        y-component
    """
    ny, nx = delta_kappa.shape
    
    # Coordinate grids centered on image center
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    x_coords = (x_coords - nx/2) * pixel_scale
    y_coords = (y_coords - ny/2) * pixel_scale
    
    if mask is not None:
        delta_kappa = np.where(mask, delta_kappa, 0)
    
    # Pixel area
    dA = pixel_scale**2
    
    # Dipole components
    d_x = np.sum(delta_kappa * x_coords) * dA
    d_y = np.sum(delta_kappa * y_coords) * dA
    
    d_magnitude = np.sqrt(d_x**2 + d_y**2)
    
    return d_magnitude, d_x, d_y


def compute_quadrupole(delta_kappa: np.ndarray,
                       pixel_scale: float = 1.0,
                       mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute quadrupole moment and anisotropy.
    
    Q_ij = ∫ Δκ(x) (x_i x_j - δ_ij |x|²/2) d²x
    
    Returns
    -------
    Q_strength : float
        Quadrupole anisotropy: (λ₁ - λ₂)/(λ₁ + λ₂), range [0, 1]
    Q_angle : float
        Position angle of major axis in degrees
    """
    ny, nx = delta_kappa.shape
    
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    x_coords = (x_coords - nx/2) * pixel_scale
    y_coords = (y_coords - ny/2) * pixel_scale
    
    if mask is not None:
        delta_kappa = np.where(mask, delta_kappa, 0)
    
    dA = pixel_scale**2
    
    # Second moments
    Q_xx = np.sum(delta_kappa * x_coords**2) * dA
    Q_yy = np.sum(delta_kappa * y_coords**2) * dA
    Q_xy = np.sum(delta_kappa * x_coords * y_coords) * dA
    
    # Traceless quadrupole tensor
    trace = Q_xx + Q_yy
    Q_xx_tl = Q_xx - trace/2
    Q_yy_tl = Q_yy - trace/2
    
    # Eigenvalues of 2x2 traceless tensor
    # For [[a, b], [b, -a]]: λ = ±√(a² + b²)
    a = (Q_xx - Q_yy) / 2
    b = Q_xy
    
    lambda_max = np.sqrt(a**2 + b**2)
    
    # Anisotropy (normalized)
    if trace != 0:
        Q_strength = 2 * lambda_max / np.abs(trace)
    else:
        Q_strength = 0
    
    Q_strength = min(Q_strength, 1.0)  # Clip to [0, 1]
    
    # Position angle
    Q_angle = 0.5 * np.degrees(np.arctan2(2*b, Q_xx - Q_yy))
    
    return Q_strength, Q_angle


def compute_tail_alignment(delta_kappa: np.ndarray,
                          merger_axis_angle: float,
                          pixel_scale: float = 1.0,
                          mask: Optional[np.ndarray] = None) -> float:
    """
    Compute tail-alignment index.
    
    T = cos(2(θ_residual - θ_merger))
    
    where θ_residual is the position angle of the quadrupole.
    
    Returns
    -------
    T : float
        Tail-alignment index, range [-1, +1]
        +1 = aligned with merger axis
        -1 = perpendicular to merger axis
    """
    _, Q_angle = compute_quadrupole(delta_kappa, pixel_scale, mask)
    
    # Angle difference (factor of 2 because quadrupole has 180° symmetry)
    angle_diff = np.radians(2 * (Q_angle - merger_axis_angle))
    
    T = np.cos(angle_diff)
    
    return T


def compute_asymmetry(delta_kappa: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """
    Compute asymmetry score (CAS-style).
    
    A = Σ|Δκ(x) - Δκ(-x)| / (2 Σ|Δκ(x)|)
    
    where Δκ(-x) is the 180° rotated image.
    
    Returns
    -------
    A : float
        Asymmetry score, range [0, 1]
        0 = perfect point symmetry
        1 = maximally asymmetric
    """
    # Rotate 180 degrees about center
    rotated = np.rot90(delta_kappa, 2)
    
    if mask is not None:
        mask_rotated = np.rot90(mask, 2)
        combined_mask = mask & mask_rotated
        delta_kappa = np.where(combined_mask, delta_kappa, 0)
        rotated = np.where(combined_mask, rotated, 0)
    
    numerator = np.sum(np.abs(delta_kappa - rotated))
    denominator = 2 * np.sum(np.abs(delta_kappa))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_centroid_offset(kappa_obs: np.ndarray,
                           kappa_baseline: np.ndarray,
                           pixel_scale: float = 1.0,
                           mask: Optional[np.ndarray] = None) -> float:
    """
    Compute centroid displacement between observed and baseline maps.
    
    |Δx_c| = |x_c,obs - x_c,baseline|
    
    Returns
    -------
    offset : float
        Centroid offset in arcsec
    """
    x_obs, y_obs = compute_centroid(kappa_obs, mask)
    x_base, y_base = compute_centroid(kappa_baseline, mask)
    
    dx = (x_obs - x_base) * pixel_scale
    dy = (y_obs - y_base) * pixel_scale
    
    return np.sqrt(dx**2 + dy**2)


def compute_power_spectrum(delta_kappa: np.ndarray,
                          pixel_scale: float = 1.0,
                          mask: Optional[np.ndarray] = None,
                          n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute 1D azimuthally-averaged power spectrum of residuals.
    
    P(k) = |Δκ̃(k)|²
    
    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centers (1/arcsec)
    P_k : np.ndarray
        Power spectrum values
    P_total : float
        Total integrated power
    slope : float
        Power-law slope from log-log fit
    """
    ny, nx = delta_kappa.shape
    
    if mask is not None:
        # Apodize masked regions
        delta_kappa = np.where(mask, delta_kappa, 0)
    
    # 2D FFT
    fft_map = fftshift(fft2(delta_kappa))
    power_2d = np.abs(fft_map)**2
    
    # Frequency coordinates
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_scale))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_scale))
    kx, ky = np.meshgrid(freq_x, freq_y)
    k_mag = np.sqrt(kx**2 + ky**2)
    
    # Azimuthal averaging
    k_max = np.min([freq_x.max(), freq_y.max()])
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
    
    P_k = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (k_mag >= k_edges[i]) & (k_mag < k_edges[i+1])
        if np.sum(in_bin) > 0:
            P_k[i] = np.mean(power_2d[in_bin])
    
    # Total power (exclude k=0)
    P_total = np.sum(P_k[1:] * np.diff(k_edges)[1:])
    
    # Fit power-law slope
    valid = (k_bins > 0) & (P_k > 0)
    if np.sum(valid) > 3:
        log_k = np.log10(k_bins[valid])
        log_P = np.log10(P_k[valid])
        slope, _ = np.polyfit(log_k, log_P, 1)
    else:
        slope = np.nan
    
    return k_bins, P_k, P_total, slope


def compute_all_metrics(cluster: ClusterData) -> MorphologyMetrics:
    """
    Compute all morphological metrics for a cluster.
    
    Parameters
    ----------
    cluster : ClusterData
        Cluster data container
    
    Returns
    -------
    MorphologyMetrics
        Container with all computed metrics
    """
    delta_kappa = cluster.delta_kappa
    pixel_scale = cluster.pixel_scale
    mask = cluster.mask
    
    # Dipole
    d_mag, d_x, d_y = compute_dipole_moment(delta_kappa, pixel_scale, mask)
    d_angle = np.degrees(np.arctan2(d_y, d_x))
    
    # Quadrupole
    Q_strength, Q_angle = compute_quadrupole(delta_kappa, pixel_scale, mask)
    
    # Tail alignment
    T = compute_tail_alignment(delta_kappa, cluster.merger_axis_angle, 
                               pixel_scale, mask)
    
    # Asymmetry
    A = compute_asymmetry(delta_kappa, mask)
    
    # Centroid offset
    offset = compute_centroid_offset(cluster.kappa_obs, cluster.kappa_baseline,
                                     pixel_scale, mask)
    
    # Power spectrum
    _, _, P_total, slope = compute_power_spectrum(delta_kappa, pixel_scale, 
                                                   mask)
    
    return MorphologyMetrics(
        dipole_magnitude=d_mag,
        dipole_angle=d_angle,
        quadrupole_strength=Q_strength,
        quadrupole_angle=Q_angle,
        tail_alignment=T,
        asymmetry=A,
        centroid_offset=offset,
        power_total=P_total,
        power_slope=slope
    )


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlations(metrics_list: List[MorphologyMetrics],
                        kinematics_list: List[KinematicParams],
                        method: str = 'spearman') -> Dict[str, Dict[str, float]]:
    """
    Compute correlations between all metrics and kinematic parameters.
    
    Parameters
    ----------
    metrics_list : list of MorphologyMetrics
        Metrics for each cluster
    kinematics_list : list of KinematicParams
        Kinematics for each cluster
    method : str
        'spearman' or 'pearson'
    
    Returns
    -------
    dict
        Nested dict: correlations[metric_name][kinematic_name] = (r, p)
    """
    metric_names = ['dipole_mag', 'quadrupole', 'tail_alignment', 
                    'asymmetry', 'centroid_offset', 'power_total']
    kinematic_names = ['v_infall', 'impact_param', 't_collision', 'viewing_angle']
    
    # Build arrays
    metric_arrays = {name: [] for name in metric_names}
    kinematic_arrays = {name: [] for name in kinematic_names}
    
    for m, k in zip(metrics_list, kinematics_list):
        m_dict = m.to_dict()
        k_dict = k.to_dict()
        
        for name in metric_names:
            metric_arrays[name].append(m_dict[name])
        for name in kinematic_names:
            kinematic_arrays[name].append(k_dict[name])
    
    # Convert to arrays
    for name in metric_names:
        metric_arrays[name] = np.array(metric_arrays[name])
    for name in kinematic_names:
        kinematic_arrays[name] = np.array(kinematic_arrays[name])
    
    # Compute correlations
    corr_func = stats.spearmanr if method == 'spearman' else stats.pearsonr
    
    correlations = {}
    for m_name in metric_names:
        correlations[m_name] = {}
        for k_name in kinematic_names:
            r, p = corr_func(metric_arrays[m_name], kinematic_arrays[k_name])
            correlations[m_name][k_name] = {'r': r, 'p': p}
    
    return correlations


def bootstrap_correlation_test(metric_values: np.ndarray,
                              kinematic_values: np.ndarray,
                              n_bootstrap: int = 10000,
                              method: str = 'spearman') -> Dict[str, float]:
    """
    Bootstrap significance test for correlation.
    
    Parameters
    ----------
    metric_values : np.ndarray
        Morphological metric values for each cluster
    kinematic_values : np.ndarray
        Kinematic parameter values for each cluster
    n_bootstrap : int
        Number of bootstrap realizations
    method : str
        'spearman' or 'pearson'
    
    Returns
    -------
    dict
        'r_observed': observed correlation
        'p_value': two-tailed p-value from bootstrap
        'ci_lower': 2.5th percentile of bootstrap distribution
        'ci_upper': 97.5th percentile
    """
    n = len(metric_values)
    corr_func = stats.spearmanr if method == 'spearman' else stats.pearsonr
    
    # Observed correlation
    r_obs, _ = corr_func(metric_values, kinematic_values)
    
    # Bootstrap under null (shuffle kinematic values)
    r_null = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        shuffled = np.random.permutation(kinematic_values)
        r_null[i], _ = corr_func(metric_values, shuffled)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(r_null) >= np.abs(r_obs))
    
    # Bootstrap confidence interval (resample pairs)
    r_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        r_boot[i], _ = corr_func(metric_values[idx], kinematic_values[idx])
    
    ci_lower = np.percentile(r_boot, 2.5)
    ci_upper = np.percentile(r_boot, 97.5)
    
    return {
        'r_observed': r_obs,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'null_distribution': r_null
    }


def benjamini_hochberg_correction(p_values: List[float], 
                                   q: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Parameters
    ----------
    p_values : list of float
        Uncorrected p-values
    q : float
        FDR threshold
    
    Returns
    -------
    list of bool
        True if significant after correction
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # BH thresholds
    thresholds = np.arange(1, n+1) / n * q
    
    # Find largest k where p_k <= threshold_k
    significant = sorted_p <= thresholds
    
    if not np.any(significant):
        return [False] * n
    
    # All tests with rank <= k are significant
    k = np.max(np.where(significant)[0])
    
    result = [False] * n
    for i in range(k + 1):
        result[sorted_indices[i]] = True
    
    return result


# =============================================================================
# POWER ANALYSIS
# =============================================================================

def power_analysis(effect_size: float, 
                   n_samples: int,
                   alpha: float = 0.05) -> float:
    """
    Compute statistical power for detecting correlation.
    
    Uses Fisher z-transformation approximation.
    
    Parameters
    ----------
    effect_size : float
        True population correlation |ρ|
    n_samples : int
        Sample size (number of clusters)
    alpha : float
        Significance level (two-tailed)
    
    Returns
    -------
    float
        Statistical power
    """
    if n_samples <= 3:
        return 0.0
    
    # Fisher z-transform
    z_r = np.arctanh(effect_size)
    se = 1 / np.sqrt(n_samples - 3)
    
    # Critical z for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Power
    power = 1 - stats.norm.cdf(z_crit - z_r/se) + stats.norm.cdf(-z_crit - z_r/se)
    
    return power


def required_sample_size(effect_size: float,
                        power: float = 0.8,
                        alpha: float = 0.05) -> int:
    """
    Compute required sample size to achieve target power.
    
    Parameters
    ----------
    effect_size : float
        Minimum detectable correlation |ρ|
    power : float
        Target power
    alpha : float
        Significance level
    
    Returns
    -------
    int
        Required sample size
    """
    for n in range(4, 500):
        if power_analysis(effect_size, n, alpha) >= power:
            return n
    return 500


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_residual_map(cluster: ClusterData, 
                      save_path: Optional[str] = None,
                      show_metrics: bool = True):
    """
    Plot residual convergence map with annotations.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Observed
    im0 = axes[0].imshow(cluster.kappa_obs, cmap='viridis', origin='lower')
    axes[0].set_title(f'{cluster.name}\nκ_obs', fontsize=12)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Baseline
    im1 = axes[1].imshow(cluster.kappa_baseline, cmap='viridis', origin='lower')
    axes[1].set_title('κ_baseline', fontsize=12)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Residual
    vmax = np.percentile(np.abs(cluster.delta_kappa), 99)
    im2 = axes[2].imshow(cluster.delta_kappa, cmap='RdBu_r', origin='lower',
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title('Δκ = κ_obs - κ_baseline', fontsize=12)
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label='Δκ')
    
    # Add merger axis
    ny, nx = cluster.kappa_obs.shape
    cx, cy = nx/2, ny/2
    length = min(nx, ny) * 0.3
    angle_rad = np.radians(cluster.merger_axis_angle)
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    axes[2].annotate('', xy=(cx+dx, cy+dy), xytext=(cx-dx, cy-dy),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    axes[2].text(cx+dx*1.1, cy+dy*1.1, 'Merger\naxis', fontsize=9, ha='center')
    
    if show_metrics:
        metrics = compute_all_metrics(cluster)
        text = f"|d| = {metrics.dipole_magnitude:.3f}\n"
        text += f"Q = {metrics.quadrupole_strength:.3f}\n"
        text += f"T = {metrics.tail_alignment:.3f}\n"
        text += f"A = {metrics.asymmetry:.3f}"
        axes[2].text(0.02, 0.98, text, transform=axes[2].transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(correlations: Dict, save_path: Optional[str] = None):
    """
    Plot correlation matrix heatmap.
    """
    import matplotlib.pyplot as plt
    
    metric_names = list(correlations.keys())
    kinematic_names = list(correlations[metric_names[0]].keys())
    
    # Build matrix
    matrix = np.zeros((len(metric_names), len(kinematic_names)))
    for i, m in enumerate(metric_names):
        for j, k in enumerate(kinematic_names):
            matrix[i, j] = correlations[m][k]['r']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(kinematic_names)))
    ax.set_yticks(range(len(metric_names)))
    ax.set_xticklabels(kinematic_names, rotation=45, ha='right')
    ax.set_yticklabels(metric_names)
    
    # Add correlation values
    for i in range(len(metric_names)):
        for j in range(len(kinematic_names)):
            r = matrix[i, j]
            p = correlations[metric_names[i]][kinematic_names[j]]['p']
            text = f'{r:.2f}'
            if p < 0.05:
                text += '*'
            if p < 0.01:
                text += '*'
            ax.text(j, i, text, ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, label='Correlation (r)')
    ax.set_title('Metric-Kinematic Correlation Matrix\n(* p<0.05, ** p<0.01)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# DEMONSTRATION WITH SYNTHETIC DATA
# =============================================================================

def generate_synthetic_cluster(name: str,
                               v_infall: float,
                               add_wake_signal: bool = False,
                               signal_strength: float = 0.1,
                               seed: int = None) -> Tuple[ClusterData, KinematicParams]:
    """
    Generate synthetic cluster data for testing pipeline.
    
    Parameters
    ----------
    name : str
        Cluster name
    v_infall : float
        Infall velocity in km/s
    add_wake_signal : bool
        If True, add velocity-dependent wake signature
    signal_strength : float
        Strength of wake signal (if added)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    cluster : ClusterData
    kinematics : KinematicParams
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Image parameters
    nx, ny = 200, 200
    pixel_scale = 0.5  # arcsec/pixel
    
    # Create coordinate grids
    y, x = np.mgrid[0:ny, 0:nx]
    x = (x - nx/2) * pixel_scale
    y = (y - ny/2) * pixel_scale
    
    # Generate two NFW-like peaks (merger)
    def nfw_profile(x, y, x0, y0, kappa_0, r_s):
        r = np.sqrt((x - x0)**2 + (y - y0)**2) + 0.1
        return kappa_0 * r_s / r * (1 - np.log(1 + r/r_s) / (r/r_s))
    
    # Main cluster
    kappa_main = nfw_profile(x, y, -10, 5, 0.8, 30)
    
    # Subcluster
    kappa_sub = nfw_profile(x, y, 15, -3, 0.4, 20)
    
    # True mass distribution
    kappa_true = kappa_main + kappa_sub
    
    # Baseline model (slight mismatch to create residuals)
    kappa_baseline = nfw_profile(x, y, -8, 4, 0.75, 32) + nfw_profile(x, y, 14, -2, 0.38, 22)
    
    # Add wake signal if requested (velocity-dependent)
    if add_wake_signal:
        # Wake aligned with merger axis, strength proportional to velocity
        merger_angle = np.arctan2(-3 - 5, 15 - (-10))  # From main to sub
        wake_x = x * np.cos(merger_angle) + y * np.sin(merger_angle)
        wake_amplitude = signal_strength * (v_infall / 2000)  # Normalized to 2000 km/s
        wake = wake_amplitude * np.exp(-wake_x**2 / 400) * np.sign(wake_x) * np.exp(-np.sqrt(x**2+y**2)/50)
        kappa_true = kappa_true + wake
    
    # Add noise
    noise_level = 0.02
    kappa_obs = kappa_true + np.random.normal(0, noise_level, (ny, nx))
    kappa_error = np.ones((ny, nx)) * noise_level
    
    # Merger axis angle
    merger_axis = np.degrees(np.arctan2(-3 - 5, 15 - (-10)))
    
    # Create cluster data
    cluster = ClusterData(
        name=name,
        redshift=0.4,
        kappa_obs=kappa_obs,
        kappa_baseline=kappa_baseline,
        kappa_error=kappa_error,
        pixel_scale=pixel_scale,
        merger_axis_angle=merger_axis,
        mask=None
    )
    
    # Kinematic parameters
    kinematics = KinematicParams(
        v_infall=v_infall,
        v_infall_err=v_infall * 0.15,
        impact_param=200 + np.random.normal(0, 50),
        impact_param_err=50,
        t_collision=0.3 + np.random.normal(0, 0.1),
        t_collision_err=0.1,
        viewing_angle=30 + np.random.normal(0, 10),
        viewing_angle_err=10
    )
    
    return cluster, kinematics


def run_demonstration():
    """
    Run complete demonstration of the analysis pipeline.
    """
    print("=" * 60)
    print("WAKE SIGNATURES ANALYSIS PIPELINE - DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic pilot sample
    print("\n1. Generating synthetic cluster sample...")
    
    cluster_configs = [
        ("MACSJ0416", 2200, 42),
        ("Abell2146", 1800, 43),
        ("JKCS041", 2800, 44),
        ("Abell2744", 2500, 45),
        ("RXJ2129", 1500, 46),
    ]
    
    clusters = []
    kinematics = []
    
    # Scenario: Add wake signal to test detection
    add_signal = True
    
    for name, v, seed in cluster_configs:
        c, k = generate_synthetic_cluster(name, v, add_wake_signal=add_signal, 
                                          signal_strength=0.15, seed=seed)
        clusters.append(c)
        kinematics.append(k)
        print(f"   {name}: v_infall = {v} km/s")
    
    # Compute metrics
    print("\n2. Computing morphological metrics...")
    metrics_list = []
    for c in clusters:
        m = compute_all_metrics(c)
        metrics_list.append(m)
        print(f"   {c.name}: |d|={m.dipole_magnitude:.4f}, Q={m.quadrupole_strength:.3f}, "
              f"T={m.tail_alignment:.3f}, A={m.asymmetry:.3f}")
    
    # Compute correlations
    print("\n3. Computing correlations...")
    correlations = compute_correlations(metrics_list, kinematics, method='spearman')
    
    print("\n   Spearman correlations with v_infall:")
    for metric in ['dipole_mag', 'quadrupole', 'tail_alignment', 'asymmetry']:
        r = correlations[metric]['v_infall']['r']
        p = correlations[metric]['v_infall']['p']
        print(f"   {metric:20s}: r = {r:+.3f}, p = {p:.4f}")
    
    # Bootstrap test for primary metric
    print("\n4. Bootstrap significance testing...")
    
    dipole_values = np.array([m.dipole_magnitude for m in metrics_list])
    v_values = np.array([k.v_infall for k in kinematics])
    
    boot_result = bootstrap_correlation_test(dipole_values, v_values, n_bootstrap=5000)
    
    print(f"   Dipole vs v_infall:")
    print(f"   Observed r = {boot_result['r_observed']:.3f}")
    print(f"   Bootstrap p-value = {boot_result['p_value']:.4f}")
    print(f"   95% CI: [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")
    
    # Power analysis
    print("\n5. Power analysis...")
    for n in [5, 10, 15, 25, 30]:
        p70 = power_analysis(0.7, n)
        p50 = power_analysis(0.5, n)
        print(f"   N={n:2d}: power(r=0.7) = {p70:.2f}, power(r=0.5) = {p50:.2f}")
    
    # FDR correction
    print("\n6. Multiple comparison correction (Benjamini-Hochberg)...")
    p_values = [correlations[m]['v_infall']['p'] for m in 
                ['dipole_mag', 'quadrupole', 'tail_alignment', 'asymmetry', 
                 'centroid_offset', 'power_total']]
    significant = benjamini_hochberg_correction(p_values, q=0.05)
    
    metric_names = ['dipole_mag', 'quadrupole', 'tail_alignment', 'asymmetry',
                    'centroid_offset', 'power_total']
    for name, p, sig in zip(metric_names, p_values, significant):
        status = "SIGNIFICANT" if sig else "not significant"
        print(f"   {name:20s}: p = {p:.4f} -> {status}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    return clusters, kinematics, metrics_list, correlations


if __name__ == "__main__":
    clusters, kinematics, metrics, correlations = run_demonstration()
