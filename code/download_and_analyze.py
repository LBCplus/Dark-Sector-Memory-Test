#!/usr/bin/env python3
"""
Dark Sector Memory Test - Data Ingestion Pipeline
==================================================

This script downloads and processes real convergence maps from MAST
and prepares them for wake signature analysis.

Usage:
    python download_and_analyze.py --download          # Download FITS files
    python download_and_analyze.py --analyze           # Run analysis
    python download_and_analyze.py --download --analyze  # Both

Requirements:
    pip install astropy numpy scipy matplotlib requests
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import numpy as np

# Check for astropy
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not installed. Install with: pip install astropy")


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class ClusterConfig:
    """Configuration for a single cluster."""
    name: str
    z_cluster: float
    
    # Published kinematic constraints
    v_infall: float  # km/s (central value)
    v_infall_err_plus: float  # km/s
    v_infall_err_minus: float  # km/s
    mach: Optional[float] = None
    mach_err: Optional[float] = None
    t_collision: Optional[float] = None  # Gyr
    t_collision_range: Optional[Tuple[float, float]] = None
    
    # Data sources
    kinematic_source: str = ""
    lensing_source: str = ""
    
    # MAST URLs (if available)
    mast_cats_url: Optional[str] = None
    mast_grale_url: Optional[str] = None
    
    # Merger geometry
    merger_axis_angle: Optional[float] = None  # degrees, E of N
    impact_parameter: Optional[str] = None  # qualitative


# Define our pilot sample with published constraints
PILOT_SAMPLE = {
    "abell2146": ClusterConfig(
        name="Abell 2146",
        z_cluster=0.232,
        v_infall=2700.0,
        v_infall_err_plus=400.0,
        v_infall_err_minus=300.0,
        mach=2.3,
        mach_err=0.2,
        t_collision=0.15,
        t_collision_range=(0.1, 0.2),
        kinematic_source="Russell et al. 2012, MNRAS 423, 236",
        lensing_source="Coleman et al. 2017, MNRAS 464, 2469",
        mast_cats_url=None,  # Not in HFF - need author contact
        mast_grale_url=None,
        merger_axis_angle=None,  # Roughly E-W in plane of sky
        impact_parameter="small (near head-on)",
    ),
    
    "abell2744": ClusterConfig(
        name="Abell 2744",
        z_cluster=0.308,
        v_infall=2000.0,  # Central value of 1800-2200 range
        v_infall_err_plus=200.0,
        v_infall_err_minus=200.0,
        mach=1.2,
        mach_err=0.1,
        t_collision=0.55,
        t_collision_range=(0.5, 0.6),
        kinematic_source="Chadayammuri et al. 2024, arXiv:2407.03142",
        lensing_source="MAST Frontier Fields + JWST UNCOVER",
        mast_cats_url="https://archive.stsci.edu/prepds/frontier/lensmodels/abell2744/CATS/",
        mast_grale_url="https://archive.stsci.edu/prepds/frontier/lensmodels/abell2744/GRALE/",
        merger_axis_angle=None,  # N-S major merger
        impact_parameter="significant",
    ),
    
    "macs0416": ClusterConfig(
        name="MACSJ0416.1-2403",
        z_cluster=0.396,
        v_infall=1600.0,  # Central value of 1200-2000 range
        v_infall_err_plus=400.0,
        v_infall_err_minus=400.0,
        mach=None,  # Not measured
        mach_err=None,
        t_collision=None,  # Post-pericenter, not precisely constrained
        t_collision_range=None,
        kinematic_source="Jauzac et al. 2015, MNRAS 446, 4132",
        lensing_source="Grayson et al. 2024, MNRAS 536, 2690",
        mast_cats_url="https://archive.stsci.edu/prepds/frontier/lensmodels/macs0416/CATS/",
        mast_grale_url="https://archive.stsci.edu/prepds/frontier/lensmodels/macs0416/GRALE/",
        merger_axis_angle=None,  # Bimodal, roughly NE-SW
        impact_parameter="non-zero (off-axis)",
    ),
}

# Phase 2 sample
PHASE2_SAMPLE = {
    "jkcs041": ClusterConfig(
        name="JKCS041",
        z_cluster=1.95,
        v_infall=None,  # Simulation-derived, not directly measured
        v_infall_err_plus=None,
        v_infall_err_minus=None,
        mach=None,
        mach_err=None,
        t_collision=0.3,
        t_collision_range=(0.2, 0.4),
        kinematic_source="Finner et al. 2024, MNRAS 534, 3676",
        lensing_source="Weak lensing + simulation synthetic maps",
        mast_cats_url=None,
        mast_grale_url=None,
    ),
}


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_file(url: str, output_path: Path, overwrite: bool = False) -> bool:
    """Download a file from URL to output_path."""
    if output_path.exists() and not overwrite:
        print(f"  File exists: {output_path}")
        return True
    
    print(f"  Downloading: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  Saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_frontier_fields_kappa(cluster_key: str, 
                                    output_dir: Path,
                                    source_redshift: int = 2,
                                    models: List[str] = ['CATS', 'GRALE']) -> Dict[str, Path]:
    """
    Download convergence maps from MAST Frontier Fields.
    
    Parameters
    ----------
    cluster_key : str
        Cluster identifier (e.g., 'macs0416', 'abell2744')
    output_dir : Path
        Directory to save files
    source_redshift : int
        Source plane redshift (1, 2, 4, or 9)
    models : list
        Which lens model teams to download ('CATS', 'GRALE', etc.)
    
    Returns
    -------
    dict : {model_name: filepath} for successfully downloaded files
    """
    config = PILOT_SAMPLE.get(cluster_key)
    if config is None:
        print(f"Unknown cluster: {cluster_key}")
        return {}
    
    # Map cluster key to MAST naming convention
    mast_names = {
        'macs0416': 'macs0416',
        'abell2744': 'abell2744',
        'abell2146': None,  # Not in HFF
    }
    
    mast_name = mast_names.get(cluster_key)
    if mast_name is None:
        print(f"  {config.name} is not in Frontier Fields - need alternative source")
        return {}
    
    downloaded = {}
    base_url = "https://archive.stsci.edu/prepds/frontier/lensmodels"
    
    for model in models:
        filename = f"{mast_name}_{model}_kappa_z{source_redshift}.fits"
        url = f"{base_url}/{mast_name}/{model}/{filename}"
        output_path = output_dir / cluster_key / f"{model.lower()}_kappa_z{source_redshift}.fits"
        
        if download_file(url, output_path):
            downloaded[model] = output_path
    
    return downloaded


def download_all_pilot_data(output_dir: Path, source_redshift: int = 2) -> Dict[str, Dict]:
    """Download all available pilot sample data."""
    print("=" * 60)
    print("DOWNLOADING PILOT SAMPLE CONVERGENCE MAPS")
    print("=" * 60)
    
    results = {}
    
    for cluster_key, config in PILOT_SAMPLE.items():
        print(f"\n{config.name}:")
        print(f"  Kinematic source: {config.kinematic_source}")
        print(f"  v_infall = {config.v_infall} (+{config.v_infall_err_plus}/-{config.v_infall_err_minus}) km/s")
        
        if cluster_key == 'abell2146':
            print(f"  NOTE: Not in Frontier Fields. Need to contact Coleman et al. or")
            print(f"        reconstruct from LENSTOOL parameters in their Table 2.")
            results[cluster_key] = {'status': 'manual_required'}
        else:
            downloaded = download_frontier_fields_kappa(cluster_key, output_dir, source_redshift)
            results[cluster_key] = {'files': downloaded, 'status': 'downloaded' if downloaded else 'failed'}
    
    return results


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ConvergenceMap:
    """Container for a loaded convergence map with metadata."""
    cluster_key: str
    model: str
    kappa: np.ndarray
    header: dict
    pixel_scale_arcsec: float
    wcs: Optional[object] = None
    source_redshift: float = 2.0
    
    @property
    def shape(self):
        return self.kappa.shape
    
    def get_physical_scale(self, z_cluster: float) -> float:
        """Return pixel scale in kpc at cluster redshift (approximate)."""
        # Simplified - would need proper cosmology
        # At z~0.3, 1 arcsec ~ 4.5 kpc
        kpc_per_arcsec = 4.5 * (1 + z_cluster) / 1.3  # Rough scaling
        return self.pixel_scale_arcsec * kpc_per_arcsec


def load_convergence_fits(filepath: Path, cluster_key: str, model: str) -> Optional[ConvergenceMap]:
    """Load a FITS convergence map."""
    if not ASTROPY_AVAILABLE:
        print("ERROR: astropy required to load FITS files")
        return None
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    print(f"Loading: {filepath}")
    
    with fits.open(filepath) as hdul:
        # Primary extension or first image extension
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data
            header = hdul[1].header
        else:
            data = hdul[0].data
            header = hdul[0].header
        
        # Get pixel scale from header
        # Try various keywords
        pixel_scale = None
        for key in ['CDELT1', 'CD1_1', 'PIXSCALE']:
            if key in header:
                pixel_scale = abs(header[key])
                if key in ['CDELT1', 'CD1_1']:
                    pixel_scale *= 3600  # deg to arcsec
                break
        
        if pixel_scale is None:
            print("  Warning: Could not determine pixel scale, assuming 0.065 arcsec")
            pixel_scale = 0.065  # Typical HFF pixel scale
        
        # Try to get WCS
        try:
            wcs = WCS(header)
        except:
            wcs = None
        
        return ConvergenceMap(
            cluster_key=cluster_key,
            model=model,
            kappa=data.astype(np.float64),
            header=dict(header),
            pixel_scale_arcsec=pixel_scale,
            wcs=wcs,
        )


def load_all_downloaded_maps(data_dir: Path) -> Dict[str, Dict[str, ConvergenceMap]]:
    """Load all downloaded convergence maps."""
    print("\n" + "=" * 60)
    print("LOADING CONVERGENCE MAPS")
    print("=" * 60)
    
    maps = {}
    
    for cluster_dir in data_dir.iterdir():
        if not cluster_dir.is_dir():
            continue
        
        cluster_key = cluster_dir.name
        maps[cluster_key] = {}
        
        for fits_file in cluster_dir.glob("*.fits"):
            # Parse model name from filename
            model = fits_file.stem.split('_')[0].upper()
            
            kappa_map = load_convergence_fits(fits_file, cluster_key, model)
            if kappa_map is not None:
                maps[cluster_key][model] = kappa_map
                print(f"  Loaded {cluster_key}/{model}: shape={kappa_map.shape}, "
                      f"pixel_scale={kappa_map.pixel_scale_arcsec:.4f} arcsec")
    
    return maps


# =============================================================================
# ANALYSIS PREPARATION
# =============================================================================

def prepare_cluster_data_for_analysis(kappa_map: ConvergenceMap, 
                                       config: ClusterConfig) -> dict:
    """
    Prepare a cluster for wake signature analysis.
    
    Returns a dict compatible with the dsmt_analysis.py ClusterData format.
    """
    # For now, use observed map as both observed and baseline
    # In real analysis, baseline would be constructed separately
    
    # Create mask for valid pixels (non-zero, finite)
    mask = np.isfinite(kappa_map.kappa) & (kappa_map.kappa != 0)
    
    # Estimate noise level from edge regions
    edge_width = 20
    edge_pixels = np.concatenate([
        kappa_map.kappa[:edge_width, :].flatten(),
        kappa_map.kappa[-edge_width:, :].flatten(),
        kappa_map.kappa[:, :edge_width].flatten(),
        kappa_map.kappa[:, -edge_width:].flatten(),
    ])
    edge_pixels = edge_pixels[np.isfinite(edge_pixels)]
    noise_estimate = np.std(edge_pixels) if len(edge_pixels) > 100 else 0.01
    
    return {
        'name': config.name,
        'cluster_key': kappa_map.cluster_key,
        'model': kappa_map.model,
        'z': config.z_cluster,
        'kappa_obs': kappa_map.kappa,
        'kappa_baseline': None,  # To be constructed
        'kappa_error': np.full_like(kappa_map.kappa, noise_estimate),
        'pixel_scale': kappa_map.pixel_scale_arcsec,
        'mask': mask,
        'merger_axis_angle': config.merger_axis_angle,
        # Kinematic parameters
        'v_infall': config.v_infall,
        'v_infall_err': (config.v_infall_err_plus + config.v_infall_err_minus) / 2,
        'mach': config.mach,
        't_collision': config.t_collision,
    }


def print_sample_summary():
    """Print a summary of the pilot sample."""
    print("\n" + "=" * 60)
    print("DARK SECTOR MEMORY TEST - PILOT SAMPLE SUMMARY")
    print("=" * 60)
    
    print("\nCluster           z      v_infall (km/s)      Mach    t_coll (Gyr)")
    print("-" * 70)
    
    for key, config in PILOT_SAMPLE.items():
        v_str = f"{config.v_infall:.0f} (+{config.v_infall_err_plus:.0f}/-{config.v_infall_err_minus:.0f})"
        mach_str = f"{config.mach:.1f}±{config.mach_err:.1f}" if config.mach else "—"
        t_str = f"{config.t_collision:.2f}" if config.t_collision else "—"
        
        print(f"{config.name:17s} {config.z_cluster:.3f}  {v_str:22s}  {mach_str:8s}  {t_str}")
    
    print("\nKinematic Sources:")
    for key, config in PILOT_SAMPLE.items():
        print(f"  {config.name}: {config.kinematic_source}")
    
    print("\nLensing Data Status:")
    print("  MACSJ0416:   MAST Frontier Fields (CATS, GRALE) - PUBLIC")
    print("  Abell 2744:  MAST Frontier Fields (CATS, GRALE) - PUBLIC")
    print("  Abell 2146:  Coleman et al. 2017 - CONTACT AUTHORS")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dark Sector Memory Test - Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_and_analyze.py --download
    python download_and_analyze.py --summary
    python download_and_analyze.py --analyze --data-dir ./dsmt_data
        """
    )
    
    parser.add_argument('--download', action='store_true',
                        help='Download convergence maps from MAST')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis on downloaded data')
    parser.add_argument('--summary', action='store_true',
                        help='Print sample summary')
    parser.add_argument('--data-dir', type=Path, default=Path('./dsmt_data'),
                        help='Data directory (default: ./dsmt_data)')
    parser.add_argument('--source-z', type=int, default=2, choices=[1, 2, 4, 9],
                        help='Source redshift for kappa maps (default: 2)')
    
    args = parser.parse_args()
    
    if args.summary or (not args.download and not args.analyze):
        print_sample_summary()
    
    if args.download:
        results = download_all_pilot_data(args.data_dir, args.source_z)
        
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        for cluster, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'downloaded':
                files = result.get('files', {})
                print(f"  {cluster}: {len(files)} files downloaded")
            else:
                print(f"  {cluster}: {status}")
    
    if args.analyze:
        # Load downloaded maps
        maps = load_all_downloaded_maps(args.data_dir)
        
        if not maps:
            print("\nNo data found. Run with --download first.")
            return
        
        # Prepare for analysis
        print("\n" + "=" * 60)
        print("PREPARING DATA FOR ANALYSIS")
        print("=" * 60)
        
        analysis_data = []
        for cluster_key, model_maps in maps.items():
            config = PILOT_SAMPLE.get(cluster_key)
            if config is None:
                continue
            
            for model, kappa_map in model_maps.items():
                data = prepare_cluster_data_for_analysis(kappa_map, config)
                analysis_data.append(data)
                print(f"  Prepared: {data['name']} ({model})")
        
        print(f"\n  Total: {len(analysis_data)} convergence maps ready for analysis")
        print("\n  Next step: Import dsmt_analysis.py and run wake signature analysis")
        print("  See notebooks/full_analysis.ipynb for example workflow")


if __name__ == "__main__":
    main()
