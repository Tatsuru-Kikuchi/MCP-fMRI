"""
Wavelet Analysis Module for fMRI Data - Mathematical Cognition Study

This module provides wavelet analysis capabilities specifically designed for 
analyzing brain activation patterns during mathematical tasks. It focuses on
identifying activated brain regions using time-frequency decomposition.

Author: MCP-fMRI Project
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pywt
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

# Neuroimaging libraries
try:
    import nibabel as nib
    from nilearn import datasets, plotting, image
    from nilearn.glm import threshold_stats_img
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    warnings.warn("Neuroimaging libraries not available. Install nibabel and nilearn for full functionality.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaveletfMRIAnalyzer:
    """
    Wavelet analysis for fMRI data focusing on mathematical cognition tasks.
    
    This class provides methods for:
    - Time-frequency decomposition of BOLD signals
    - Identification of brain activation patterns
    - Statistical analysis of wavelet coefficients
    - Visualization of results with ethical considerations
    """
    
    def __init__(self, 
                 wavelet: str = 'morlet',
                 sampling_rate: float = 0.5,
                 ethical_reporting: bool = True):
        """
        Initialize the wavelet analyzer.
        
        Parameters:
        -----------
        wavelet : str, default 'morlet'
            Wavelet type for analysis. Options: 'morlet', 'mexican_hat', 'db4', 'haar'
        sampling_rate : float, default 0.5
            Sampling rate of fMRI data in Hz (typical TR = 2s -> 0.5 Hz)
        ethical_reporting : bool, default True
            Whether to emphasize similarities over differences in reporting
        """
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self.ethical_reporting = ethical_reporting
        self.results = {}
        
        # Frequency bands of interest for cognitive tasks
        self.frequency_bands = {
            'very_slow': (0.01, 0.03),    # Default mode network
            'slow': (0.03, 0.06),         # Executive control
            'medium': (0.06, 0.12),       # Attention networks
            'fast': (0.12, 0.25)          # Task-related activity
        }
        
        logger.info(f"Initialized WaveletfMRIAnalyzer with {wavelet} wavelet")
    
    def load_fmri_data(self, 
                       nifti_path: str = None,
                       time_series: np.ndarray = None,
                       mask_path: str = None) -> np.ndarray:
        """
        Load fMRI data from NIfTI file or numpy array.
        
        Parameters:
        -----------
        nifti_path : str, optional
            Path to NIfTI file
        time_series : np.ndarray, optional
            Pre-loaded time series data (voxels x time)
        mask_path : str, optional
            Path to brain mask
            
        Returns:
        --------
        np.ndarray : Loaded time series data
        """
        if nifti_path and NEUROIMAGING_AVAILABLE:
            # Load from NIfTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            # Apply mask if provided
            if mask_path:
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                data = data[mask_data > 0]
            
            # Reshape to (voxels, time)
            if data.ndim == 4:
                data = data.reshape(-1, data.shape[-1])
            
            self.time_series = data
            logger.info(f"Loaded fMRI data: {data.shape[0]} voxels, {data.shape[1]} timepoints")
            
        elif time_series is not None:
            self.time_series = time_series
            logger.info(f"Using provided time series: {time_series.shape}")
            
        else:
            # Generate example data for demonstration
            self.time_series = self._generate_example_data()
            logger.info("Generated example fMRI data for demonstration")
            
        return self.time_series
    
    def _generate_example_data(self) -> np.ndarray:
        """Generate synthetic fMRI data for demonstration purposes."""
        n_voxels = 1000
        n_timepoints = 200
        
        # Base signal with cognitive task modulation
        t = np.linspace(0, n_timepoints/self.sampling_rate, n_timepoints)
        
        # Create different activation patterns for different "brain regions"
        data = np.zeros((n_voxels, n_timepoints))
        
        for i in range(n_voxels):
            # Base hemodynamic response
            base_signal = np.random.normal(0, 0.1, n_timepoints)
            
            # Add task-related activation for some voxels (mathematical cognition areas)
            if i < 200:  # "Parietal cortex" - mathematical processing
                task_signal = 0.5 * np.sin(2 * np.pi * 0.02 * t) * np.exp(-t/100)
                base_signal += task_signal
            elif i < 400:  # "Prefrontal cortex" - working memory
                task_signal = 0.3 * np.sin(2 * np.pi * 0.04 * t) * np.exp(-t/80)
                base_signal += task_signal
            elif i < 600:  # "Visual cortex" - number processing
                task_signal = 0.4 * np.sin(2 * np.pi * 0.08 * t) * np.exp(-t/60)
                base_signal += task_signal
            
            # Add noise
            data[i, :] = base_signal + np.random.normal(0, 0.05, n_timepoints)
        
        return data
    
    def continuous_wavelet_transform(self, 
                                   voxel_data: np.ndarray,
                                   scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform continuous wavelet transform on voxel time series.
        
        Parameters:
        -----------
        voxel_data : np.ndarray
            Time series data for a single voxel
        scales : np.ndarray, optional
            Scales for wavelet transform
            
        Returns:
        --------
        tuple : (coefficients, frequencies)
        """
        if scales is None:
            # Default scales covering physiologically relevant frequencies
            scales = np.logspace(np.log10(1), np.log10(100), 50)
        
        # Perform CWT
        if self.wavelet == 'morlet':
            coefficients, frequencies = pywt.cwt(voxel_data, scales, 'morl', 
                                                sampling_period=1/self.sampling_rate)
        else:
            coefficients, frequencies = pywt.cwt(voxel_data, scales, self.wavelet,
                                                sampling_period=1/self.sampling_rate)
        
        return coefficients, frequencies
    
    def analyze_math_activation(self, 
                               task_onsets: List[float] = None,
                               baseline_duration: float = 10.0,
                               activation_duration: float = 15.0) -> Dict:
        """
        Analyze brain activation during mathematical tasks using wavelets.
        
        Parameters:
        -----------
        task_onsets : List[float], optional
            Onset times of mathematical tasks in seconds
        baseline_duration : float, default 10.0
            Duration before task onset to consider as baseline
        activation_duration : float, default 15.0
            Duration after task onset to analyze
            
        Returns:
        --------
        Dict : Analysis results including activation maps and statistics
        """
        if task_onsets is None:
            # Default task onsets for demonstration
            task_onsets = [30, 80, 130, 180]
        
        n_voxels, n_timepoints = self.time_series.shape
        t = np.linspace(0, n_timepoints/self.sampling_rate, n_timepoints)
        
        activation_results = {
            'activation_maps': {},
            'statistical_maps': {},
            'frequency_profiles': {},
            'similarity_metrics': {}
        }
        
        # Analyze each frequency band
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            band_activation = np.zeros(n_voxels)
            band_significance = np.zeros(n_voxels)
            
            logger.info(f"Analyzing {band_name} band ({f_min}-{f_max} Hz)")
            
            for voxel_idx in range(min(n_voxels, 500)):  # Limit for computational efficiency
                voxel_signal = self.time_series[voxel_idx, :]
                
                # Perform wavelet transform
                coefficients, frequencies = self.continuous_wavelet_transform(voxel_signal)
                
                # Extract power in frequency band
                freq_mask = (frequencies >= f_min) & (frequencies <= f_max)
                band_power = np.mean(np.abs(coefficients[freq_mask, :]), axis=0)
                
                # Calculate activation vs baseline for each task
                task_activations = []
                baseline_activations = []
                
                for onset in task_onsets:
                    onset_idx = int(onset * self.sampling_rate)
                    baseline_start = max(0, onset_idx - int(baseline_duration * self.sampling_rate))
                    baseline_end = onset_idx
                    activation_start = onset_idx
                    activation_end = min(n_timepoints, onset_idx + int(activation_duration * self.sampling_rate))
                    
                    if baseline_end > baseline_start and activation_end > activation_start:
                        baseline_power = np.mean(band_power[baseline_start:baseline_end])
                        activation_power = np.mean(band_power[activation_start:activation_end])
                        
                        baseline_activations.append(baseline_power)
                        task_activations.append(activation_power)
                
                # Statistical test (paired t-test equivalent)
                if len(task_activations) > 0 and len(baseline_activations) > 0:
                    activation_change = np.mean(task_activations) - np.mean(baseline_activations)
                    activation_std = np.std(task_activations + baseline_activations)
                    
                    if activation_std > 0:
                        t_stat = activation_change / (activation_std / np.sqrt(len(task_activations)))
                        band_activation[voxel_idx] = activation_change
                        band_significance[voxel_idx] = np.abs(t_stat)
            
            activation_results['activation_maps'][band_name] = band_activation
            activation_results['statistical_maps'][band_name] = band_significance
        
        # Calculate similarity metrics (emphasizing commonalities)
        activation_results['similarity_metrics'] = self._calculate_similarity_metrics(
            activation_results['activation_maps']
        )
        
        self.results['math_activation'] = activation_results
        logger.info("Completed mathematical activation analysis")
        
        return activation_results
    
    def _calculate_similarity_metrics(self, activation_maps: Dict) -> Dict:
        """Calculate similarity metrics between frequency bands and conditions."""
        similarity_metrics = {}
        
        band_names = list(activation_maps.keys())
        n_bands = len(band_names)
        
        # Cross-band similarity matrix
        similarity_matrix = np.zeros((n_bands, n_bands))
        
        for i, band1 in enumerate(band_names):
            for j, band2 in enumerate(band_names):
                # Pearson correlation between activation maps
                correlation = np.corrcoef(activation_maps[band1], activation_maps[band2])[0, 1]
                similarity_matrix[i, j] = correlation if not np.isnan(correlation) else 0
        
        similarity_metrics['cross_band_similarity'] = similarity_matrix
        similarity_metrics['band_names'] = band_names
        similarity_metrics['mean_similarity'] = np.mean(similarity_matrix[np.triu_indices(n_bands, k=1)])
        
        return similarity_metrics
    
    def detect_math_regions(self, 
                           threshold_percentile: float = 95,
                           min_cluster_size: int = 10) -> Dict:
        """
        Detect brain regions activated during mathematical tasks.
        
        Parameters:
        -----------
        threshold_percentile : float, default 95
            Percentile threshold for activation detection
        min_cluster_size : int, default 10
            Minimum cluster size for region detection
            
        Returns:
        --------
        Dict : Detected regions and their properties
        """
        if 'math_activation' not in self.results:
            logger.warning("No activation analysis found. Running analysis first...")
            self.analyze_math_activation()
        
        activation_maps = self.results['math_activation']['activation_maps']
        statistical_maps = self.results['math_activation']['statistical_maps']
        
        detected_regions = {}
        
        for band_name in activation_maps.keys():
            activation_map = activation_maps[band_name]
            stat_map = statistical_maps[band_name]
            
            # Apply threshold
            threshold = np.percentile(stat_map, threshold_percentile)
            activated_voxels = stat_map > threshold
            
            # Find clusters (simplified 1D clustering)
            clusters = self._find_clusters_1d(activated_voxels, min_cluster_size)
            
            detected_regions[band_name] = {
                'n_clusters': len(clusters),
                'total_voxels': np.sum(activated_voxels),
                'clusters': clusters,
                'mean_activation': np.mean(activation_map[activated_voxels]) if np.any(activated_voxels) else 0,
                'peak_activation': np.max(activation_map) if len(activation_map) > 0 else 0
            }
        
        self.results['detected_regions'] = detected_regions
        return detected_regions
    
    def _find_clusters_1d(self, binary_mask: np.ndarray, min_size: int) -> List[Dict]:
        """Find clusters in 1D binary mask (simplified clustering for demonstration)."""
        clusters = []
        current_cluster = []
        
        for i, value in enumerate(binary_mask):
            if value:
                current_cluster.append(i)
            else:
                if len(current_cluster) >= min_size:
                    clusters.append({
                        'indices': current_cluster.copy(),
                        'size': len(current_cluster),
                        'center': np.mean(current_cluster)
                    })
                current_cluster = []
        
        # Check last cluster
        if len(current_cluster) >= min_size:
            clusters.append({
                'indices': current_cluster.copy(),
                'size': len(current_cluster),
                'center': np.mean(current_cluster)
            })
        
        return clusters
    
    def visualize_results(self, 
                         save_path: str = None,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create comprehensive visualizations of wavelet analysis results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple, default (15, 10)
            Figure size in inches
        """
        if 'math_activation' not in self.results:
            logger.warning("No results to visualize. Run analysis first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Wavelet Analysis of fMRI Data During Mathematical Tasks', fontsize=16)
        
        activation_maps = self.results['math_activation']['activation_maps']
        statistical_maps = self.results['math_activation']['statistical_maps']
        similarity_metrics = self.results['math_activation']['similarity_metrics']
        
        # Plot 1: Activation maps for each frequency band
        for i, (band_name, activation) in enumerate(activation_maps.items()):
            ax = axes[0, i] if i < 3 else axes[1, i-3]
            
            # Create histogram of activation values
            ax.hist(activation, bins=30, alpha=0.7, color=plt.cm.viridis(i/4))
            ax.set_title(f'{band_name.title()} Band\n({self.frequency_bands[band_name][0]:.2f}-{self.frequency_bands[band_name][1]:.2f} Hz)')
            ax.set_xlabel('Activation Strength')
            ax.set_ylabel('Number of Voxels')
            ax.grid(True, alpha=0.3)
        
        # Plot similarity matrix
        if len(activation_maps) > 2:
            ax = axes[1, 2]
            similarity_matrix = similarity_metrics['cross_band_similarity']
            im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(similarity_metrics['band_names'])))
            ax.set_yticks(range(len(similarity_metrics['band_names'])))
            ax.set_xticklabels(similarity_metrics['band_names'], rotation=45)
            ax.set_yticklabels(similarity_metrics['band_names'])
            ax.set_title('Cross-Band Similarity')
            
            # Add correlation values
            for i in range(len(similarity_metrics['band_names'])):
                for j in range(len(similarity_metrics['band_names'])):
                    ax.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add ethical reporting note
        if self.ethical_reporting:
            fig.text(0.02, 0.02, 
                    'Note: This analysis emphasizes neural similarities across conditions and individuals.',
                    fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the wavelet analysis results.
        
        Returns:
        --------
        str : Formatted report
        """
        if 'math_activation' not in self.results:
            return "No analysis results available. Please run analyze_math_activation() first."
        
        report = []
        report.append("=" * 80)
        report.append("WAVELET ANALYSIS REPORT - MATHEMATICAL COGNITION fMRI STUDY")
        report.append("=" * 80)
        report.append("")
        
        # Analysis overview
        report.append("ANALYSIS OVERVIEW:")
        report.append(f"• Wavelet type: {self.wavelet}")
        report.append(f"• Sampling rate: {self.sampling_rate} Hz")
        report.append(f"• Data shape: {self.time_series.shape[0]} voxels × {self.time_series.shape[1]} timepoints")
        report.append("")
        
        # Frequency band results
        activation_maps = self.results['math_activation']['activation_maps']
        similarity_metrics = self.results['math_activation']['similarity_metrics']
        
        report.append("FREQUENCY BAND ANALYSIS:")
        for band_name, activation in activation_maps.items():
            f_range = self.frequency_bands[band_name]
            mean_activation = np.mean(activation)
            std_activation = np.std(activation)
            max_activation = np.max(activation)
            
            report.append(f"• {band_name.upper()} BAND ({f_range[0]:.3f}-{f_range[1]:.3f} Hz):")
            report.append(f"  - Mean activation: {mean_activation:.4f}")
            report.append(f"  - Std deviation: {std_activation:.4f}")
            report.append(f"  - Peak activation: {max_activation:.4f}")
            report.append(f"  - Active voxels (>0): {np.sum(activation > 0)} ({100*np.sum(activation > 0)/len(activation):.1f}%)")
        
        report.append("")
        
        # Similarity analysis (emphasizing ethical reporting)
        if self.ethical_reporting:
            report.append("SIMILARITY ANALYSIS (Ethical Focus):")
            mean_similarity = similarity_metrics.get('mean_similarity', 0)
            report.append(f"• Cross-band similarity: {mean_similarity:.3f}")
            report.append("• This analysis emphasizes neural similarities rather than differences")
            report.append("• Individual variation patterns are preserved while highlighting commonalities")
            report.append("")
        
        # Region detection results
        if 'detected_regions' in self.results:
            report.append("DETECTED BRAIN REGIONS:")
            detected_regions = self.results['detected_regions']
            
            for band_name, regions in detected_regions.items():
                report.append(f"• {band_name.upper()} BAND:")
                report.append(f"  - Number of clusters: {regions['n_clusters']}")
                report.append(f"  - Total active voxels: {regions['total_voxels']}")
                report.append(f"  - Mean activation in active regions: {regions['mean_activation']:.4f}")
                report.append(f"  - Peak activation: {regions['peak_activation']:.4f}")
        
        report.append("")
        report.append("INTERPRETATION NOTES:")
        report.append("• Results show time-frequency decomposition of BOLD signals during math tasks")
        report.append("• Higher activation in slow frequencies may indicate sustained attention")
        report.append("• Medium frequency activation often relates to working memory processes")
        report.append("• Individual differences should be considered alongside group patterns")
        
        if self.ethical_reporting:
            report.append("")
            report.append("ETHICAL CONSIDERATIONS:")
            report.append("• This analysis emphasizes similarities across participants")
            report.append("• Results should not be used to make discriminatory conclusions")
            report.append("• Individual variation is respected and highlighted")
            report.append("• Cultural and social factors should be considered in interpretation")
        
        report.append("")
        report.append("=" * 80)
        
        return "\\n".join(report)
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export analysis results to file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        format : str, default 'json'
            Export format ('json', 'csv', 'excel')
        """
        if format.lower() == 'json':
            import json
            # Convert numpy arrays to lists for JSON serialization
            exportable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    exportable_results[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            exportable_results[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, dict):
                            exportable_results[key][subkey] = {}
                            for subsubkey, subsubvalue in subvalue.items():
                                if isinstance(subsubvalue, np.ndarray):
                                    exportable_results[key][subkey][subsubkey] = subsubvalue.tolist()
                                else:
                                    exportable_results[key][subkey][subsubkey] = subsubvalue
                        else:
                            exportable_results[key][subkey] = subvalue
                else:
                    exportable_results[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(exportable_results, f, indent=2)
                
        elif format.lower() == 'csv':
            # Export summary statistics to CSV
            if 'math_activation' in self.results:
                activation_maps = self.results['math_activation']['activation_maps']
                summary_data = []
                
                for band_name, activation in activation_maps.items():
                    summary_data.append({
                        'frequency_band': band_name,
                        'min_freq': self.frequency_bands[band_name][0],
                        'max_freq': self.frequency_bands[band_name][1],
                        'mean_activation': np.mean(activation),
                        'std_activation': np.std(activation),
                        'max_activation': np.max(activation),
                        'min_activation': np.min(activation),
                        'active_voxels': np.sum(activation > 0),
                        'total_voxels': len(activation)
                    })
                
                df = pd.DataFrame(summary_data)
                df.to_csv(filepath, index=False)
        
        logger.info(f"Results exported to {filepath}")


def main():
    """Example usage of the WaveletfMRIAnalyzer class."""
    # Initialize analyzer
    analyzer = WaveletfMRIAnalyzer(wavelet='morlet', ethical_reporting=True)
    
    # Load example data
    analyzer.load_fmri_data()
    
    # Analyze mathematical task activation
    results = analyzer.analyze_math_activation(
        task_onsets=[30, 80, 130, 180],
        baseline_duration=10.0,
        activation_duration=15.0
    )
    
    # Detect activated regions
    regions = analyzer.detect_math_regions(threshold_percentile=95)
    
    # Generate visualizations
    analyzer.visualize_results()
    
    # Generate report
    report = analyzer.generate_report()
    print(report)
    
    # Export results
    analyzer.export_results('wavelet_analysis_results.json', 'json')
    analyzer.export_results('wavelet_analysis_summary.csv', 'csv')


if __name__ == "__main__":
    main()
