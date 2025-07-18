"""
Generate Demonstration Figures for Wavelet Analysis

This script creates example figures showing wavelet analysis results
for mathematical cognition fMRI data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pandas as pd

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class WaveletAnalysisVisualizer:
    def __init__(self):
        self.sampling_rate = 0.5  # Hz (TR = 2s)
        self.frequency_bands = {
            'very_slow': (0.01, 0.03),
            'slow': (0.03, 0.06), 
            'medium': (0.06, 0.12),
            'fast': (0.12, 0.25)
        }
    
    def generate_demo_data(self):
        """Generate synthetic fMRI data with mathematical task activation."""
        n_voxels = 1000
        n_timepoints = 200
        
        # Time vector
        t = np.linspace(0, n_timepoints/self.sampling_rate, n_timepoints)
        
        # Task onsets
        task_onsets = [30, 80, 130, 180]
        
        data = np.zeros((n_voxels, n_timepoints))
        
        # Create activation patterns for different brain regions
        for i in range(n_voxels):
            base_signal = np.random.normal(0, 0.1, n_timepoints)
            
            if i < 200:  # Parietal cortex - mathematical processing
                for onset in task_onsets:
                    onset_idx = int(onset * self.sampling_rate)
                    end_idx = min(onset_idx + 15, n_timepoints)
                    hrf_length = end_idx - onset_idx
                    hrf = np.exp(-np.arange(hrf_length)/5) * np.sin(np.arange(hrf_length)/2)
                    base_signal[onset_idx:end_idx] += 0.8 * hrf
                    
            elif i < 400:  # Prefrontal cortex - working memory
                for onset in task_onsets:
                    onset_idx = int(onset * self.sampling_rate)
                    end_idx = min(onset_idx + 20, n_timepoints)
                    hrf_length = end_idx - onset_idx
                    hrf = np.exp(-np.arange(hrf_length)/8)
                    base_signal[onset_idx:end_idx] += 0.6 * hrf
                    
            elif i < 600:  # Visual cortex - number processing
                for onset in task_onsets:
                    onset_idx = max(0, int(onset * self.sampling_rate) - 2)
                    end_idx = min(onset_idx + 8, n_timepoints)
                    hrf_length = end_idx - onset_idx
                    hrf = np.exp(-np.arange(hrf_length)/3)
                    base_signal[onset_idx:end_idx] += 0.4 * hrf
            
            # Add noise and drift
            drift = 0.1 * np.sin(2 * np.pi * 0.008 * t)
            noise = np.random.normal(0, 0.08, n_timepoints)
            data[i, :] = base_signal + drift + noise
        
        return data, t, task_onsets
    
    def analyze_frequency_bands(self, data):
        """Simulate wavelet analysis results for different frequency bands."""
        n_voxels = data.shape[0]
        
        # Simulate activation maps for each frequency band
        activation_maps = {}
        statistical_maps = {}
        
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            # Simulate different activation patterns
            if band_name == 'very_slow':
                activation = np.random.gamma(1.5, 0.2, n_voxels)
                activation[:200] += np.random.gamma(2, 0.3, 200)  # Parietal
            elif band_name == 'slow':
                activation = np.random.gamma(2, 0.25, n_voxels)
                activation[200:400] += np.random.gamma(2.5, 0.4, 200)  # Prefrontal
            elif band_name == 'medium':
                activation = np.random.gamma(1.8, 0.3, n_voxels)
                activation[400:600] += np.random.gamma(2.2, 0.35, 200)  # Visual
            else:  # fast
                activation = np.random.gamma(1.2, 0.2, n_voxels)
                activation[:400] += np.random.gamma(1.8, 0.25, 400)  # Task-related
            
            # Add some negative values for realistic data
            activation = activation - np.mean(activation)
            
            # Statistical significance (t-statistics)
            t_stats = np.abs(activation) / (np.std(activation) * np.sqrt(n_voxels/100))
            
            activation_maps[band_name] = activation
            statistical_maps[band_name] = t_stats
        
        return activation_maps, statistical_maps
    
    def create_figures(self):
        """Create comprehensive demonstration figures."""
        # Generate data
        print("Generating synthetic fMRI data...")
        fmri_data, time_vector, task_onsets = self.generate_demo_data()
        activation_maps, statistical_maps = self.analyze_frequency_bands(fmri_data)
        
        # Figure 1: Time series visualization
        self.plot_time_series(fmri_data, time_vector, task_onsets)
        
        # Figure 2: Frequency band activation
        self.plot_frequency_activation(activation_maps)
        
        # Figure 3: Statistical significance maps
        self.plot_statistical_maps(statistical_maps)
        
        # Figure 4: Brain connectivity
        self.plot_brain_connectivity(activation_maps)
        
        # Figure 5: Wavelet spectrogram
        self.plot_wavelet_spectrogram(fmri_data, time_vector, task_onsets)
        
        print("All figures generated successfully!")
    
    def plot_time_series(self, data, time_vector, task_onsets):
        """Plot sample time series from different brain regions."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Sample voxels from different regions
        parietal_voxel = 50   # Mathematical processing
        prefrontal_voxel = 250  # Working memory
        visual_voxel = 450    # Number processing
        
        regions = [
            (parietal_voxel, "Parietal Cortex (Mathematical Processing)", "red"),
            (prefrontal_voxel, "Prefrontal Cortex (Working Memory)", "blue"), 
            (visual_voxel, "Visual Cortex (Number Symbols)", "green")
        ]
        
        for i, (voxel_idx, region_name, color) in enumerate(regions):
            ax = axes[i]
            signal_data = data[voxel_idx, :]
            
            # Plot signal
            ax.plot(time_vector, signal_data, color=color, linewidth=1.5, alpha=0.8)
            
            # Mark task onsets
            for onset in task_onsets:
                ax.axvline(x=onset, color='black', linestyle='--', alpha=0.7, linewidth=2)
                ax.axvspan(onset, onset+15, alpha=0.2, color='yellow', label='Math Task' if onset == task_onsets[0] else '')
            
            ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('BOLD Signal Change (%)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(time_vector))
            
            if i == 0:
                ax.legend(loc='upper right')
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle('fMRI Time Series During Mathematical Tasks', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Time series plot saved to 'results/time_series_analysis.png'")
    
    def plot_frequency_activation(self, activation_maps):
        """Plot activation across frequency bands."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['darkblue', 'blue', 'orange', 'red']
        
        for i, (band_name, activation) in enumerate(activation_maps.items()):
            ax = axes[i]
            
            # Create histogram of activation values
            ax.hist(activation, bins=50, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_act = np.mean(activation)
            std_act = np.std(activation)
            positive_percent = 100 * np.sum(activation > 0) / len(activation)
            
            ax.axvline(mean_act, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_act:.3f}')
            ax.axvline(mean_act + 2*std_act, color='orange', linestyle=':', linewidth=2, 
                      label=f'2σ threshold: {mean_act + 2*std_act:.3f}')
            
            freq_range = self.frequency_bands[band_name]
            ax.set_title(f'{band_name.title()} Band ({freq_range[0]:.3f}-{freq_range[1]:.3f} Hz)\n'
                        f'{positive_percent:.1f}% positive activation', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Activation Strength')
            ax.set_ylabel('Number of Voxels')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Wavelet Analysis: Activation by Frequency Band', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/frequency_band_activation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Frequency band activation plot saved to 'results/frequency_band_activation.png'")
    
    def plot_statistical_maps(self, statistical_maps):
        """Plot statistical significance maps."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (band_name, t_stats) in enumerate(statistical_maps.items()):
            ax = axes[i]
            
            # Create 2D representation for visualization
            map_size = int(np.sqrt(len(t_stats)))
            if map_size * map_size != len(t_stats):
                map_size = 32
                t_stats_2d = np.zeros((map_size, map_size))
                t_stats_2d.flat[:len(t_stats)] = t_stats[:map_size*map_size]
            else:
                t_stats_2d = t_stats.reshape(map_size, map_size)
            
            # Apply threshold (p < 0.05 corresponds to t > 1.96 approximately)
            threshold = 1.96
            masked_stats = np.where(t_stats_2d > threshold, t_stats_2d, np.nan)
            
            # Plot statistical map
            im = ax.imshow(t_stats_2d, cmap='hot', interpolation='bilinear', aspect='auto')
            
            # Overlay significant regions
            ax.contour(masked_stats, levels=[threshold], colors='cyan', linewidths=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('t-statistic')
            
            # Statistics
            n_significant = np.sum(t_stats > threshold)
            percent_significant = 100 * n_significant / len(t_stats)
            
            ax.set_title(f'{band_name.title()} Band Statistical Map\n'
                        f'{n_significant} voxels significant ({percent_significant:.1f}%)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Voxel Index (X)')
            ax.set_ylabel('Voxel Index (Y)')
        
        plt.suptitle('Statistical Significance Maps (t-statistics)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/statistical_significance_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Statistical maps saved to 'results/statistical_significance_maps.png'")
    
    def plot_brain_connectivity(self, activation_maps):
        """Plot cross-frequency connectivity analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate cross-band correlation matrix
        band_names = list(activation_maps.keys())
        n_bands = len(band_names)
        correlation_matrix = np.zeros((n_bands, n_bands))
        
        for i, band1 in enumerate(band_names):
            for j, band2 in enumerate(band_names):
                correlation = np.corrcoef(activation_maps[band1], activation_maps[band2])[0, 1]
                correlation_matrix[i, j] = correlation
        
        # Plot 1: Correlation heatmap
        im1 = ax1.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(n_bands))
        ax1.set_yticks(range(n_bands))
        ax1.set_xticklabels([name.title() for name in band_names], rotation=45)
        ax1.set_yticklabels([name.title() for name in band_names])
        ax1.set_title('Cross-Frequency Band Correlations', fontsize=14, fontweight='bold')
        
        # Add correlation values to heatmap
        for i in range(n_bands):
            for j in range(n_bands):
                ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=10,
                        color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Correlation Coefficient')
        
        # Plot 2: Network graph
        angles = np.linspace(0, 2*np.pi, n_bands, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Plot nodes
        colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
        ax2.scatter(x_pos, y_pos, s=800, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, (x, y, name) in enumerate(zip(x_pos, y_pos, band_names)):
            ax2.text(x*1.15, y*1.15, name.title().replace('_', ' '), 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plot connections for strong correlations
        threshold = 0.3
        for i in range(n_bands):
            for j in range(i+1, n_bands):
                correlation = abs(correlation_matrix[i, j])
                if correlation > threshold:
                    linewidth = correlation * 8
                    color = 'red' if correlation_matrix[i, j] > 0 else 'blue'
                    alpha = min(correlation, 0.8)
                    
                    ax2.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                            color=color, linewidth=linewidth, alpha=alpha)
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.set_title('Frequency Band Network\n(Connections > 0.3)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle('Cross-Frequency Connectivity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/brain_connectivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Brain connectivity plot saved to 'results/brain_connectivity_analysis.png'")
    
    def plot_wavelet_spectrogram(self, data, time_vector, task_onsets):
        """Plot wavelet spectrogram for a sample voxel."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Select a highly activated voxel (from parietal region)
        voxel_signal = data[50, :]  # Parietal cortex voxel
        
        # Create spectrogram using scipy
        frequencies = np.logspace(-2, -0.5, 50)  # 0.01 to ~0.3 Hz
        widths = 6.0 / (2 * np.pi * frequencies)  # Scale for Morlet wavelet
        
        # Continuous wavelet transform approximation
        cwt_matrix = signal.cwt(voxel_signal, signal.morlet2, widths, w=6)
        cwt_power = np.abs(cwt_matrix) ** 2
        
        # Plot 1: Original signal
        ax1 = axes[0]
        ax1.plot(time_vector, voxel_signal, 'b-', linewidth=2, label='BOLD Signal')
        
        # Mark task periods
        for i, onset in enumerate(task_onsets):
            ax1.axvline(x=onset, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax1.axvspan(onset, onset+15, alpha=0.2, color='yellow', 
                       label='Math Task' if i == 0 else '')
        
        ax1.set_title('BOLD Signal from Parietal Cortex (Mathematical Processing Region)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Signal Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_vector))
        
        # Plot 2: Wavelet spectrogram
        ax2 = axes[1]
        im = ax2.imshow(cwt_power, extent=[0, max(time_vector), frequencies[0], frequencies[-1]], 
                       cmap='jet', aspect='auto', origin='lower', interpolation='bilinear')
        
        # Mark task periods on spectrogram
        for onset in task_onsets:
            ax2.axvline(x=onset, color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add frequency band boundaries
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            ax2.axhline(y=f_min, color='white', linestyle='-', alpha=0.6)
            ax2.axhline(y=f_max, color='white', linestyle='-', alpha=0.6)
            # Add band labels
            ax2.text(10, (f_min + f_max)/2, band_name.title().replace('_', ' '), 
                    color='white', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax2.set_title('Wavelet Time-Frequency Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_yscale('log')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Wavelet Power')
        
        plt.tight_layout()
        plt.savefig('results/wavelet_spectrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Wavelet spectrogram saved to 'results/wavelet_spectrogram.png'")


def main():
    """Generate all demonstration figures."""
    import os
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize visualizer and create figures
    visualizer = WaveletAnalysisVisualizer()
    
    print("🌊 Generating Wavelet Analysis Demonstration Figures")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        visualizer.create_figures()
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS: All figures generated successfully!")
        print("\nGenerated files in 'results/' directory:")
        print("  • time_series_analysis.png - Sample BOLD time series")
        print("  • frequency_band_activation.png - Multi-band activation analysis")
        print("  • statistical_significance_maps.png - Statistical t-maps")
        print("  • brain_connectivity_analysis.png - Cross-frequency connectivity")
        print("  • wavelet_spectrogram.png - Time-frequency decomposition")
        print("\n🧠 These figures demonstrate how wavelet analysis identifies")
        print("   brain regions activated during mathematical tasks.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")

if __name__ == "__main__":
    main()
