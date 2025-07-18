"""
Brain Visualization Module for Wavelet Analysis Results

This module provides advanced visualization capabilities for fMRI wavelet analysis
results, with specific focus on mathematical cognition activation patterns.

Author: MCP-fMRI Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# Try to import neuroimaging visualization libraries
try:
    from nilearn import plotting, datasets
    from nilearn.image import load_img
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    warnings.warn("Nilearn not available. Using standard plotting only.")

logger = logging.getLogger(__name__)


class BrainActivationVisualizer:
    """
    Advanced visualization class for fMRI wavelet analysis results.
    
    Provides methods for:
    - Interactive brain activation maps
    - Time-frequency spectrograms
    - Statistical significance maps
    - Comparative analysis plots
    - 3D brain rendering (when nilearn available)
    """
    
    def __init__(self, ethical_reporting: bool = True):
        """
        Initialize the brain visualization class.
        
        Parameters:
        -----------
        ethical_reporting : bool, default True
            Whether to emphasize similarities in visualizations
        """
        self.ethical_reporting = ethical_reporting
        
        # Custom colormaps for brain activation
        self.activation_cmap = LinearSegmentedColormap.from_list(
            'activation', ['darkblue', 'blue', 'lightblue', 'white', 
                          'yellow', 'orange', 'red', 'darkred'])
        
        # Mathematical brain regions (simplified coordinates for demonstration)
        self.math_regions = {
            'left_parietal': {'x': -40, 'y': -50, 'z': 50, 'name': 'Left Parietal (Number Processing)'},
            'right_parietal': {'x': 40, 'y': -50, 'z': 50, 'name': 'Right Parietal (Spatial Processing)'},
            'left_prefrontal': {'x': -45, 'y': 20, 'z': 30, 'name': 'Left PFC (Working Memory)'},
            'right_prefrontal': {'x': 45, 'y': 20, 'z': 30, 'name': 'Right PFC (Attention)'},
            'anterior_cingulate': {'x': 0, 'y': 10, 'z': 35, 'name': 'ACC (Cognitive Control)'},
            'visual_cortex': {'x': 0, 'y': -80, 'z': 10, 'name': 'Visual Cortex (Number Symbols)'}
        }
    
    def plot_activation_timeseries(self, 
                                  wavelet_results: Dict,
                                  region_indices: Dict = None,
                                  save_path: str = None) -> None:
        """
        Plot time series of activation in different brain regions.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        region_indices : Dict, optional
            Indices for different brain regions
        save_path : str, optional
            Path to save the figure
        """
        if 'activation_maps' not in wavelet_results:
            logger.error("No activation maps found in results")
            return
        
        activation_maps = wavelet_results['activation_maps']
        n_bands = len(activation_maps)
        
        fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3*n_bands))
        if n_bands == 1:
            axes = [axes]
        
        # Time points (assuming 0.5 Hz sampling rate)
        n_timepoints = 200  # Default from example data
        time_points = np.linspace(0, n_timepoints/0.5, n_timepoints)
        
        for i, (band_name, activation) in enumerate(activation_maps.items()):
            ax = axes[i]
            
            # Simulate time series from activation maps (simplified)
            # In practice, this would use the actual time-frequency data
            n_voxels = len(activation)
            selected_voxels = np.random.choice(n_voxels, min(10, n_voxels), replace=False)
            
            for voxel_idx in selected_voxels:
                # Create example time series based on activation strength
                base_signal = np.random.normal(0, 0.1, n_timepoints)
                activation_signal = activation[voxel_idx] * np.sin(2 * np.pi * 0.02 * time_points)
                combined_signal = base_signal + activation_signal
                
                ax.plot(time_points, combined_signal, alpha=0.6, linewidth=1)
            
            ax.set_title(f'{band_name.title()} Band Activation Time Series')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('BOLD Signal Change (%)')
            ax.grid(True, alpha=0.3)
            
            # Add task onset markers
            task_onsets = [30, 80, 130, 180]
            for onset in task_onsets:
                ax.axvline(x=onset, color='red', linestyle='--', alpha=0.7, label='Math Task' if onset == task_onsets[0] else '')
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        
        if self.ethical_reporting:
            fig.text(0.02, 0.02, 
                    'Note: Individual time courses show diverse patterns highlighting neural variability.',
                    fontsize=9, style='italic', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        plt.show()
    
    def plot_frequency_spectrum(self, 
                               wavelet_results: Dict,
                               save_path: str = None) -> None:
        """
        Plot frequency spectrum of activation across different bands.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        save_path : str, optional
            Path to save the figure
        """
        if 'activation_maps' not in wavelet_results:
            logger.error("No activation maps found in results")
            return
        
        activation_maps = wavelet_results['activation_maps']
        frequency_bands = {
            'very_slow': (0.01, 0.03),
            'slow': (0.03, 0.06),
            'medium': (0.06, 0.12),
            'fast': (0.12, 0.25)
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of mean activation per frequency band
        band_names = list(activation_maps.keys())
        mean_activations = [np.mean(activation_maps[band]) for band in band_names]
        std_activations = [np.std(activation_maps[band]) for band in band_names]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(band_names)))
        bars = ax1.bar(band_names, mean_activations, yerr=std_activations, 
                      capsize=5, color=colors, alpha=0.8)
        
        ax1.set_title('Mean Activation by Frequency Band')
        ax1.set_ylabel('Mean Activation Strength')
        ax1.set_xlabel('Frequency Band')
        ax1.grid(True, alpha=0.3)
        
        # Add frequency range labels
        for i, (bar, band) in enumerate(zip(bars, band_names)):
            freq_range = frequency_bands.get(band, (0, 0))
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_activations[i] + 0.01,
                    f'{freq_range[0]:.2f}-{freq_range[1]:.2f} Hz', 
                    ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Violin plot showing distribution of activations
        activation_data = []
        band_labels = []
        
        for band_name, activation in activation_maps.items():
            activation_data.extend(activation)
            band_labels.extend([band_name] * len(activation))
        
        import pandas as pd
        df = pd.DataFrame({'activation': activation_data, 'band': band_labels})
        
        sns.violinplot(data=df, x='band', y='activation', ax=ax2, palette='viridis')
        ax2.set_title('Distribution of Activation Values')
        ax2.set_ylabel('Activation Strength')
        ax2.set_xlabel('Frequency Band')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Frequency spectrum plot saved to {save_path}")
        
        plt.show()
    
    def plot_brain_connectivity(self, 
                               wavelet_results: Dict,
                               save_path: str = None) -> None:
        """
        Plot brain connectivity patterns based on wavelet coherence.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        save_path : str, optional
            Path to save the figure
        """
        if 'similarity_metrics' not in wavelet_results:
            logger.error("No similarity metrics found in results")
            return
        
        similarity_metrics = wavelet_results['similarity_metrics']
        similarity_matrix = similarity_metrics.get('cross_band_similarity', np.eye(4))
        band_names = similarity_metrics.get('band_names', ['Band 1', 'Band 2', 'Band 3', 'Band 4'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Connectivity matrix heatmap
        im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(band_names)))
        ax1.set_yticks(range(len(band_names)))
        ax1.set_xticklabels(band_names, rotation=45)
        ax1.set_yticklabels(band_names)
        ax1.set_title('Cross-Frequency Band Connectivity')
        
        # Add correlation values
        for i in range(len(band_names)):
            for j in range(len(band_names)):
                ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=10,
                        color='white' if np.abs(similarity_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Correlation Coefficient')
        
        # Network graph visualization
        if len(band_names) > 1:
            # Create network layout
            n_nodes = len(band_names)
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            # Plot nodes
            ax2.scatter(x_pos, y_pos, s=500, c=range(n_nodes), cmap='viridis', alpha=0.8)
            
            # Add node labels
            for i, (x, y, name) in enumerate(zip(x_pos, y_pos, band_names)):
                ax2.text(x*1.1, y*1.1, name, ha='center', va='center', fontsize=10)
            
            # Plot connections (edges)
            threshold = 0.3  # Only show strong connections
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    correlation = similarity_matrix[i, j]
                    if abs(correlation) > threshold:
                        # Line thickness proportional to correlation strength
                        linewidth = abs(correlation) * 5
                        color = 'red' if correlation > 0 else 'blue'
                        alpha = min(abs(correlation), 0.8)
                        
                        ax2.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                                color=color, linewidth=linewidth, alpha=alpha)
            
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_aspect('equal')
            ax2.set_title('Frequency Band Network\\n(Connections > 0.3)')
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Connectivity plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_3d_brain(self, 
                                   wavelet_results: Dict,
                                   save_path: str = None) -> go.Figure:
        """
        Create interactive 3D brain visualization with activation overlays.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        save_path : str, optional
            Path to save the HTML file
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive 3D plot
        """
        activation_maps = wavelet_results.get('activation_maps', {})
        
        # Create 3D brain surface (simplified sphere for demonstration)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_brain = np.outer(np.cos(u), np.sin(v))
        y_brain = np.outer(np.sin(u), np.sin(v))
        z_brain = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add brain surface
        fig.add_trace(go.Surface(
            x=x_brain, y=y_brain, z=z_brain,
            colorscale='Greys',
            opacity=0.3,
            showscale=False,
            name='Brain Surface'
        ))
        
        # Add activation points for mathematical regions
        if activation_maps:
            # Use first frequency band for demonstration
            first_band = list(activation_maps.keys())[0]
            activations = activation_maps[first_band]
            
            # Select top activated "voxels" and map to brain regions
            n_regions = min(len(self.math_regions), 20)
            top_indices = np.argsort(activations)[-n_regions:]
            
            region_names = list(self.math_regions.keys())[:n_regions]
            
            for i, (region_key, activation_idx) in enumerate(zip(region_names, top_indices)):
                region = self.math_regions[region_key]
                activation_strength = activations[activation_idx]
                
                # Normalize coordinates to brain surface
                x, y, z = region['x']/100, region['y']/100, region['z']/100
                
                # Add activation marker
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(
                        size=max(5, activation_strength * 20),
                        color=activation_strength,
                        colorscale='Reds',
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name=region['name'],
                    text=f"Activation: {activation_strength:.3f}",
                    hovertemplate="<b>%{text}</b><br>Region: %{fullData.name}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive 3D Brain Activation Map<br><sub>Mathematical Cognition Task</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if self.ethical_reporting:
            fig.add_annotation(
                text="Note: Visualization emphasizes individual activation patterns and neural diversity",
                showarrow=False,
                x=0, y=0,
                xref="paper", yref="paper",
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="gray")
            )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive 3D brain plot saved to {save_path}")
        
        return fig
    
    def plot_statistical_maps(self, 
                             wavelet_results: Dict,
                             threshold: float = 2.0,
                             save_path: str = None) -> None:
        """
        Plot statistical significance maps of activation.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        threshold : float, default 2.0
            Statistical threshold for significance
        save_path : str, optional
            Path to save the figure
        """
        if 'statistical_maps' not in wavelet_results:
            logger.error("No statistical maps found in results")
            return
        
        statistical_maps = wavelet_results['statistical_maps']
        n_bands = len(statistical_maps)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (band_name, stat_map) in enumerate(statistical_maps.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            
            # Create 2D representation of statistical map
            # Reshape to approximate brain slice for visualization
            map_size = int(np.sqrt(len(stat_map)))
            if map_size * map_size == len(stat_map):
                stat_map_2d = stat_map.reshape(map_size, map_size)
            else:
                # Pad or truncate to make square
                target_size = 32
                if len(stat_map) > target_size**2:
                    stat_map_2d = stat_map[:target_size**2].reshape(target_size, target_size)
                else:
                    padded = np.zeros(target_size**2)
                    padded[:len(stat_map)] = stat_map
                    stat_map_2d = padded.reshape(target_size, target_size)
            
            # Apply threshold
            thresholded_map = np.where(stat_map_2d > threshold, stat_map_2d, np.nan)
            
            # Plot
            im = ax.imshow(stat_map_2d, cmap='hot', interpolation='bilinear')
            
            # Overlay thresholded regions
            ax.contour(thresholded_map, levels=[threshold], colors='cyan', linewidths=2)
            
            ax.set_title(f'{band_name.title()} Band\\nStatistical Map (t-statistic)')
            ax.set_xlabel('Voxel Index (X)')
            ax.set_ylabel('Voxel Index (Y)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8, label='t-statistic')
            
            # Add threshold line to colorbar
            ax.text(0.02, 0.98, f'Threshold: {threshold:.1f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for j in range(i+1, 4):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if self.ethical_reporting:
            fig.text(0.02, 0.02, 
                    'Note: Statistical maps show areas of reliable activation while preserving individual differences.',
                    fontsize=10, style='italic', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistical maps saved to {save_path}")
        
        plt.show()
    
    def create_dashboard(self, 
                        wavelet_results: Dict,
                        save_path: str = None) -> go.Figure:
        """
        Create comprehensive interactive dashboard of all results.
        
        Parameters:
        -----------
        wavelet_results : Dict
            Results from wavelet analysis
        save_path : str, optional
            Path to save the HTML file
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Activation by Frequency Band', 'Cross-Band Similarity', 
                           'Statistical Significance', 'Region Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        activation_maps = wavelet_results.get('activation_maps', {})
        similarity_metrics = wavelet_results.get('similarity_metrics', {})
        statistical_maps = wavelet_results.get('statistical_maps', {})
        
        # Plot 1: Activation by frequency band
        if activation_maps:
            band_names = list(activation_maps.keys())
            mean_activations = [np.mean(activation_maps[band]) for band in band_names]
            std_activations = [np.std(activation_maps[band]) for band in band_names]
            
            fig.add_trace(
                go.Bar(
                    x=band_names,
                    y=mean_activations,
                    error_y=dict(type='data', array=std_activations),
                    name='Mean Activation',
                    marker_color='skyblue'
                ),
                row=1, col=1
            )
        
        # Plot 2: Similarity matrix
        if 'cross_band_similarity' in similarity_metrics:
            similarity_matrix = similarity_metrics['cross_band_similarity']
            band_names = similarity_metrics.get('band_names', [])
            
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    x=band_names,
                    y=band_names,
                    colorscale='RdYlBu_r',
                    zmin=-1, zmax=1,
                    showscale=True
                ),
                row=1, col=2
            )
        
        # Plot 3: Statistical significance
        if statistical_maps:
            first_band = list(statistical_maps.keys())[0]
            stat_values = statistical_maps[first_band]
            
            fig.add_trace(
                go.Histogram(
                    x=stat_values,
                    nbinsx=30,
                    name='t-statistics',
                    marker_color='lightcoral'
                ),
                row=2, col=1
            )
        
        # Table 4: Summary statistics
        if activation_maps:
            table_data = []
            for band_name, activation in activation_maps.items():
                table_data.append([
                    band_name.title(),
                    f"{np.mean(activation):.3f}",
                    f"{np.std(activation):.3f}",
                    f"{np.max(activation):.3f}",
                    f"{np.sum(activation > 0)}"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Band', 'Mean', 'Std', 'Max', 'Active Voxels'],
                               fill_color='lightblue'),
                    cells=dict(values=list(zip(*table_data)),
                              fill_color='white')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Wavelet Analysis Dashboard - Mathematical Cognition fMRI",
                x=0.5,
                font=dict(size=16)
            ),
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Frequency Band", row=1, col=1)
        fig.update_yaxes(title_text="Mean Activation", row=1, col=1)
        fig.update_xaxes(title_text="t-statistic", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig


def main():
    """Example usage of the BrainActivationVisualizer class."""
    # Example wavelet results (normally from WaveletfMRIAnalyzer)
    example_results = {
        'activation_maps': {
            'very_slow': np.random.normal(0.1, 0.3, 1000),
            'slow': np.random.normal(0.2, 0.4, 1000),
            'medium': np.random.normal(0.15, 0.35, 1000),
            'fast': np.random.normal(0.05, 0.2, 1000)
        },
        'statistical_maps': {
            'very_slow': np.random.gamma(2, 1, 1000),
            'slow': np.random.gamma(2.5, 1, 1000),
            'medium': np.random.gamma(2.2, 1, 1000),
            'fast': np.random.gamma(1.8, 1, 1000)
        },
        'similarity_metrics': {
            'cross_band_similarity': np.array([
                [1.0, 0.6, 0.4, 0.2],
                [0.6, 1.0, 0.7, 0.3],
                [0.4, 0.7, 1.0, 0.5],
                [0.2, 0.3, 0.5, 1.0]
            ]),
            'band_names': ['very_slow', 'slow', 'medium', 'fast']
        }
    }
    
    # Initialize visualizer
    visualizer = BrainActivationVisualizer(ethical_reporting=True)
    
    # Create visualizations
    visualizer.plot_activation_timeseries(example_results)
    visualizer.plot_frequency_spectrum(example_results)
    visualizer.plot_brain_connectivity(example_results)
    visualizer.plot_statistical_maps(example_results)
    
    # Create interactive plots
    brain_3d = visualizer.create_interactive_3d_brain(example_results)
    dashboard = visualizer.create_dashboard(example_results)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
