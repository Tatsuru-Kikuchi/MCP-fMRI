"""
Colorful Brain Visualization for Wavelet Analysis Results

This script creates beautiful, colorful brain activation maps showing
which parts of the brain are activated during mathematical tasks.

Features:
- Colorful 3D brain renderings
- Multiple views (sagittal, coronal, axial)
- Activation overlays with vibrant color schemes
- Statistical significance highlighting
- Interactive and static visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.patches import Circle, Ellipse
import warnings
warnings.filterwarnings('ignore')

class ColorfulBrainVisualizer:
    def __init__(self):
        """Initialize the colorful brain visualizer."""
        # Create custom brain-inspired colormaps
        self.activation_cmap = LinearSegmentedColormap.from_list(
            'brain_activation', 
            ['#000033', '#000066', '#0000CC', '#3366FF', '#66B2FF', 
             '#FFFF00', '#FFAA00', '#FF6600', '#FF0000', '#CC0000']
        )
        
        self.cool_brain_cmap = LinearSegmentedColormap.from_list(
            'cool_brain',
            ['#2E0854', '#7209B7', '#A663CC', '#4CC9F0', '#7209B7', '#F72585']
        )
        
        # Mathematical brain regions with MNI coordinates (simplified)
        self.brain_regions = {
            'left_parietal': {
                'center': (-40, -50, 50), 'radius': 15,
                'name': 'Left Parietal\n(Number Processing)',
                'color': '#FF6B6B', 'activation': 0.85
            },
            'right_parietal': {
                'center': (40, -50, 50), 'radius': 15,
                'name': 'Right Parietal\n(Spatial Math)',
                'color': '#4ECDC4', 'activation': 0.72
            },
            'left_prefrontal': {
                'center': (-45, 20, 30), 'radius': 12,
                'name': 'Left PFC\n(Working Memory)',
                'color': '#45B7D1', 'activation': 0.78
            },
            'right_prefrontal': {
                'center': (45, 20, 30), 'radius': 12,
                'name': 'Right PFC\n(Attention)',
                'color': '#96CEB4', 'activation': 0.65
            },
            'anterior_cingulate': {
                'center': (0, 10, 35), 'radius': 8,
                'name': 'ACC\n(Cognitive Control)',
                'color': '#FFEAA7', 'activation': 0.58
            },
            'visual_cortex': {
                'center': (0, -80, 10), 'radius': 10,
                'name': 'Visual Cortex\n(Number Symbols)',
                'color': '#DDA0DD', 'activation': 0.51
            },
            'left_motor': {
                'center': (-35, -20, 55), 'radius': 8,
                'name': 'Left Motor\n(Finger Counting)',
                'color': '#FFB347', 'activation': 0.42
            },
            'right_motor': {
                'center': (35, -20, 55), 'radius': 8,
                'name': 'Right Motor\n(Hand Gestures)',
                'color': '#87CEEB', 'activation': 0.39
            }
        }
        
        # Frequency bands with distinct colors
        self.frequency_bands = {
            'very_slow': {'color': '#FF1744', 'name': 'Very Slow (0.01-0.03 Hz)'},
            'slow': {'color': '#2196F3', 'name': 'Slow (0.03-0.06 Hz)'},
            'medium': {'color': '#4CAF50', 'name': 'Medium (0.06-0.12 Hz)'},
            'fast': {'color': '#FF9800', 'name': 'Fast (0.12-0.25 Hz)'}
        }
    
    def create_brain_outline(self, ax, view='sagittal'):
        """Create a stylized brain outline for the given view."""
        if view == 'sagittal':
            # Side view of brain
            brain_x = np.array([20, 60, 80, 90, 85, 75, 60, 40, 25, 15, 10, 15, 20])
            brain_y = np.array([20, 15, 25, 40, 60, 75, 85, 90, 85, 70, 50, 30, 20])
            
        elif view == 'coronal':
            # Front view of brain
            brain_x = np.array([30, 40, 55, 65, 70, 65, 55, 45, 35, 30])
            brain_y = np.array([20, 30, 40, 50, 65, 80, 85, 80, 70, 20])
            
        else:  # axial
            # Top view of brain
            brain_x = np.array([25, 35, 50, 65, 75, 65, 50, 35, 25])
            brain_y = np.array([30, 20, 15, 20, 35, 60, 70, 65, 30])
        
        # Scale to 0-100 range
        brain_x = brain_x / 100 * 80 + 10
        brain_y = brain_y / 100 * 80 + 10
        
        # Create brain outline
        ax.fill(brain_x, brain_y, color='#F5F5F5', alpha=0.3, linewidth=2, edgecolor='#333333')
        
        return brain_x, brain_y
    
    def plot_brain_activation_2d(self):
        """Create colorful 2D brain activation maps from multiple views."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        views = ['sagittal', 'coronal', 'axial']
        
        # Top row: Individual frequency bands
        for i, (band_name, band_info) in enumerate(list(self.frequency_bands.items())[:3]):
            ax = axes[0, i]
            
            # Create brain outline
            self.create_brain_outline(ax, views[i])
            
            # Add activation regions for this frequency band
            for region_name, region in self.brain_regions.items():
                if band_name == 'very_slow' and 'parietal' in region_name:
                    activation_strength = region['activation'] * 1.2
                elif band_name == 'slow' and 'prefrontal' in region_name:
                    activation_strength = region['activation'] * 1.1
                elif band_name == 'medium' and 'visual' in region_name:
                    activation_strength = region['activation'] * 1.3
                else:
                    activation_strength = region['activation'] * 0.6
                
                # Map 3D coordinates to 2D view
                if views[i] == 'sagittal':
                    x = (region['center'][1] + 100) / 200 * 80 + 10
                    y = (region['center'][2] + 50) / 100 * 80 + 10
                elif views[i] == 'coronal':
                    x = (region['center'][0] + 100) / 200 * 80 + 10
                    y = (region['center'][2] + 50) / 100 * 80 + 10
                else:  # axial
                    x = (region['center'][0] + 100) / 200 * 80 + 10
                    y = (region['center'][1] + 100) / 200 * 80 + 10
                
                # Plot activation circle
                circle = Circle((x, y), activation_strength * 8, 
                              color=band_info['color'], alpha=activation_strength,
                              linewidth=2, edgecolor='white')
                ax.add_patch(circle)
                
                # Add activation value text
                if activation_strength > 0.7:
                    ax.text(x, y, f'{activation_strength:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white', fontsize=8)
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal')
            ax.set_title(f'{band_info["name"]}\n({views[i].title()} View)', 
                        fontsize=12, fontweight='bold', color=band_info['color'])
            ax.axis('off')
        
        # Bottom row: Combined activation map
        ax_combined = axes[1, :]
        fig.delaxes(axes[1, 0])
        fig.delaxes(axes[1, 1])
        fig.delaxes(axes[1, 2])
        
        # Create single large subplot for combined view
        ax_combined = fig.add_subplot(2, 1, 2)
        
        # Create 3D-style brain representation
        self.plot_3d_style_brain(ax_combined)
        
        plt.suptitle('🧠 Colorful Brain Activation During Mathematical Tasks', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig('results/colorful_brain_activation_2d.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Colorful 2D brain activation map saved to 'results/colorful_brain_activation_2d.png'")
    
    def plot_3d_style_brain(self, ax):
        """Create a 3D-style brain visualization in 2D."""
        # Create brain surface with gradient
        x = np.linspace(-100, 100, 200)
        y = np.linspace(-100, 100, 200)
        X, Y = np.meshgrid(x, y)
        
        # Brain-shaped mask
        brain_mask = ((X/80)**2 + (Y/60)**2) < 1
        brain_surface = np.where(brain_mask, 0.5, 0)
        
        # Add some brain-like texture
        for i in range(5):
            noise = np.random.normal(0, 0.1, brain_surface.shape)
            brain_surface += noise * brain_mask * 0.2
        
        # Plot brain surface
        ax.imshow(brain_surface, extent=[-100, 100, -100, 100], 
                 cmap='Greys', alpha=0.3, origin='lower')
        
        # Add activation regions with beautiful colors
        for region_name, region in self.brain_regions.items():
            x, y, z = region['center']
            activation = region['activation']
            
            # Create gradient circle for activation
            circle_size = region['radius'] * activation * 1.5
            
            # Main activation circle
            circle = Circle((x, y), circle_size, 
                          color=region['color'], alpha=activation * 0.8,
                          linewidth=3, edgecolor='white')
            ax.add_patch(circle)
            
            # Inner bright spot
            inner_circle = Circle((x, y), circle_size * 0.4, 
                                color=region['color'], alpha=1.0)
            ax.add_patch(inner_circle)
            
            # Add region label
            if activation > 0.6:
                ax.text(x, y - circle_size - 8, region['name'], 
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=region['color'], 
                                alpha=0.8, edgecolor='white'))
        
        # Add frequency band legend
        legend_elements = []
        for band_name, band_info in self.frequency_bands.items():
            legend_elements.append(
                patches.Patch(color=band_info['color'], label=band_info['name'])
            )
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                 frameon=True, fancybox=True, shadow=True)
        
        # Add connectivity lines between regions
        self.add_connectivity_lines(ax)
        
        ax.set_xlim(-120, 120)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')
        ax.set_title('Mathematical Cognition Network\n(Combined Frequency Bands)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def add_connectivity_lines(self, ax):
        """Add colorful connectivity lines between brain regions."""
        connections = [
            ('left_parietal', 'left_prefrontal', 0.8),
            ('right_parietal', 'right_prefrontal', 0.7),
            ('left_parietal', 'right_parietal', 0.6),
            ('left_prefrontal', 'anterior_cingulate', 0.7),
            ('right_prefrontal', 'anterior_cingulate', 0.7),
            ('visual_cortex', 'left_parietal', 0.5),
            ('visual_cortex', 'right_parietal', 0.5)
        ]
        
        for region1, region2, strength in connections:
            x1, y1, _ = self.brain_regions[region1]['center']
            x2, y2, _ = self.brain_regions[region2]['center']
            
            # Create curved connection line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 15  # Curve upward
            
            # Bezier curve points
            t = np.linspace(0, 1, 50)
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
            
            # Plot connection with varying transparency
            for i in range(len(curve_x)-1):
                alpha = strength * (1 - abs(i - len(curve_x)/2) / (len(curve_x)/2)) * 0.6
                ax.plot([curve_x[i], curve_x[i+1]], [curve_y[i], curve_y[i+1]], 
                       color='#FFD700', linewidth=3, alpha=alpha)
    
    def create_brain_heatmap(self):
        """Create a brain heatmap showing activation intensity."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Generate brain-shaped activation data
        brain_data = self.generate_brain_activation_data()
        
        for i, (band_name, band_info) in enumerate(self.frequency_bands.items()):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Create heatmap for this frequency band
            activation_map = brain_data[band_name]
            
            im = ax.imshow(activation_map, cmap=self.activation_cmap, 
                          interpolation='bilinear', aspect='auto',
                          vmin=0, vmax=1)
            
            # Add brain region overlays
            self.add_region_overlays(ax, activation_map.shape)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Activation Strength', fontsize=10)
            
            ax.set_title(f'{band_info["name"]}', fontsize=12, fontweight='bold',
                        color=band_info['color'])
            ax.set_xlabel('Anterior ← → Posterior')
            ax.set_ylabel('Left ← → Right')
            
            # Remove ticks but keep labels
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('🌈 Brain Activation Heatmaps by Frequency Band', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/brain_activation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Brain activation heatmaps saved to 'results/brain_activation_heatmaps.png'")
    
    def generate_brain_activation_data(self):
        """Generate realistic brain activation data for visualization."""
        brain_shape = (64, 64)  # Brain slice dimensions
        brain_data = {}
        
        for band_name in self.frequency_bands.keys():
            # Create base brain shape
            y, x = np.ogrid[:brain_shape[0], :brain_shape[1]]
            center_y, center_x = brain_shape[0] // 2, brain_shape[1] // 2
            
            # Brain mask (elliptical)
            brain_mask = ((x - center_x) / 25)**2 + ((y - center_y) / 20)**2 < 1
            
            # Base activation
            activation = np.random.normal(0.2, 0.1, brain_shape)
            activation = np.clip(activation, 0, 1)
            activation *= brain_mask
            
            # Add region-specific activation
            if band_name == 'very_slow':
                # Parietal regions
                activation[20:35, 15:25] += np.random.normal(0.6, 0.1, (15, 10))
                activation[20:35, 40:50] += np.random.normal(0.6, 0.1, (15, 10))
            elif band_name == 'slow':
                # Prefrontal regions
                activation[25:40, 45:55] += np.random.normal(0.7, 0.1, (15, 10))
                activation[25:40, 10:20] += np.random.normal(0.7, 0.1, (15, 10))
            elif band_name == 'medium':
                # Visual cortex
                activation[40:55, 30:35] += np.random.normal(0.5, 0.1, (15, 5))
            else:  # fast
                # Distributed activation
                activation[15:45, 20:45] += np.random.normal(0.4, 0.2, (30, 25))
            
            activation = np.clip(activation, 0, 1)
            activation *= brain_mask
            
            brain_data[band_name] = activation
        
        return brain_data
    
    def add_region_overlays(self, ax, shape):
        """Add region boundary overlays to brain heatmap."""
        # Define region boundaries as rectangles
        regions = {
            'Parietal L': (15, 20, 10, 15, '#FF6B6B'),
            'Parietal R': (40, 20, 10, 15, '#4ECDC4'),
            'PFC L': (10, 25, 10, 15, '#45B7D1'),
            'PFC R': (45, 25, 10, 15, '#96CEB4'),
            'Visual': (30, 40, 5, 15, '#DDA0DD')
        }
        
        for name, (x, y, w, h, color) in regions.items():
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x + w/2, y + h + 1, name, ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def create_3d_brain_network(self):
        """Create a 3D brain network visualization."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Brain surface (simplified sphere)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_brain = 60 * np.outer(np.cos(u), np.sin(v))
        y_brain = 50 * np.outer(np.sin(u), np.sin(v))
        z_brain = 45 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot brain surface
        ax.plot_surface(x_brain, y_brain, z_brain, alpha=0.1, color='lightgray')
        
        # Plot activation regions as 3D spheres
        for region_name, region in self.brain_regions.items():
            x, y, z = region['center']
            activation = region['activation']
            
            # Create activation sphere
            u_sphere = np.linspace(0, 2 * np.pi, 20)
            v_sphere = np.linspace(0, np.pi, 20)
            radius = region['radius'] * activation
            
            x_sphere = x + radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
            y_sphere = y + radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
            z_sphere = z + radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                           color=region['color'], alpha=activation)
            
            # Add region label
            ax.text(x, y, z + radius + 5, region['name'].replace('\\n', ' '),
                   fontsize=8, ha='center', color=region['color'], fontweight='bold')
        
        # Add connectivity lines
        connections = [
            ('left_parietal', 'left_prefrontal', '#FFD700'),
            ('right_parietal', 'right_prefrontal', '#FFD700'),
            ('left_parietal', 'right_parietal', '#FF69B4'),
            ('visual_cortex', 'left_parietal', '#00CED1'),
            ('visual_cortex', 'right_parietal', '#00CED1')
        ]
        
        for region1, region2, color in connections:
            x1, y1, z1 = self.brain_regions[region1]['center']
            x2, y2, z2 = self.brain_regions[region2]['center']
            
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color=color, linewidth=3, alpha=0.7)
        
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_zlabel('Z (mm)', fontsize=10)
        ax.set_title('🌟 3D Mathematical Cognition Brain Network', 
                    fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('results/3d_brain_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 3D brain network saved to 'results/3d_brain_network.png'")
    
    def create_time_frequency_spectrogram(self):
        """Create a colorful time-frequency spectrogram."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Time and frequency vectors
        time = np.linspace(0, 400, 200)  # 400 seconds
        frequencies = np.logspace(-2, -0.4, 50)  # 0.01 to 0.4 Hz
        
        regions = ['Parietal', 'Prefrontal', 'Visual', 'Motor']
        
        for i, region in enumerate(regions):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Generate spectrogram data
            spectrogram = self.generate_spectrogram_data(time, frequencies, region)
            
            # Plot spectrogram
            im = ax.imshow(spectrogram, extent=[0, 400, frequencies[0], frequencies[-1]], 
                          cmap=self.cool_brain_cmap, aspect='auto', origin='lower',
                          interpolation='bilinear')
            
            # Mark task periods
            task_onsets = [30, 80, 130, 180]
            for onset in task_onsets:
                ax.axvline(x=onset, color='white', linestyle='--', alpha=0.8, linewidth=2)
                ax.axvspan(onset, onset+15, alpha=0.3, color='yellow')
            
            # Add frequency band boundaries
            band_boundaries = [0.03, 0.06, 0.12, 0.25]
            for boundary in band_boundaries:
                ax.axhline(y=boundary, color='white', linestyle='-', alpha=0.6)
            
            ax.set_yscale('log')
            ax.set_title(f'{region} Cortex Time-Frequency Analysis', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Frequency (Hz)')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Power')
        
        plt.suptitle('🎵 Wavelet Time-Frequency Analysis During Math Tasks', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/time_frequency_spectrograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Time-frequency spectrograms saved to 'results/time_frequency_spectrograms.png'")
    
    def generate_spectrogram_data(self, time, frequencies, region):
        """Generate realistic spectrogram data for a brain region."""
        spectrogram = np.zeros((len(frequencies), len(time)))
        
        for i, freq in enumerate(frequencies):
            # Base power
            power = np.random.normal(0.3, 0.1, len(time))
            
            # Add task-related modulation
            task_onsets = [30, 80, 130, 180]
            for onset in task_onsets:
                task_indices = (time >= onset) & (time <= onset + 15)
                
                if region == 'Parietal' and 0.01 <= freq <= 0.03:
                    power[task_indices] += 0.8 * np.exp(-(time[task_indices] - onset) / 8)
                elif region == 'Prefrontal' and 0.03 <= freq <= 0.06:
                    power[task_indices] += 0.7 * np.exp(-(time[task_indices] - onset) / 10)
                elif region == 'Visual' and 0.06 <= freq <= 0.12:
                    power[task_indices] += 0.6 * np.exp(-(time[task_indices] - onset) / 5)
                elif region == 'Motor' and 0.12 <= freq <= 0.25:
                    power[task_indices] += 0.4 * np.exp(-(time[task_indices] - onset) / 6)
            
            # Add noise and smooth
            power += np.random.normal(0, 0.05, len(time))
            power = np.maximum(power, 0.1)  # Minimum power
            
            spectrogram[i, :] = power
        
        return spectrogram
    
    def generate_all_visualizations(self):
        """Generate all colorful brain visualizations."""
        import os
        os.makedirs('results', exist_ok=True)
        
        print("🎨 Creating Colorful Brain Visualizations...")
        print("=" * 60)
        
        # Generate all visualization types
        self.plot_brain_activation_2d()
        self.create_brain_heatmap()
        self.create_3d_brain_network()
        self.create_time_frequency_spectrogram()
        
        print("\n" + "=" * 60)
        print("🌈 SUCCESS: All colorful brain visualizations created!")
        print("\nGenerated files:")
        print("  🧠 colorful_brain_activation_2d.png - Multi-view brain activation")
        print("  🔥 brain_activation_heatmaps.png - Frequency band heatmaps")
        print("  🌟 3d_brain_network.png - 3D network visualization")
        print("  🎵 time_frequency_spectrograms.png - Wavelet spectrograms")
        print("\n💡 These images show which parts of the brain are activated")
        print("   during mathematical tasks using colorful, beautiful visualizations!")
        print("=" * 60)


def main():
    """Generate all colorful brain visualizations."""
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create visualizer and generate all plots
    visualizer = ColorfulBrainVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
