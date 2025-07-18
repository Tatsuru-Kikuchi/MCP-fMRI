#!/usr/bin/env python3
"""
Command-line interface for fMRI Wavelet Analysis

This script provides a simple command-line interface for running wavelet analysis
on fMRI data during mathematical cognition tasks.

Usage:
    python run_wavelet_analysis.py [options]

Example:
    python run_wavelet_analysis.py --input data.nii.gz --output results/ --visualize

Author: MCP-fMRI Project
License: MIT
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from wavelet_analysis import WaveletfMRIAnalyzer
from brain_visualization import BrainActivationVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Wavelet analysis for fMRI mathematical cognition data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with synthetic data
  python run_wavelet_analysis.py --output results/

  # Analysis with real fMRI data
  python run_wavelet_analysis.py --input data.nii.gz --mask brain_mask.nii.gz --output results/

  # Full analysis with visualizations
  python run_wavelet_analysis.py --input data.nii.gz --output results/ --visualize --interactive

  # Custom task timing
  python run_wavelet_analysis.py --onsets 30,80,130,180 --baseline 15 --duration 20
        """
    )
    
    # Input/Output options
    parser.add_argument('--input', '-i', type=str,
                        help='Path to input fMRI NIfTI file')
    parser.add_argument('--mask', '-m', type=str,
                        help='Path to brain mask NIfTI file')
    parser.add_argument('--output', '-o', type=str, default='results',
                        help='Output directory for results (default: results)')
    
    # Analysis parameters
    parser.add_argument('--wavelet', '-w', type=str, default='morlet',
                        choices=['morlet', 'mexican_hat', 'db4', 'haar'],
                        help='Wavelet type for analysis (default: morlet)')
    parser.add_argument('--sampling-rate', '-sr', type=float, default=0.5,
                        help='fMRI sampling rate in Hz (default: 0.5)')
    parser.add_argument('--onsets', type=str, default='30,80,130,180',
                        help='Task onset times in seconds, comma-separated (default: 30,80,130,180)')
    parser.add_argument('--baseline', '-b', type=float, default=10.0,
                        help='Baseline duration before task onset in seconds (default: 10.0)')
    parser.add_argument('--duration', '-d', type=float, default=15.0,
                        help='Activation duration after task onset in seconds (default: 15.0)')
    
    # Visualization options
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Create static visualizations')
    parser.add_argument('--interactive', action='store_true',
                        help='Create interactive visualizations')
    parser.add_argument('--no-ethics', action='store_true',
                        help='Disable ethical reporting mode')
    
    # Detection parameters
    parser.add_argument('--threshold', '-t', type=float, default=95.0,
                        help='Percentile threshold for region detection (default: 95.0)')
    parser.add_argument('--min-cluster', type=int, default=10,
                        help='Minimum cluster size for region detection (default: 10)')
    
    # Export options
    parser.add_argument('--export-json', action='store_true',
                        help='Export results as JSON')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export summary as CSV')
    parser.add_argument('--save-report', action='store_true',
                        help='Save text report')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output except errors')
    
    return parser.parse_args()


def setup_logging(args):
    """Configure logging based on arguments."""
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def parse_onsets(onset_string):
    """Parse onset times from string."""
    try:
        onsets = [float(x.strip()) for x in onset_string.split(',')]
        return onsets
    except ValueError:
        raise ValueError(f"Invalid onset format: {onset_string}. Use comma-separated numbers.")


def main():
    """Main analysis pipeline."""
    args = parse_arguments()
    setup_logging(args)
    
    logger.info("Starting fMRI Wavelet Analysis")
    logger.info(f"Wavelet type: {args.wavelet}")
    logger.info(f"Sampling rate: {args.sampling_rate} Hz")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Parse task onsets
    try:
        task_onsets = parse_onsets(args.onsets)
        logger.info(f"Task onsets: {task_onsets} seconds")
    except ValueError as e:
        logger.error(f"Error parsing onsets: {e}")
        sys.exit(1)
    
    # Initialize analyzer
    logger.info("Initializing wavelet analyzer...")
    analyzer = WaveletfMRIAnalyzer(
        wavelet=args.wavelet,
        sampling_rate=args.sampling_rate,
        ethical_reporting=not args.no_ethics
    )
    
    # Load data
    logger.info("Loading fMRI data...")
    try:
        if args.input:
            time_series = analyzer.load_fmri_data(
                nifti_path=args.input,
                mask_path=args.mask
            )
            logger.info(f"Loaded data from {args.input}")
        else:
            time_series = analyzer.load_fmri_data()
            logger.info("Using synthetic demonstration data")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run wavelet analysis
    logger.info("Performing wavelet analysis...")
    try:
        activation_results = analyzer.analyze_math_activation(
            task_onsets=task_onsets,
            baseline_duration=args.baseline,
            activation_duration=args.duration
        )
        logger.info("Wavelet analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in wavelet analysis: {e}")
        sys.exit(1)
    
    # Detect brain regions
    logger.info("Detecting activated brain regions...")
    try:
        detected_regions = analyzer.detect_math_regions(
            threshold_percentile=args.threshold,
            min_cluster_size=args.min_cluster
        )
        logger.info("Region detection completed")
        
        # Print summary
        total_regions = sum(regions['n_clusters'] for regions in detected_regions.values())
        logger.info(f"Total activated regions found: {total_regions}")
        
    except Exception as e:
        logger.error(f"Error in region detection: {e}")
        sys.exit(1)
    
    # Generate report
    if args.save_report:
        logger.info("Generating analysis report...")
        report = analyzer.generate_report()
        report_path = output_dir / 'wavelet_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    # Export results
    if args.export_json:
        logger.info("Exporting results to JSON...")
        json_path = output_dir / 'wavelet_results.json'
        analyzer.export_results(str(json_path), 'json')
        logger.info(f"JSON results saved to {json_path}")
    
    if args.export_csv:
        logger.info("Exporting summary to CSV...")
        csv_path = output_dir / 'wavelet_summary.csv'
        analyzer.export_results(str(csv_path), 'csv')
        logger.info(f"CSV summary saved to {csv_path}")
    
    # Create visualizations
    if args.visualize or args.interactive:
        logger.info("Creating visualizations...")
        visualizer = BrainActivationVisualizer(ethical_reporting=not args.no_ethics)
        
        if args.visualize:
            # Static visualizations
            viz_path = output_dir / 'wavelet_analysis_plots.png'
            analyzer.visualize_results(save_path=str(viz_path))
            logger.info(f"Static plots saved to {viz_path}")
            
            # Additional plots
            freq_path = output_dir / 'frequency_spectrum.png'
            visualizer.plot_frequency_spectrum(activation_results, save_path=str(freq_path))
            
            conn_path = output_dir / 'brain_connectivity.png'
            visualizer.plot_brain_connectivity(activation_results, save_path=str(conn_path))
            
            stat_path = output_dir / 'statistical_maps.png'
            visualizer.plot_statistical_maps(activation_results, save_path=str(stat_path))
            
            logger.info("Additional static visualizations saved")
        
        if args.interactive:
            # Interactive visualizations
            brain_3d_path = output_dir / 'interactive_brain_3d.html'
            brain_3d = visualizer.create_interactive_3d_brain(
                activation_results, save_path=str(brain_3d_path)
            )
            logger.info(f"Interactive 3D brain saved to {brain_3d_path}")
            
            dashboard_path = output_dir / 'analysis_dashboard.html'
            dashboard = visualizer.create_dashboard(
                activation_results, save_path=str(dashboard_path)
            )
            logger.info(f"Interactive dashboard saved to {dashboard_path}")
    
    # Final summary
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    
    if not args.quiet:
        print("\n" + "="*60)
        print("WAVELET ANALYSIS SUMMARY")
        print("="*60)
        print(f"Data: {time_series.shape[0]} voxels, {time_series.shape[1]} timepoints")
        print(f"Task onsets: {task_onsets}")
        print(f"Frequency bands analyzed: {len(activation_results['activation_maps'])}")
        print(f"Total activated regions: {sum(regions['n_clusters'] for regions in detected_regions.values())}")
        print(f"Output directory: {output_dir}")
        
        if not args.no_ethics:
            print("\nETHICAL NOTE:")
            print("This analysis emphasizes neural similarities and individual differences.")
            print("Results should be interpreted within cultural and educational contexts.")
        
        print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
