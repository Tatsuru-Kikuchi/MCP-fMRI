#!/usr/bin/env python3
"""
MCP-fMRI Data Preprocessing Pipeline
Standardized preprocessing for Japanese mathematical cognition study
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class fMRIPreprocessor:
    """Preprocessing pipeline for fMRI data."""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        """Initialize preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw fMRI data
            output_dir: Directory for preprocessed output
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Preprocessing parameters
        self.params = {
            'motion_threshold': 3.0,  # mm
            'rotation_threshold': 3.0,  # degrees
            'smoothing_fwhm': 8.0,  # mm
            'high_pass_cutoff': 128.0,  # seconds
            'tr': 2.0,  # repetition time in seconds
            'slice_timing_ref': 0.5  # reference slice (0.5 = middle)
        }
        
        self.quality_metrics = {}
        
    def load_participant_data(self, participant_id: str) -> Dict:
        """Load data for a single participant.
        
        Args:
            participant_id: Unique participant identifier
            
        Returns:
            Dictionary containing participant data
        """
        participant_dir = self.raw_data_dir / participant_id
        
        if not participant_dir.exists():
            raise FileNotFoundError(f"No data found for participant {participant_id}")
        
        # Load functional data
        func_files = list(participant_dir.glob("*_task-math_*.nii.gz"))
        if not func_files:
            raise FileNotFoundError(f"No functional data found for {participant_id}")
        
        # Load anatomical data
        anat_files = list(participant_dir.glob("*_T1w.nii.gz"))
        if not anat_files:
            logger.warning(f"No anatomical data found for {participant_id}")
            anat_files = [None]
        
        return {
            'participant_id': participant_id,
            'functional': func_files[0],
            'anatomical': anat_files[0],
            'output_dir': self.output_dir / participant_id
        }
    
    def motion_correction(self, func_file: Path, output_dir: Path) -> Tuple[Path, Dict]:
        """Perform motion correction on functional data.
        
        Args:
            func_file: Path to functional data
            output_dir: Output directory
            
        Returns:
            Tuple of (corrected_file_path, motion_parameters)
        """
        logger.info(f"Performing motion correction for {func_file.name}")
        
        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Simulate motion correction (in real implementation, use SPM/FSL/AFNI)
        corrected_file = output_dir / f"mc_{func_file.name}"
        
        # Simulate motion parameters
        # In real implementation, extract from motion correction output
        n_volumes = 200  # Typical number of volumes
        motion_params = {
            'translation_x': np.random.normal(0, 0.5, n_volumes),
            'translation_y': np.random.normal(0, 0.5, n_volumes),
            'translation_z': np.random.normal(0, 0.5, n_volumes),
            'rotation_x': np.random.normal(0, 0.1, n_volumes),
            'rotation_y': np.random.normal(0, 0.1, n_volumes),
            'rotation_z': np.random.normal(0, 0.1, n_volumes)
        }
        
        # Calculate motion summary statistics
        max_translation = max([
            np.max(np.abs(motion_params['translation_x'])),
            np.max(np.abs(motion_params['translation_y'])),
            np.max(np.abs(motion_params['translation_z']))
        ])
        
        max_rotation = max([
            np.max(np.abs(motion_params['rotation_x'])),
            np.max(np.abs(motion_params['rotation_y'])),
            np.max(np.abs(motion_params['rotation_z']))
        ]) * 180 / np.pi  # Convert to degrees
        
        motion_summary = {
            'max_translation_mm': max_translation,
            'max_rotation_deg': max_rotation,
            'mean_fd': np.mean(np.sqrt(np.sum([
                np.diff(motion_params['translation_x'])**2,
                np.diff(motion_params['translation_y'])**2,
                np.diff(motion_params['translation_z'])**2
            ], axis=0))),
            'passes_qc': max_translation < self.params['motion_threshold'] and 
                        max_rotation < self.params['rotation_threshold']
        }
        
        # Save motion parameters
        motion_df = pd.DataFrame(motion_params)
        motion_df.to_csv(output_dir / 'motion_parameters.csv', index=False)
        
        return corrected_file, motion_summary
    
    def slice_timing_correction(self, func_file: Path, output_dir: Path) -> Path:
        """Perform slice timing correction.
        
        Args:
            func_file: Motion-corrected functional data
            output_dir: Output directory
            
        Returns:
            Path to slice-time corrected file
        """
        logger.info(f"Performing slice timing correction for {func_file.name}")
        
        # In real implementation, use neuroimaging tools
        corrected_file = output_dir / f"st_{func_file.name}"
        
        return corrected_file
    
    def spatial_normalization(self, func_file: Path, anat_file: Path, output_dir: Path) -> Path:
        """Normalize to standard space (MNI).
        
        Args:
            func_file: Slice-time corrected functional data
            anat_file: Anatomical reference image
            output_dir: Output directory
            
        Returns:
            Path to normalized file
        """
        logger.info(f"Performing spatial normalization for {func_file.name}")
        
        # In real implementation, use registration tools
        normalized_file = output_dir / f"norm_{func_file.name}"
        
        return normalized_file
    
    def spatial_smoothing(self, func_file: Path, output_dir: Path) -> Path:
        """Apply spatial smoothing.
        
        Args:
            func_file: Normalized functional data
            output_dir: Output directory
            
        Returns:
            Path to smoothed file
        """
        logger.info(f"Applying spatial smoothing ({self.params['smoothing_fwhm']}mm FWHM)")
        
        # In real implementation, apply Gaussian smoothing
        smoothed_file = output_dir / f"smooth_{func_file.name}"
        
        return smoothed_file
    
    def temporal_filtering(self, func_file: Path, output_dir: Path) -> Path:
        """Apply temporal filtering.
        
        Args:
            func_file: Smoothed functional data
            output_dir: Output directory
            
        Returns:
            Path to filtered file
        """
        logger.info(f"Applying temporal filtering (high-pass: {self.params['high_pass_cutoff']}s)")
        
        # In real implementation, apply high-pass filter
        filtered_file = output_dir / f"filtered_{func_file.name}"
        
        return filtered_file
    
    def calculate_quality_metrics(self, func_file: Path, motion_summary: Dict) -> Dict:
        """Calculate data quality metrics.
        
        Args:
            func_file: Preprocessed functional data
            motion_summary: Motion correction summary
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Calculating quality metrics")
        
        # Simulate quality metrics
        # In real implementation, calculate from actual data
        quality_metrics = {
            'snr': np.random.normal(120, 20),  # Signal-to-noise ratio
            'tsnr': np.random.normal(80, 15),  # Temporal SNR
            'mean_fd': motion_summary['mean_fd'],
            'max_translation': motion_summary['max_translation_mm'],
            'max_rotation': motion_summary['max_rotation_deg'],
            'passes_motion_qc': motion_summary['passes_qc'],
            'ghost_to_signal_ratio': np.random.uniform(0.01, 0.05),
            'temporal_std': np.random.normal(2.5, 0.5)
        }
        
        # Overall quality assessment
        quality_metrics['overall_quality'] = (
            quality_metrics['snr'] > 100 and
            quality_metrics['tsnr'] > 50 and
            quality_metrics['passes_motion_qc'] and
            quality_metrics['ghost_to_signal_ratio'] < 0.1
        )
        
        return quality_metrics
    
    def preprocess_participant(self, participant_id: str) -> Dict:
        """Run complete preprocessing pipeline for one participant.
        
        Args:
            participant_id: Unique participant identifier
            
        Returns:
            Dictionary containing preprocessing results
        """
        logger.info(f"Starting preprocessing for participant {participant_id}")
        
        try:
            # Load participant data
            data = self.load_participant_data(participant_id)
            output_dir = data['output_dir']
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Preprocessing pipeline
            # 1. Motion correction
            mc_file, motion_summary = self.motion_correction(
                data['functional'], output_dir
            )
            
            # 2. Slice timing correction
            st_file = self.slice_timing_correction(mc_file, output_dir)
            
            # 3. Spatial normalization
            norm_file = self.spatial_normalization(
                st_file, data['anatomical'], output_dir
            )
            
            # 4. Spatial smoothing
            smooth_file = self.spatial_smoothing(norm_file, output_dir)
            
            # 5. Temporal filtering
            final_file = self.temporal_filtering(smooth_file, output_dir)
            
            # 6. Quality assessment
            quality_metrics = self.calculate_quality_metrics(final_file, motion_summary)
            
            # Save quality metrics
            quality_df = pd.DataFrame([quality_metrics])
            quality_df.to_csv(output_dir / 'quality_metrics.csv', index=False)
            
            # Store results
            self.quality_metrics[participant_id] = quality_metrics
            
            result = {
                'participant_id': participant_id,
                'status': 'success',
                'final_file': final_file,
                'quality_metrics': quality_metrics,
                'passes_qc': quality_metrics['overall_quality']
            }
            
            logger.info(f"Preprocessing completed for {participant_id} - "
                       f"QC: {'PASS' if result['passes_qc'] else 'FAIL'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {participant_id}: {str(e)}")
            return {
                'participant_id': participant_id,
                'status': 'failed',
                'error': str(e),
                'passes_qc': False
            }
    
    def preprocess_batch(self, participant_list: List[str]) -> pd.DataFrame:
        """Preprocess multiple participants.
        
        Args:
            participant_list: List of participant IDs
            
        Returns:
            DataFrame with preprocessing results
        """
        logger.info(f"Starting batch preprocessing for {len(participant_list)} participants")
        
        results = []
        for participant_id in participant_list:
            result = self.preprocess_participant(participant_id)
            results.append(result)
        
        # Create summary DataFrame
        results_df = pd.DataFrame(results)
        
        # Save batch summary
        results_df.to_csv(self.output_dir / 'preprocessing_summary.csv', index=False)
        
        # Print summary statistics
        n_success = len(results_df[results_df['status'] == 'success'])
        n_pass_qc = len(results_df[results_df['passes_qc'] == True])
        
        logger.info(f"Batch preprocessing complete:")
        logger.info(f"  Successful: {n_success}/{len(participant_list)}")
        logger.info(f"  Passed QC: {n_pass_qc}/{len(participant_list)}")
        
        return results_df
    
    def generate_qc_report(self) -> None:
        """Generate quality control report."""
        if not self.quality_metrics:
            logger.warning("No quality metrics available for report")
            return
        
        # Compile all quality metrics
        qc_df = pd.DataFrame.from_dict(self.quality_metrics, orient='index')
        
        # Calculate summary statistics
        summary_stats = {
            'total_participants': len(qc_df),
            'passed_qc': len(qc_df[qc_df['overall_quality'] == True]),
            'mean_snr': qc_df['snr'].mean(),
            'mean_tsnr': qc_df['tsnr'].mean(),
            'mean_motion': qc_df['mean_fd'].mean(),
            'max_motion_exceeded': len(qc_df[qc_df['passes_motion_qc'] == False])
        }
        
        # Save detailed QC report
        qc_df.to_csv(self.output_dir / 'quality_control_detailed.csv')
        
        # Save summary report
        with open(self.output_dir / 'quality_control_summary.txt', 'w') as f:
            f.write("MCP-fMRI Preprocessing Quality Control Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total participants processed: {summary_stats['total_participants']}\n")
            f.write(f"Passed quality control: {summary_stats['passed_qc']}\n")
            f.write(f"QC pass rate: {summary_stats['passed_qc']/summary_stats['total_participants']*100:.1f}%\n\n")
            f.write(f"Mean SNR: {summary_stats['mean_snr']:.1f}\n")
            f.write(f"Mean temporal SNR: {summary_stats['mean_tsnr']:.1f}\n")
            f.write(f"Mean motion (FD): {summary_stats['mean_motion']:.3f} mm\n")
            f.write(f"Participants exceeding motion threshold: {summary_stats['max_motion_exceeded']}\n")
        
        logger.info("Quality control report generated")

def main():
    """Main preprocessing pipeline."""
    # Configuration
    raw_data_dir = "../data/raw_fmri"
    output_dir = "../data/preprocessed"
    
    # Generate participant list (JP001 to JP156)
    participant_list = [f"JP{i:03d}" for i in range(1, 157)]
    
    # Initialize preprocessor
    preprocessor = fMRIPreprocessor(raw_data_dir, output_dir)
    
    # Run batch preprocessing
    results = preprocessor.preprocess_batch(participant_list)
    
    # Generate QC report
    preprocessor.generate_qc_report()
    
    print("\nPreprocessing pipeline completed!")
    print(f"Check {output_dir} for results")

if __name__ == "__main__":
    main()