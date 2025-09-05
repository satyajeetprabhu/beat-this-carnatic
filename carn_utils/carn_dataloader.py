#!/usr/bin/env python3
"""
Script to create dataset splits for Carnatic rhythm dataset.
Converts the create_splits.ipynb notebook into a reusable function.
"""

import os
import shutil
import pandas as pd
import mirdata
import json


def copy_by_fold(dataset_path: str, splits_csv_path: str = 'cmr_splits.csv', 
                         output_dir: str = 'data', validate = False, verbose: bool = True) -> None:
    """
    Create dataset splits for Carnatic rhythm dataset by copying audio and annotation files
    into fold-specific directories.
    
    Args:
        dataset_path (str): Path to the Carnatic rhythm dataset
        splits_csv_path (str): Path to the CSV file containing fold assignments
        output_dir (str): Directory where the split data will be stored
        verbose (bool): Whether to print progress messages
    
    Returns:
        None
    """
    
    def print_msg(msg: str) -> None:
        """Helper function to print messages if verbose is True"""
        if verbose:
            print(msg)
    
    try:
        
        # Initialize the dataset
        print_msg("Initializing Carnatic rhythm dataset...")
        carn = mirdata.initialize('compmusic_carnatic_rhythm', 
                                version='full_dataset_1.0', 
                                data_home=dataset_path)
        
        # Download index if needed
        print_msg("Downloading dataset index...")
        carn.download(['index'])
        
        if validate:
            # Validate the dataset
            print_msg("Validating dataset...")
            carn.validate()
        
        # Load tracks
        print_msg("Loading tracks...")
        carn_tracks = carn.load_tracks()
        
        # Read splits CSV file
        splits_df = pd.read_csv(splits_csv_path, dtype={'track_id': str})
        
        # Create base data directory
        os.makedirs(output_dir, exist_ok=True)
        
        fold_audio_paths = []  # Initialize list to collect fold paths
        
        # Process each fold
        for fold_id, fold_name in zip([1, 2], ['cmr-fold1', 'cmr-fold2']):
            fold_audio_dir = os.path.join(output_dir, 'audio', fold_name)
            fold_mono_dir = os.path.join(output_dir, 'audio','mono_tracks', fold_name)
            fold_spec_dir = os.path.join(output_dir, 'audio', 'spectrograms', fold_name)
            fold_anno_dir = os.path.join(output_dir, 'annotations', fold_name, 'annotations', 'beats')
            
            # Create directories for this fold
            os.makedirs(fold_audio_dir, exist_ok=True)
            os.makedirs(fold_anno_dir, exist_ok=True)
            os.makedirs(fold_mono_dir, exist_ok=True)
            os.makedirs(fold_spec_dir, exist_ok=True)
            
            # Add to fold paths list
            fold_audio_paths.append([fold_name, fold_audio_dir])
            
            # Get tracks for this fold
            fold_tracks = splits_df[splits_df['Fold'] == fold_id]['track_id']
            
            # Copy audio files
            print_msg(f'Copying audio files for fold {fold_id}')
            for track_id in fold_tracks:
                if track_id in carn_tracks:
                    src_audio_path = carn_tracks[track_id].audio_path
                    filename = os.path.basename(src_audio_path)
                    dst_audio_path = os.path.join(fold_audio_dir, filename)
                    
                    if os.path.exists(src_audio_path):
                        try:
                            shutil.copy2(src_audio_path, dst_audio_path)
                        except Exception as e:
                            print_msg(f"Error copying {filename}: {e}")
                    else:
                        print_msg(f"Source file missing: {src_audio_path}")
                else:
                    print_msg(f"Track {track_id} not found in dataset")
                    
            # Copy annotation files
            print_msg(f'Copying annotation files for fold {fold_id}')
            for track_id in fold_tracks:
                if track_id in carn_tracks:
                    src_anno_path = carn_tracks[track_id].beats_path
                    filename = os.path.basename(src_anno_path)
                    dst_anno_path = os.path.join(fold_anno_dir, filename)
                    
                    if os.path.exists(src_anno_path):
                        try:
                            shutil.copy2(src_anno_path, dst_anno_path)
                        except Exception as e:
                            print_msg(f"Error copying {filename}: {e}")
                    else:
                        print_msg(f"Source file missing: {src_anno_path}")
                else:
                    print_msg(f"Track {track_id} not found in dataset")
        
            # Create info.json file for the fold
            print_msg(f'Creating info.json for fold {fold_id}')
            json_path = os.path.join(output_dir, 'annotations', fold_name, 'info.json')
            text = {"has_downbeats": True}
            with open(json_path, "w") as f:
                json.dump(text, f) 
        
        # Create TSV file with audio paths
        audio_paths_df = pd.DataFrame(fold_audio_paths, columns=['fold_name', 'audio_path'])
        tsv_path = os.path.join(output_dir, 'audio_paths.tsv')
        audio_paths_df.to_csv(tsv_path, sep='\t', index=False, header=False)
        print_msg(f"Created audio_paths.tsv file at {tsv_path}")
        
        print_msg("Dataset splits creation completed successfully!")
        

        
    except Exception as e:
        print_msg(f"Error during dataset splits creation: {e}")
        raise


def main():
    """Main function to run the dataset splits creation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create dataset splits for Carnatic rhythm dataset')
    parser.add_argument('dataset_path', help='Path to the Carnatic rhythm dataset')
    parser.add_argument('--splits-csv', default='cmr_splits.csv', 
                       help='Path to the CSV file containing fold assignments (default: cmr_splits.csv)')
    parser.add_argument('--output-dir', default='data', 
                       help='Directory where the split data will be stored (default: data)')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    copy_by_fold(
        dataset_path=args.dataset_path,
        splits_csv_path=args.splits_csv,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main() 