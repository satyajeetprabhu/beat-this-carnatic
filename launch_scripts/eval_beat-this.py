import os
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.getcwd()))

import mirdata
import mir_eval

def all_metrics(gt_times, pred_times):
    
    reference = mir_eval.beat.trim_beats(gt_times)
    estimated = mir_eval.beat.trim_beats(pred_times)
    
    # Compute the beat evaluation metrics
    scores = mir_eval.beat.evaluate(reference, estimated)
    
    return scores

def flatten_dict(track_id, beat_scores, downbeat_scores):
    flat_result = {'track_id': track_id}
    
    for k, v in beat_scores.items():
        flat_result[f'beat_{k}'] = v
    for k, v in downbeat_scores.items():
        flat_result[f'downbeat_{k}'] = v
    
    return flat_result

def read_beats_file(path):
    data = np.loadtxt(path)
    beats = data[:, 0].astype(np.float32)
    positions = data[:, 1].astype(int)
    downbeats = beats[positions == 1]
    return beats, downbeats


def main():
    parser = argparse.ArgumentParser(description='Evaluate Beat This predictions against CMR ground truth.')
    parser.add_argument('--data-home', required=True, help='Path to the dataset root')
    parser.add_argument('--mode', required=True, choices=['bl', 'ft'], help="Which prediction version subfolder to evaluate: 'bl' or 'ft'")
    args = parser.parse_args()

    data_home = args.data_home
    mode = args.mode

    # ----- Load Dataset -----
    carn = mirdata.initialize('compmusic_carnatic_rhythm', version='full_dataset_1.0', data_home=data_home)
    carn.download(['index'])
    carn_tracks = carn.load_tracks()
    carn_keys = list(carn_tracks.keys())

    # ----- Load Predictions -----
    pred_folder = 'output/predictions'
    path = os.path.join(ROOT, pred_folder, mode)

    # Scan prediction models
    models = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # ----- Evaluate Models -----
    print(f"Evaluating Beat This version - {mode}")
    for model in models:
        print(f"Evaluating model: {model}")

        files = [f for f in os.listdir(os.path.join(path, model)) if f.endswith('.beats')]
        files.sort(key=lambda x: x[:5])

        all_results = []
        total_tracks = len(files)

        for idx, file in enumerate(files, 1):
            track_id = file[:5]
            sys.stdout.write(f"\rProcessing {idx}/{total_tracks} tracks")

            if track_id not in carn_keys:
                print(f"Track {track_id} not found in dataset")
                continue

            beats_pred, downbeats_pred = read_beats_file(os.path.join(path, model, file))
            track = carn_tracks[track_id]

            beats_gt = track.beats.times.astype(np.float32)
            beat_pos = track.beats.positions.astype(int)
            downbeats_gt = beats_gt[beat_pos == 1]

            beat_scores = all_metrics(beats_gt, beats_pred)
            downbeat_scores = all_metrics(downbeats_gt, downbeats_pred)

            result = flatten_dict(track_id, beat_scores, downbeat_scores)
            all_results.append(result)

            sys.stdout.write(f"\rProcessing {idx}/{total_tracks} tracks")

        # Create main DataFrame
        df = pd.DataFrame(all_results)
        df['taala'] = df['track_id'].apply(lambda x: carn_tracks[x].taala if x in carn_tracks else '')
        df = df[['track_id', 'taala'] + [col for col in df.columns if col not in ['track_id', 'taala']]]

        # Compute averages
        avg_metrics = df.drop(columns=['track_id', 'taala']).mean()
        avg_metrics['track_id'] = 'average'
        avg_metrics['taala'] = ''

        taala_averages = []
        for taala, group in df.groupby('taala'):
            taala_avg = group.drop(columns=['track_id', 'taala']).mean()
            taala_avg['track_id'] = ''
            taala_avg['taala'] = taala
            taala_averages.append(taala_avg)

        taala_avg_df = pd.DataFrame(taala_averages)

        df_with_avg = pd.concat([df, pd.DataFrame([avg_metrics]), taala_avg_df], ignore_index=True)

        # Export results
        output_dir = os.path.join(ROOT, 'output', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'beat-this_{mode}_{model}.csv')
        df_with_avg.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()


