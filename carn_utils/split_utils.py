import os
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import mirdata
import json

def stratified_split(train_df, split_col='label', test_size=0.2, seed=42):
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    for train_idx, val_idx in sss.split(train_df, train_df[split_col]):
        train_ids = train_df.iloc[train_idx]['track_id'].tolist()
        val_ids = train_df.iloc[val_idx]['track_id'].tolist()
        return train_ids, val_ids

def get_split_keys(keys, labels, test_size=0.2, seed=42):
    
    df = pd.DataFrame({'track_id': keys, 'label': labels})
    
    trainval_keys, test_keys = stratified_split(df, split_col='label', test_size=test_size, seed=seed)
    
    trainval_labels = df[df['track_id'].isin(trainval_keys)]['label'].tolist()
    train_df = pd.DataFrame({'track_id': trainval_keys, 'label': trainval_labels})
    
    train_keys, val_keys = stratified_split(train_df, split_col='label', test_size=test_size, seed=seed+1)
    
    return train_keys, val_keys, test_keys
    
    

def carn_split_keys(csv_path, train_fold, test_fold, split_col='Taala', test_size=0.2, seed=42, reorder=True):

    
    split_df = pd.read_csv(csv_path, dtype={'track_id': str})
    
    train_df = split_df[split_df['Fold'] == train_fold]
    test_df = split_df[split_df['Fold'] == test_fold]

    train_keys, val_keys = stratified_split(train_df, split_col=split_col, test_size=test_size, seed=seed)
    
    test_keys = test_df.sort_values([split_col, 'track_id'])['track_id'].tolist()
    
    if reorder:
        train_keys = reorder_by_taala_proportion(train_df, train_keys)

    return train_keys, val_keys, test_keys


def reorder_by_taala_proportion(df, train_keys):

    train_df = df[df['track_id'].isin(train_keys)].copy()

    # Group track_ids by Taala
    grouped = train_df.groupby('Taala')['track_id'].apply(list).to_dict()

    # Sort track_ids for reproducibility
    for taala in grouped:
        grouped[taala].sort()

    # Use Pandas to count examples per Taala
    taala_counts = train_df['Taala'].value_counts()
    total = taala_counts.sum()
    taala_ratios = (taala_counts / total).to_dict()

    # Initialize trackers
    used_counts = {taala: 0 for taala in grouped}
    quotas = {taala: 0.0 for taala in grouped}
    reordered_keys = []

    for _ in range(total):
        # Update quotas based on ratios
        for taala in quotas:
            quotas[taala] += taala_ratios[taala]

        # Choose the Taala with max quota and remaining samples
        available = [(quota, taala) for taala, quota in quotas.items() if used_counts[taala] < taala_counts[taala]]
        _, chosen_taala = max(available)

        # Append the next track_id
        index = used_counts[chosen_taala]
        reordered_keys.append(grouped[chosen_taala][index])
        used_counts[chosen_taala] += 1
        quotas[chosen_taala] -= 1

    return reordered_keys

def export_split_tsv(dataset_path, train_fold, seed=0, csv_path='cmr_splits.csv'):
    
    fold_df = pd.read_csv(csv_path, dtype={'track_id': str})
    
    if train_fold == 1:
        test_fold = 2
    else:
        test_fold = 1
    
    carn_train_keys, carn_val_keys, carn_test_keys = carn_split_keys(
                                                        csv_path=csv_path,
                                                        train_fold=train_fold,
                                                        test_fold=test_fold,
                                                        split_col='Taala',
                                                        seed=seed,
                                                        reorder=True
                                                        )   
    
    carn = mirdata.initialize('compmusic_carnatic_rhythm', 
                                version='full_dataset_1.0', 
                                data_home=dataset_path)
    carn.download(['index'])
    carn_tracks = carn.load_tracks() 
    
    split_dict = {}

    for track_id in carn_train_keys:
        src_audio_path = carn_tracks[track_id].audio_path
        filename = os.path.basename(src_audio_path)
        name_only = os.path.splitext(filename)[0]
        
        split_dict[name_only] = 'train'

    for track_id in carn_val_keys:
        src_audio_path = carn_tracks[track_id].audio_path
        filename = os.path.basename(src_audio_path)
        name_only = os.path.splitext(filename)[0]
        
        split_dict[name_only] = 'val'
    
    fold_df = pd.DataFrame.from_dict(split_dict, orient='index').reset_index()
    fold_df.columns = ['track', 'fold']
    
    # Dictionary mapping fold names to their audio folder paths
    available_folds = {
        "1": "data/annotations/cmr-fold1",
        "2": "data/annotations/cmr-fold2",
        # Add more folds here as needed
    }

    target_folder = available_folds[str(train_fold)]
    fold_df.to_csv(os.path.join(target_folder, "single.split"), sep='\t', index=False, header=False)

    '''    
    data = {"has_downbeats": True}
    json_path = os.path.join(target_folder, "info.json")

    # Write to JSON
    with open(json_path, "w") as f:
        json.dump(data, f)
    '''
    
        
    
        
