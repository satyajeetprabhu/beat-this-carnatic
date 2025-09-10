# Beat This! fine-tuned on the Carnatic Music Rhythm Dataset

This repository is a companion to the Master's thesis "Revisiting Meter Tracking in Carnatic Music using Deep Learning Approaches," submitted towards a Master in Sound and Music Computing at Universitat Pompeu Fabra (August 2025). 

It includes the implementation of [Beat This!](https://github.com/CPJKU/beat_this) beat and downbeat tracker modified for fine-tuning, specifically on the Carnatic Music Rhythm Dataset (CMR). The notebooks provided can be used to reproduce the study's Beat This baseline (BeatThis-BL) and fine-tuning (BeatThis-FT) results; the evaluation results reported in the study are provided under `output/results/`.


## Installation
This repository has been only tested for use on Google Colab through the provided notebooks. For local installation and detailed documentation, refer to the original [Beat This!](https://github.com/CPJKU/beat_this) repository.

## Dataset Setup

1. **Download the Dataset**: Request access to the [CompMusic Carnatic Rhythm Dataset (CMR)](https://zenodo.org/records/1264394#.WyeLDByxXMU)

2. **Folder Structure**: 

Place the entire `CMR_full_dataset_1.0` folder inside `Datasets/CMR/` in google drive. Additionally, create a `Beat_This_CMR/` folder in your drive, if not running all the notebooks in order.
   ```
   your-drive/
   ├── Datasets/
   │   └── CMR/
   │       └── CMR_full_dataset_1.0/     # Place the entire dataset folder here
   └── Beat_This_CMR/                    # Create manually or will be created by notebooks if run in order below
   ```
   
   Update the dataset path in notebooks accordingly. Additionally, create a `Beat_This_CMR/` folder in your drive if not running all the notebooks in order.

## Notebooks

To reproduce the results of the study, we provide four notebooks in the `notebooks/`. Each contains a clickable link to run the notebook in Google Colab.  
Run the notebooks in the following order:

1. **`Preprocess.ipynb`**: Creates data splits, preprocesses audio, and organizes data for training
2. **`Beat_This_FT.ipynb`**: Fine-tunes Beat This! model with different seeds and cross-validation folds as described in the study
3. **`Eval_FT.ipynb`**: Evaluates fine-tuned models and exports results to `output/results/`
4. **`Eval_BL.ipynb`**: Evaluates base Beat This! model on CMR dataset

## Weights & Biases (wandb)

This project supports online logging using wandb. To enable online logging:

1. Create a free account at https://wandb.ai
2. Find your API key in Settings → API key
3. Update the necessary field in the notebook

## Results

Evaluation results are automatically saved to `output/results/` as CSV files. The results include:
- Baseline model performance across different base model checkpoints (final0, final1 and final2)
- Fine-tuned model performance for each training fold and seed combination
