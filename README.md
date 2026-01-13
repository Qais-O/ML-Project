# Speech Emotion Recognition — Experiments & Pipeline

## Overview

This repository contains two Jupyter notebooks used for building a speech emotion recognition (SER) pipeline:

- `experiments.ipynb` — feature extraction, dataset scanning, utilities, multiple model architectures (CNN, CRNN, CRNN+GRU), and training/evaluation pipelines.
- `pipeline.ipynb` — reusable processing utilities and the final "Strategy C: Regularized GRU + Robust SpecAugment" training and evaluation results.

The notebooks create and consume a unified feature HDF5 file (`unified_audio_features.h5`) and output model checkpoints, metrics CSVs, and plots.

---

## Quick Start 🔧

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # see suggestions below
```

2. (macOS / Apple Silicon note) For best performance on M1/M2, follow TensorFlow macOS install instructions (e.g., `tensorflow-macos` + `tensorflow-metal`) or use Conda.

3. Open the notebooks in Jupyter or Colab and run cells in order. Verify and adjust dataset paths in the configuration cells before running.

---

## Requirements (suggested)

- Python 3.8+
- librosa, numpy, pandas, h5py, tqdm
- tensorflow (or tensorflow-macos + tensorflow-metal on macOS M1/M2)
- scikit-learn, matplotlib, seaborn
- noisereduce, ffmpeg (system dependency for some audio ops)

Example (minimal):

```text
librosa
numpy
pandas
h5py
tqdm
tensorflow
scikit-learn
matplotlib
seaborn
noisereduce
ffmpeg-python

# Optional (macOS):
# tensorflow-macos
# tensorflow-metal
```

---

## Notebooks & Key Functions

### `experiments.ipynb`
- Feature extraction pipeline using MFCCs (`extract_mfcc_features`) and dataset scanning across folder-based datasets and CSVs.
- `create_unified_feature_file()` builds `unified_audio_features.h5` and a `features_summary.csv` summary.
- Model training pipelines included:
  - Basic CNN (`create_basic_cnn_model`, `run_training_pipeline`) — M1/M2-friendly variants
  - CRNN (CNN + LSTM) (`create_crnn_model`, `train_on_real_data`)
  - Continuation/evaluation utilities (`continue_evaluation_only`, `continue_evaluation_with_metrics`)
- Hyperparameters and reproducible training configuration are centralized in `HYPERPARAMS`.

### `pipeline.ipynb`
- Reusable audio processing utilities and a `quick_bulk_process` helper for batch normalization, noise reduction, and silence removal.
- Finalized Strategy C: **Regularized GRU + Robust SpecAugment** implementation and evaluation (see last script / results in this notebook).
- Includes a safe `spec_augment` implementation that guards against mask sizes larger than spectrogram dimensions.

---

## Outputs (examples) 💾
- `unified_audio_features.h5` — HDF5 features dataset
- `features_summary.csv` — index for features
- Model checkpoints: `basic_cnn_best_model.keras`, `crnn_best_model.keras`, `crnn_best_model_regularized.keras`
- Predictions/metrics CSVs: `basic_cnn_predictions.csv`, `crnn_predictions.csv`, `crnn_regularized_predictions.csv`
- Training logs & plots: `training_history_cnn.csv`, `cnn_training_history.png`, `crnn_training_plot.png`, `training_history_regularized.png`, `confusion_matrix_regularized.png`

---

## Usage Tips & Notes
- Check dataset paths: update `DATASET_PATHS` and `CSV_DATASETS` (mappings) before running scanning functions.
- Small spectrogram dimensions can cause SpecAugment masking issues; the notebooks include safety checks to avoid this.
- For macOS M1/M2, some architectures were adjusted (e.g., BatchNormalization removed in a CNN variant) to avoid shape incompatibility issues.
- If CSV-based datasets contain remote/relative prefixes, update the `replacements` mapping in `CSV_DATASETS` so `scan_csv_files` resolves local paths correctly.

---

## Reproducibility & Tips for Running Experiments
- Use a fixed random seed where applicable and the same `random_state` when splitting data.
- Training callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) are configured in the notebooks — tune patience and learning rate schedule as needed.

---

## Contact & License
- Authors: Team 3 (Qais, Bashar and Mahmmoud)
- License: MIT
