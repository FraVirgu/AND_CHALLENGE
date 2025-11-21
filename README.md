# AND_CHALLENGE

A compact PyTorch project for modeling and predicting pain levels from time-series sensor data ("pirate pain" dataset). The repo includes data preprocessing, sequence construction, model training, evaluation, and inference utilities.

**Key Features**

- Sliding-window sequence builder for time-series samples.
- Multi-input model support (numeric features + per-time categorical inputs like pain surveys and limb counts).
- Class balancing with weighted sampling and loss weights.
- Training scripts and experiment files with saved models in `models/` and results in `results/`.

**Prerequisites**

- Python 3.8+ recommended
- GPU recommended for model training (PyTorch)

Suggested Python packages:

- pandas, numpy, torch, scikit-learn, seaborn, matplotlib

Install example:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy torch scikit-learn seaborn matplotlib
```

(If you prefer, create a `requirements.txt` with these packages.)

**Project Layout**

- `data_import.py`: Loads CSV files from `DATA/`, preprocesses features, builds sliding-window sequences, creates PyTorch `TensorDataset`s and `DataLoader`s, and sets up class-balancing utilities.
- `model.py`: Model definitions (GRU / attention / multi-input architectures).
- `model_train.py`: Training loop and checkpoint saving (uses datasets produced by `data_import.py`).
- `prediction.py`: Inference / prediction utilities and generation of submission CSVs.
- `main.py`: Entry point (if applicable) to run experiments or orchestrate training/inference.
- `experiment_*.py`: Saved experiment scripts and configurations.
- `models/`: Directory where trained model checkpoints are stored.
- `results/`: Experiment results and outputs.
- `logs/`: Training logs (e.g., for TensorBoard or plain logging).
- `DATA/`: Expected input CSV files:
  - `pirate_pain_train.csv`
  - `pirate_pain_train_labels.csv`
  - `pirate_pain_test.csv`

**Data & Preprocessing**

- `data_import.py` reads the CSVs (no header) and converts the first row to column names.
- Categorical mappings convert `n_legs`, `n_hands`, `n_eyes` to numeric codes.
- A robust normalization (median/IQR-ish via mean/std fallback) is applied to continuous features.
- Sliding-window sequence builder settings:
  - `WINDOW_SIZE` (default `40`)
  - `STRIDE` (default `20`)
  - `BATCH_SIZE` (default `32`)
  - `LEARNING_RATE` (default `1e-3`)

Note: `PERCENTAGE_SPLIT` in `data_import.py` controls train/val split. By default it is set to `1` in the current file (which places all samples into training). Adjust it to a value < 1.0 (e.g., `0.8`) to create a validation set.

**Quick Start — Training**

1. Ensure `DATA/` contains the three CSVs described above.
2. Create and activate a virtual environment (see install steps).
3. Run training (example):

```
python model_train.py
```

- `model_train.py` imports data and uses the `TensorDataset`s / `DataLoader`s created in `data_import.py`.
- Check `models/` for saved checkpoints (filenames include hyperparameters) and `logs/` for logs.

If you prefer to run a single orchestrator (if present):

```
python main.py
```

**Quick Start — Inference / Prediction**
To run prediction and generate submission CSVs (example):

```
python prediction.py
```

Generated submissions in the repo (examples): `submission_attention_kfold_ensemble.csv`, `submission_attention_multiinput.csv`, `submission_gru_model.csv`.

**Hyperparameters & Configurations**

- Many hyperparameters are stored or set inside the experiment scripts (`experiment_*.py`) and `data_import.py` constants.
- Typical parameters to tune: window size, stride, batch size, learning rate, dropout, GRU hidden size, class-balancing strategy.

**Outputs**

- Trained model checkpoints: `models/` (named with hyperparameter summaries).
- Experiment outputs and plots: `results/<exp_name>/`.
- Submission CSVs in repository root (examples already present).

**Development Tips**

- To change preprocessing behavior, update `data_import.py` (e.g., `PERCENTAGE_SPLIT`, feature scaling, outlier clipping).
- To add a new model, edit or add a file in `model.py` and update training in `model_train.py` or a new experiment script.
- Use `torch.cuda.is_available()` in `data_import.py` / training scripts to leverage GPU.

**Notes & Known Caveats**

- `data_import.py` currently sets `PERCENTAGE_SPLIT = 1`, so adjust it for validation testing.
- Ensure numeric columns are present and not dropped by CSV parsing — the parser expects first row to be the header.

**Contact & Next Steps**
If you want, I can:

- Add a `requirements.txt` or `environment.yml`.
- Add example commands for distributed / GPU training.
- Create a short `run.sh` convenience script.

---

License: Add a license file if desired (e.g., MIT).
