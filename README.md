# Loan Approval Prediction

This repository contains a small ML project that predicts whether a loan application will be approved. The analysis and experiments are available in the notebook `notebooks/Project_Notebook.ipynb`. The project includes reusable utilities, a training pipeline, saved model artifacts, and a Streamlit app for interactive inference.

## Streamlit App Link
[Loan Approval Predictor](https://loan-approval-status-predictor.streamlit.app/)

## Dataset

This project uses the Kaggle "Playground Series - Season 4, Episode 10" dataset. Please cite or attribute the dataset if you reuse this work:

- Kaggle competition / dataset: Playground Series - Season 4, Episode 10
- URL: [Kaggle Dataset for Loan Approval Status Prediction](https://www.kaggle.com/competitions/playground-series-s4e10/data)


## Project contents (high level)

- `notebooks/Project_Notebook.ipynb` - end-to-end exploratory analysis, feature engineering, model training, evaluation, and saving artifacts.
- `data/train.csv` - training dataset (expected to be the Kaggle Playground Series dataset).
- `src/utils.py` - helper functions for loading data, preprocessing (train & inference modes), model save/load, training and testing helpers.
- `src/training_pipeline.py` - script to run the training pipeline which trains several classifiers and saves trained models.
- `app/app.py` - Streamlit app that loads a trained RandomForest model and provides an interactive UI for single-row inference.
- `models/` - folder containing fitted transformers/encoders and trained model files created by the notebook or training pipeline.
- `reports/` - visualizations exported by the notebook (correlation matrix, feature plots, confusion matrices, etc.).

## Key implementation notes

- Feature engineering performed in the notebook (and mirrored in `src/utils.py`) includes:

  - Renaming `cb_person_default_on_file` -> `cb_defaulter_on_file`.
  - Ordinal encoding on `loan_grade`.
  - One-hot encoding on `loan_intent` and `person_home_ownership` (but many loan_intent dummies are dropped later).
  - Label encoding of the `cb_defaulter_on_file` target column.
  - Dropping lower-importance columns such as `person_age`, `cb_person_cred_hist_length`, and `person_home_ownership_OTHER`.
  - Quantile transformation (QuantileTransformer) is used as a scaler for numeric features before training.

## Saved artifacts created by the project

- `models/ordinal_encoder.joblib` - fitted OrdinalEncoder for `loan_grade`.
- `models/label_encoder.joblib` - fitted LabelEncoder for `cb_defaulter_on_file`.
- `models/quantile_transformer.joblib` - fitted QuantileTransformer used to transform features prior to model training.
- `models/trained_RFC.joblib` - trained RandomForestClassifier (used by the Streamlit app).
- `models/trained_HGBC.joblib` - trained HistGradientBoostingClassifier.
- `models/trained_KNC.joblib` - trained KNeighborsClassifier.

If these files are missing you can recreate them by running the notebook or `src/training_pipeline.py`.

## How to run

Prerequisites

- Python 3.8+ (project used typical data-science stack)
- Install requirements listed in `requirements.txt` (or create the conda environment using `conda-environment.yml` / `conda-requirements.txt` if present).

Install with pip:

```powershell
pip install -r requirements.txt
```

Run training (recreates encoders, transformer and trained models):

```powershell
python src/training_pipeline.py
```

This script will:

- Load `data/train.csv` using `src/utils.load_data`.
- Preprocess the data (splitting, encoding, saving encoders/transformer via `src/utils.save_model`).
- Train RandomForest, HistGradientBoosting and KNeighbors classifiers and save them to `models/`.

Run the Streamlit app for inference (requires `models/trained_RFC.joblib` to exist):

```powershell
streamlit run app/app.py
```

Open the URL shown by Streamlit in your browser (usually shown in the terminal). For example: `http://localhost:8501`

Notes on the Streamlit app

- The app uses `src/utils.preprocess_data(..., mode='inference')` to transform user-provided single-row inputs to the required numerical feature order and scale before calling the RandomForest model.
- The app expects the `models/` artifacts (ordinal encoder, label encoder, quantile transformer and the trained RandomForest) to be available under the repository `models/` directory.
- If the model files are not found the app will display an error asking you to run the training pipeline first.

## Using `src/utils.py` programmatically

Key functions (short contract):

- `load_data(data_path: str) -> pd.DataFrame` — loads CSV with `id` as index.
- `preprocess_data(df: pd.DataFrame, target: str = 'loan_status', mode: str = 'train')` —

  - mode='train': returns (x_train_tfd, y_train, x_val_tfd, y_val) and saves encoders/transformer to `models/`.
  - mode='inference': returns a scaled numpy array for a single-row or batch DataFrame ready for model.predict.
- `load_model(path: str)` / `save_model(model, path: str)` — wrappers around joblib with basic path checks.
- `train_model(model, x_train, y_train)` — fits the model and returns the fitted estimator.
- `test_model(model, x_test)` — returns predictions for x_test.

Edge cases & assumptions

- The helper `assert_path` in `src/utils.py` checks that file paths exist relative to the current working directory and that the path ends with `.csv` or `.joblib`. When saving models, ensure the `models/` folder exists and that working directory is the project root.
- `preprocess_data(..., mode='inference')` will add missing one-hot columns for `person_home_ownership` (RENT/MORTGAGE/OWN) if they are not present in supplied input and will reorder features to the expected sequence before applying the saved QuantileTransformer.

## Reproducing the notebook analysis

Open `notebooks/Project_Notebook.ipynb` and run the cells top-to-bottom. The notebook performs EDA (correlation heatmap, histplots and boxplots), feature selection using mutual information, trains several classifiers, evaluates them, and saves multiple artifacts into `models/` and figures into `reports/`.

## Troubleshooting

- If the Streamlit app shows "Model not found", ensure you have run the training pipeline and the `models/` directory contains `trained_RFC.joblib` and the encoder/transformer files.
- If `src/utils.assert_path` raises a ValueError when saving models, make sure the working directory is the repository root and `models/` exists.

## Author

Ajinkya Tamhankar

