# Heart Failure Prediction Model

This project builds a machine learning model to predict heart disease from clinical features and now includes a polished Streamlit interface for browser-based interaction.

## Project Files

- `notebooks/data.ipynb`: Full workflow (data loading, preprocessing, model training, evaluation, and interactive prediction form).
- `data/raw_heart.csv`: Dataset used for training and testing.
- `app.py`: Streamlit dashboard for predictions and dataset exploration.

## What This Notebook Does

1. Loads and inspects the dataset.
2. Preprocesses data using scaling and one-hot encoding.
3. Splits data into train and test sets.
4. Trains a Logistic Regression model.
5. Evaluates performance using accuracy, F1 score, ROC AUC, and a classification report.
6. Provides an interactive Streamlit form to enter patient details and get prediction output.

## Requirements

Install these Python packages:

```bash
pip install -r requirements.txt
```

## How To Run

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

Then open the browser URL Streamlit prints in the terminal.

## Input Categories (Full Forms)

### ChestPainType

- `ATA`: Atypical Angina
- `NAP`: Non-Anginal Pain
- `ASY`: Asymptomatic
- `TA`: Typical Angina

### RestingECG

- `Normal`: Normal ECG
- `ST`: ST-T Wave Abnormality
- `LVH`: Left Ventricular Hypertrophy

## Model Information

- Algorithm: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- Data split: 80% train / 20% test
- Prediction helper function aligns user input columns with training columns before inference.

## Notes

- Keep the CSV files in the `data/` folder.
- If the app does not start, make sure `streamlit` is installed in the active Python environment.
