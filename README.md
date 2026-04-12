# Heart Failure Prediction Model

This project builds a machine learning model to predict heart disease from clinical features and provides an interactive input form inside the notebook.

## Project Files

- `data.ipynb`: Full workflow (data loading, preprocessing, model training, evaluation, and interactive prediction form).
- `heart.csv`: Dataset used for training and testing.

## What This Notebook Does

1. Loads and inspects the dataset.
2. Preprocesses data using encoding and column alignment.
3. Splits data into train and test sets.
4. Trains a Logistic Regression model.
5. Evaluates performance using accuracy and classification report.
6. Provides an interactive form (widgets) to enter patient details and get prediction output.

## Requirements

Install these Python packages:

```bash
pip install pandas numpy scikit-learn ipywidgets
```

## How To Run

1. Open `data.ipynb` in VS Code or Jupyter.
2. Run all cells from top to bottom.
3. In the final form cell:
   - Enter patient values.
   - Click **Predict**.
   - View:
     - Predicted class (`Heart Disease: Yes/No`)
     - Prediction probability

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

- Keep `heart.csv` in the same folder as `data.ipynb`.
- If widgets do not display, make sure `ipywidgets` is installed in the active notebook kernel.
