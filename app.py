from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw_heart.csv"

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [column.strip() for column in df.columns]
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df["HeartDisease"].astype(int)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["No Heart Disease", "Heart Disease"]),
    }

    return model, metrics


def make_prediction_frame(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs], columns=ALL_FEATURES)


def section_card(title: str, value: str, delta: str | None = None) -> None:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    df = load_data()
    model, metrics = train_model(df)

    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 99, 132, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(0, 200, 150, 0.14), transparent 24%),
                    linear-gradient(180deg, #081120 0%, #0e1729 45%, #101b33 100%);
                color: #f4f7fb;
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            [data-testid="stSidebar"] {
                background: rgba(7, 12, 22, 0.9);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            .hero {
                padding: 1.4rem 1.4rem 1.1rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(14px);
            }

            .eyebrow {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.2em;
                color: #8fd3ff;
                margin-bottom: 0.4rem;
            }

            .title {
                font-size: 2.4rem;
                font-weight: 800;
                line-height: 1.05;
                margin: 0;
                color: #ffffff;
            }

            .subtitle {
                max-width: 60rem;
                margin-top: 0.7rem;
                color: rgba(233, 240, 255, 0.78);
                font-size: 1rem;
            }

            .metric-card {
                border-radius: 20px;
                padding: 1rem 1.1rem;
                border: 1px solid rgba(255, 255, 255, 0.09);
                background: rgba(14, 23, 41, 0.78);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }

            .metric-label {
                color: rgba(244, 247, 251, 0.68);
                font-size: 0.84rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }

            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                margin-top: 0.35rem;
                color: #ffffff;
            }

            .delta {
                margin-top: 0.25rem;
                color: #9be5b0;
                font-size: 0.82rem;
            }

            .panel {
                border-radius: 22px;
                padding: 1.2rem;
                background: rgba(255, 255, 255, 0.045);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
            }

            .stDataFrame, .stTable {
                border-radius: 16px;
                overflow: hidden;
            }

            div[data-testid="stForm"] {
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255, 255, 255, 0.03);
                padding: 1rem;
                border-radius: 18px;
            }

            .stButton > button {
                background: linear-gradient(135deg, #8fd3ff, #6ee7b7);
                color: #07101b;
                border: none;
                padding: 0.75rem 1rem;
                border-radius: 14px;
                font-weight: 700;
            }

            .stButton > button:hover {
                filter: brightness(1.03);
                transform: translateY(-1px);
                transition: 180ms ease;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Heart Failure Prediction Studio</div>
            <div class="title">Clinical signals, polished into a fast prediction dashboard.</div>
            <div class="subtitle">
                Train a lightweight classifier from the bundled dataset, explore the data, and predict risk from a clean form UI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_patients = len(df)
    positive_rate = df["HeartDisease"].mean() * 100
    section_cols = st.columns(4)
    with section_cols[0]:
        section_card("Patients", f"{total_patients:,}")
    with section_cols[1]:
        section_card("Positive Rate", f"{positive_rate:.1f}%")
    with section_cols[2]:
        section_card("Accuracy", f"{metrics['accuracy']:.3f}")
    with section_cols[3]:
        section_card("ROC AUC", f"{metrics['roc_auc']:.3f}")

    st.write("")
    tab_predictor, tab_dataset, tab_model = st.tabs(["Predictor", "Dataset", "Model"])

    with tab_predictor:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Patient profile")
            st.caption("Adjust the clinical values and generate a prediction instantly.")

            with st.form("prediction_form", clear_on_submit=False):
                default_row = df[ALL_FEATURES].median(numeric_only=True).to_dict()
                mode_row = df[CATEGORICAL_FEATURES].mode(dropna=True).iloc[0].to_dict()

                col_a, col_b = st.columns(2)
                with col_a:
                    age = st.slider("Age", 18, 100, int(default_row["Age"]))
                    sex = st.selectbox("Sex", ["M", "F"], index=0 if mode_row["Sex"] == "M" else 1)
                    chest_pain = st.selectbox(
                        "Chest Pain Type",
                        ["ATA", "NAP", "ASY", "TA"],
                        index=["ATA", "NAP", "ASY", "TA"].index(mode_row["ChestPainType"])
                        if mode_row["ChestPainType"] in ["ATA", "NAP", "ASY", "TA"]
                        else 0,
                    )
                    resting_bp = st.slider("Resting BP", 70, 220, int(default_row["RestingBP"]))
                    cholesterol = st.slider("Cholesterol", 0, 700, int(default_row["Cholesterol"]))

                with col_b:
                    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=int(default_row["FastingBS"]))
                    resting_ecg = st.selectbox(
                        "Resting ECG",
                        ["Normal", "ST", "LVH"],
                        index=["Normal", "ST", "LVH"].index(mode_row["RestingECG"])
                        if mode_row["RestingECG"] in ["Normal", "ST", "LVH"]
                        else 0,
                    )
                    max_hr = st.slider("Max Heart Rate", 60, 220, int(default_row["MaxHR"]))
                    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"], index=0 if mode_row["ExerciseAngina"] == "N" else 1)
                    oldpeak = st.slider("Oldpeak", 0.0, 6.5, float(default_row["Oldpeak"]), 0.1)
                    st_slope = st.selectbox(
                        "ST Slope",
                        ["Down", "Flat", "Up"],
                        index=["Down", "Flat", "Up"].index(mode_row["ST_Slope"])
                        if mode_row["ST_Slope"] in ["Down", "Flat", "Up"]
                        else 2,
                    )

                submitted = st.form_submit_button("Predict risk")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Prediction output")

            if submitted:
                payload = make_prediction_frame(
                    {
                        "Age": age,
                        "RestingBP": resting_bp,
                        "Cholesterol": cholesterol,
                        "FastingBS": fasting_bs,
                        "MaxHR": max_hr,
                        "Oldpeak": oldpeak,
                        "Sex": sex,
                        "ChestPainType": chest_pain,
                        "RestingECG": resting_ecg,
                        "ExerciseAngina": exercise_angina,
                        "ST_Slope": st_slope,
                    }
                )
                probability = float(model.predict_proba(payload)[0, 1])
                prediction = int(probability >= 0.5)
                risk_text = "High risk" if prediction == 1 else "Lower risk"
                risk_color = "#ff7b7b" if prediction == 1 else "#7fe7b7"

                st.markdown(
                    f"""
                    <div style="padding: 1rem 1.1rem; border-radius: 18px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);">
                        <div style="color: rgba(244,247,251,0.7); text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem;">Predicted class</div>
                        <div style="font-size: 2rem; font-weight: 800; color: {risk_color}; margin-top: 0.3rem;">{risk_text}</div>
                        <div style="color: rgba(244,247,251,0.86); margin-top: 0.45rem;">Estimated probability of heart disease: <strong>{probability:.1%}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(probability)
                st.write("")

                summary = pd.DataFrame(
                    {
                        "Feature": ["Age", "Resting BP", "Cholesterol", "Max HR", "Oldpeak", "Fasting BS"],
                        "Value": [age, resting_bp, cholesterol, max_hr, oldpeak, fasting_bs],
                    }
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.info("Fill the form on the left and press Predict risk to see the result here.")

            st.markdown("</div>", unsafe_allow_html=True)

    with tab_dataset:
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Target balance")
            target_counts = df["HeartDisease"].value_counts().sort_index()
            target_counts.index = ["No disease", "Disease"]
            st.bar_chart(target_counts)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Feature snapshot")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_model:
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Evaluation summary")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
            metric_cols[1].metric("F1 score", f"{metrics['f1']:.3f}")
            metric_cols[2].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
            st.text_area("Classification report", metrics["report"], height=220)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Confusion matrix")
            cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm, use_container_width=True)
            st.caption("The model is trained directly from the bundled CSV using a scaled + one-hot encoded logistic regression pipeline.")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
