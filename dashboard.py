"""
Streamlit dashboard — grading model + doubt triage with routing.
Inference only. Models are trained and exported from the notebook (see models/).

Run:
    pip install streamlit joblib
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
st.set_page_config(page_title="LMS ML Pipeline", layout="centered")
st.title("LMS ML Pipeline — Demo")
with st.expander("⚠️ Limitations & Honest Notes (click to expand)"):
    st.markdown("""
- **Synthetic data**: Both models are trained on synthetic/proxy data — results demonstrate sound modeling practice, not real-world accuracy.
- **Urgency leakage**: The urgency label was partly derived from topic, so the urgency model learns topic priors more than genuine text signals. Topic-only leakage floor F1: **0.689** vs text model F1: **0.602**.
- **Conservative threshold**: Routing threshold of **0.90** (cost ratio 5:1) means ~94% of doubts go to teacher review by design — this is intentional, not a bug.
- **Grading inputs are synthetic proxies**: Slider values may not map intuitively to predictions since features were rank-scaled during training.
""")

grading_model = joblib.load(MODELS_DIR / "grading_model.pkl")
grading_features = joblib.load(MODELS_DIR / "grading_feature_names.pkl")
tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
urgency_model = joblib.load(MODELS_DIR / "urgency_model.pkl")  # calibrated
threshold = joblib.load(MODELS_DIR / "routing_config.pkl")["threshold"]

tab1, tab2 = st.tabs(["Grading Model", "Doubt Triage"])

with tab1:
    st.subheader("Predict Submission Quality")
    test_pass_rate = st.slider("Test pass rate", 0.0, 1.0, 0.75)
    cyclomatic_complexity = st.slider("Cyclomatic complexity", 1, 50, 10)
    lint_error_count = st.number_input("Lint error count", 0, 200, 5)
    execution_time_ms = st.number_input("Execution time (ms)", 0, 20000, 500)
    documentation_score = st.slider("Documentation score", 0.0, 1.0, 0.5)
    memory_usage_mb = st.number_input("Memory usage (MB)", 0, 4096, 128)

    QUALITY_LABELS = {0: "Needs Revision", 1: "Good Submission"}

    if st.button("Predict quality"):
        risk_score = cyclomatic_complexity * (1 + lint_error_count) / (test_pass_rate + 1e-3)
        row = dict(test_pass_rate=test_pass_rate, cyclomatic_complexity=cyclomatic_complexity,
                   lint_error_count=lint_error_count, execution_time_ms=execution_time_ms,
                   documentation_score=documentation_score, memory_usage_mb=memory_usage_mb,
                   risk_score=risk_score)
        X = [[row[f] for f in grading_features]]
        pred = grading_model.predict(X)[0]
        proba = grading_model.predict_proba(X)[0]
        label = QUALITY_LABELS.get(pred, str(pred))
        st.success(f"Predicted quality: **{label}**")
        st.bar_chart({"probability": proba}, x_label="class")

with tab2:
    st.subheader("Classify & Route a Doubt")

    SAMPLE_DOUBTS = {
        "-- Type my own --": "",
        "Urgent: assignment error": "I'm getting a critical error and my submission is failing right before the deadline, please help urgently.",
        "General code question": "Can someone explain how gradient descent updates the weights in each iteration?",
        "Casual follow-up": "Thanks for the last answer, just wanted to confirm if that approach also works for multiclass problems.",
    }
    choice = st.selectbox("Try a sample doubt (optional)", list(SAMPLE_DOUBTS.keys()))
    default_text = SAMPLE_DOUBTS[choice]
    text = st.text_area("Student doubt text", value=default_text, height=120)

    if st.button("Classify") and text.strip():
        X = tfidf.transform([text])
        proba = urgency_model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        pred = urgency_model.classes_[idx]
        confidence = float(proba[idx])
        decision = "✅ Auto-approve" if confidence >= threshold else "🧑‍🏫 Teacher review"

        st.write(f"**Predicted urgency:** {pred}")
        st.write(f"**Confidence:** {confidence:.2f} (threshold: {threshold:.2f})")
        st.markdown(f"### {decision}")