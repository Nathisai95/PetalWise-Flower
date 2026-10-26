from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "iris_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
ASSETS = ROOT / "assets"

st.set_page_config(page_title="PetalWise | Iris AI", page_icon="🌸", layout="wide")

st.markdown("""
<style>
:root { color-scheme: light; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
.hero { padding: 2rem 2.2rem; border-radius: 24px; background: linear-gradient(120deg,#111827 0%,#312e81 55%,#7c3aed 100%); color: white; margin-bottom: 1.2rem; }
.hero h1 { color: white; font-size: 2.6rem; margin: 0; letter-spacing: -1px; }
.hero p { color: #e0e7ff; font-size: 1.05rem; margin: .5rem 0 0 0; }
.metric-card { padding: 1.15rem; border: 1px solid #e5e7eb; border-radius: 18px; background: #fff; min-height: 115px; }
.species-card { border: 1px solid #e5e7eb; border-radius: 18px; overflow: hidden; background: white; }
.small-muted { color: #6b7280; font-size: .9rem; }
div[data-testid="stMetric"] { background: #f8fafc; padding: 1rem; border-radius: 14px; border: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model files are missing. Open the project terminal and run: python train.py")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

model = get_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
species = [
    {"name":"Iris setosa", "scientific":"Iris setosa", "file":"setosa.svg", "desc":"Often distinguished by shorter petals."},
    {"name":"Iris versicolor", "scientific":"Iris versicolor", "file":"versicolor.svg", "desc":"An intermediate-sized Iris species."},
    {"name":"Iris virginica", "scientific":"Iris virginica", "file":"virginica.svg", "desc":"Often has larger flower measurements."},
]

st.markdown("""
<div class="hero">
  <h1>🌸 PetalWise</h1>
  <p>Explore the Iris dataset with machine learning — clear predictions, model evaluation, and an interactive experience.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Selected model", metrics["selected_model"])
m2.metric("Held-out test accuracy", f'{metrics["test_accuracy"]:.1%}')
best_cv = metrics["cv_comparison"][metrics["selected_model"]]["cv_mean_accuracy"]
m3.metric("5-fold CV accuracy", f"{best_cv:.1%}")
st.caption("Metrics are computed when you run train.py. Accuracy is a measured result, not a guarantee of future performance.")

tab_predict, tab_species, tab_model = st.tabs(["🔮 Predict a flower", "🌷 Meet the species", "📊 Model performance"])

with tab_predict:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Flower measurements")
        st.write("Enter measurements in centimetres.")
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)
        predict = st.button("✨ Predict species", type="primary", use_container_width=True)
    with right:
        st.subheader("Prediction result")
        if predict:
            values = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred_idx = int(model.predict(values)[0])
            st.image(str(ASSETS / species[pred_idx]["file"]), use_container_width=True)
            st.success(f"Predicted species: **{species[pred_idx]['name']}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(values)[0]
                probability_df = pd.DataFrame({"Species":[x["name"] for x in species], "Model estimate":probs})
                st.bar_chart(probability_df.set_index("Species"))
                st.caption("These are model probability estimates, not a guarantee of correctness.")
        else:
            st.image(str(ASSETS / "setosa.svg"), use_container_width=True)
            st.info("Adjust the measurements and click **Predict species** to see the result.")
with tab_species:
    st.subheader("Three Iris species")
    cols = st.columns(3)
    for col, item in zip(cols, species):
        with col:
            st.image(str(ASSETS / item["file"]), use_container_width=True)
            st.markdown(f"**{item['name']}**")
            st.caption(item["desc"])
with tab_model:
    st.subheader("Compare candidate models")
    comp = metrics["cv_comparison"]
    df = pd.DataFrame([
        {"Model": name, "Mean 5-fold CV accuracy": vals["cv_mean_accuracy"], "CV standard deviation": vals["cv_std"]}
        for name, vals in comp.items()
    ]).sort_values("Mean 5-fold CV accuracy", ascending=False)
    st.dataframe(df.style.format({"Mean 5-fold CV accuracy":"{:.1%}", "CV standard deviation":"{:.1%}"}), use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Model")["Mean 5-fold CV accuracy"])
    st.markdown("**Confusion matrix (held-out test set)**")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual setosa","Actual versicolor","Actual virginica"], columns=["Predicted setosa","Predicted versicolor","Predicted virginica"])
    st.dataframe(cm, use_container_width=True)
    st.markdown("**Per-class evaluation**")
    report = metrics["classification_report"]
    rows = []
    for label in ["setosa", "versicolor", "virginica"]:
        rows.append({"Species": label.title(), "Precision": report[label]["precision"], "Recall": report[label]["recall"], "F1-score": report[label]["f1-score"], "Support": int(report[label]["support"])})
    st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1-score":"{:.2f}"}), use_container_width=True, hide_index=True)

st.divider()
st.caption("PetalWise is an educational ML demo using the classic Iris dataset. The flower illustrations are original illustrative artwork, not diagnostic botanical photographs.")
