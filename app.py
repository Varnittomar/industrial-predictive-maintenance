"""Streamlit dashboard for the predictive-maintenance demo."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/machine_sensor_data.csv")
MODEL_PATH = Path("models/failure_pipeline.joblib")
METRICS_PATH = Path("reports/metrics.json")

st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭", layout="wide")
st.title("🏭 Industrial Predictive Maintenance")
st.caption("Educational demo using synthetic sensor data — not a validated industrial safety system.")

if not DATA_PATH.exists():
    st.error("Dataset not found. In the project folder, run: `python generate_data.py`")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.subheader("Sensor data overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Failure records", f"{int(df['failure'].sum()):,}")
c3.metric("Failure rate", f"{df['failure'].mean():.1%}")
c4.metric("Missing sensor cells", f"{int(df.isna().sum().sum()):,}")

left, right = st.columns(2)
with left:
    st.markdown("#### Failure counts")
    st.bar_chart(df["failure"].value_counts().sort_index().rename(index={0: "No failure", 1: "Failure"}))
with right:
    st.markdown("#### Average sensor readings by outcome")
    summary = df.groupby("failure")[[
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min"
    ]].mean().T
    summary.columns = ["No failure", "Failure"]
    st.dataframe(summary.round(2), use_container_width=True)

st.markdown("#### Explore sensor relationships")
sensor = st.selectbox("Choose a sensor", [
    "air_temperature_k", "process_temperature_k",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min"
])
chart_data = df[["failure", sensor]].dropna().copy()
chart_data["Outcome"] = chart_data["failure"].map({0: "No failure", 1: "Failure"})
st.bar_chart(chart_data.groupby("Outcome")[sensor].mean())

st.markdown("#### Sample records")
st.dataframe(df.head(100), use_container_width=True)

st.divider()
st.subheader("Score a hypothetical machine reading")
if not MODEL_PATH.exists():
    st.warning("Model not trained yet. Run `python train_model.py` to create it.")
else:
    model = joblib.load(MODEL_PATH)
    a, b, c = st.columns(3)
    with a:
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature (K)", min_value=280.0, max_value=330.0, value=300.0)
    with b:
        process_temp = st.number_input("Process temperature (K)", min_value=280.0, max_value=350.0, value=310.0)
        speed = st.number_input("Rotational speed (RPM)", min_value=500, max_value=4000, value=1500)
    with c:
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=40.0)
        wear = st.number_input("Tool wear (minutes)", min_value=0, max_value=300, value=100)

    if st.button("Estimate failure risk", type="primary"):
        sample = pd.DataFrame([{
            "air_temperature_k": air_temp,
            "process_temperature_k": process_temp,
            "rotational_speed_rpm": speed,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "machine_type": machine_type
        }])
        risk = float(model.predict_proba(sample)[0, 1])
        st.metric("Model-estimated failure probability", f"{risk:.1%}")
        st.info("This is a model score on synthetic demo data, not a real-world probability or maintenance recommendation.")

if METRICS_PATH.exists():
    with st.expander("View saved test-set metrics"):
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
