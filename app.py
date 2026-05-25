print(">>> APP.PY STARTED <<<")

import streamlit as st
import pandas as pd
from pdm.data import prepare_data, split_train_test
from pdm.model import build_model, train_model
from pdm.features import add_derived_features

# -----------------------------
# PART 1 — Train model on Streamlit Cloud
# -----------------------------
@st.cache_resource
def get_trained_model():
    # Load dataset from repo
    df = pd.read_csv("data/ai4i2020.csv")

    # Prepare features and target
    X, y = prepare_data("data/ai4i2020.csv")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Train model
    model = build_model()
    model = train_model(model, X_train, y_train)

    return model

model = get_trained_model()

with st.expander("Show model probability distribution"):
    # Load full dataset
    df = pd.read_csv("data/ai4i2020.csv")

    # Prepare features
    X, y = prepare_data("data/ai4i2020.csv")

    # Add derived features
    X = add_derived_features(X)

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]

    st.write("Summary statistics:")
    st.write(pd.Series(probs).describe())

    st.write("Histogram of predicted probabilities:")
    st.bar_chart(pd.Series(probs))



# -----------------------------
# PART 2 — Streamlit UI
# -----------------------------
st.title("Machine Failure Risk Assessment")

st.write("Enter machine readings below:")

air_temp = st.number_input("Air temperature [K]", value=300.0)
proc_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=0.0)

def risk_level(prob):
    if prob < 0.2:
        return "Very Low"
    elif prob < 0.4:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    elif prob < 0.8:
        return "High"
    else:
        return "Very High"

if st.button("Predict Risk"):
    raw = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    df_features = add_derived_features(raw)

    prob = model.predict_proba(df_features)[0][1]
    level = risk_level(prob)

    st.metric("Failure Probability", f"{prob:.2f}")
    st.subheader(f"Risk Level: {level}")
