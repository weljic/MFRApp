import streamlit as st
import pandas as pd
from pdm.serialize import load_model
from pdm.features import add_derived_features

# Load trained model
model = load_model("models/model.joblib")

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

st.title("Machine Failure Risk Assessment")

st.write("Enter machine readings below:")

# Raw inputs needed for feature engineering
air_temp = st.number_input("Air temperature [K]", value=300.0)
proc_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=0.0)

if st.button("Predict Risk"):
    raw = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    # Apply same feature engineering as training
    df_features = add_derived_features(raw)

    # Predict
    prob = model.predict_proba(df_features)[0][1]
    level = risk_level(prob)

    st.metric("Failure Probability", f"{prob:.2f}")
    st.subheader(f"Risk Level: {level}")