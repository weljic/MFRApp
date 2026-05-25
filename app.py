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

# -----------------------------
# PART 2 — Streamlit UI
# -----------------------------
st.title("Machine Failure Risk Assessment")

st.write("Enter machine readings below:")

air_temp = st.number_input("Air temperature [K]", value=300.0)
proc_temp = st