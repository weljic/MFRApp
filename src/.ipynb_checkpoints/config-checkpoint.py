# src/pdm/config.py

# Full dataset path (Windows-safe raw string)
DATA_PATH = r"C:\Users\PRS02K50\OneDrive - AholdDelhaize.com\Documents\razno\positions\ips_energy\ai4i2020.csv"

TARGET_COL = "Machine failure"

# Columns we want to drop from the raw CSV
COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "Type",                 
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "MPower [W]",
    "Temperature_difference [K]",
]
