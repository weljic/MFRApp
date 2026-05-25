\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&#x09;PART 1

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



1\. Problem statement, access, and error costs

We want to predict the short‑term failure risk of a production machine based on real‑time sensor readings (temperatures, rotational speed, torque, tool wear, etc.). Predictions are exposed to maintenance engineers through a simple web app: they enter the current readings and receive a failure probability plus a qualitative risk band (e.g., Very Low, Low, Medium, High, Very High) to help prioritize inspections and interventions.



Success means:

The model meaningfully separates high‑risk from low‑risk states (e.g., high‑risk machines fail more often than low‑risk ones). The tool integrates smoothly into existing workflows (fast, stable, understandable).



Error costs:

False negative (missed failure): high cost—unplanned downtime, safety risk, scrap, rush repairs. This is more critical.

False positive (unnecessary intervention): moderate cost—extra inspections, possible over‑maintenance, some lost time.



Operationally, we’d bias the system toward catching more potential failures (higher recall on the failure class), accepting some extra false alarms.



2\. System design diagram (conceptual)



Data ingestion

* Source: machine sensor logs / historian / MES.
* Batch export (e.g., daily CSV/Parquet) into a data lake or warehouse.



Feature store / preprocessing

* ETL jobs to clean data, handle missing values, encode categories, engineer features (e.g., temperature deltas, load indicators).
* Store curated features and labels in a feature store or analytics DB.



Training pipeline



* Training script / pipeline (e.g., in pdm package) that:

&#x09;Splits train/test.

&#x09;Fits preprocessing on train only.

&#x09;Trains model(s), evaluates, logs metrics.

&#x09;Model registry (e.g., MLflow, internal registry) to store versions, metrics, and artifacts.



Model serving

* Deployed model as:

&#x09;A web app (Streamlit) for interactive use.

&#x09;Optionally, a REST API for integration with other tools.

* The app loads the serialized model + preprocessing pipeline and scores new readings.



Monitoring



* Logging of inputs and predictions (with timestamps and machine IDs).



* Dashboards for:

&#x09;Data drift (feature distributions vs. training).

&#x09;Prediction distribution shift.

&#x09;Outcome‑based metrics when labels become available (precision/recall, calibration).



Components missing in this challenge but needed in production

* Authentication/authorization (role‑based access).
* CI/CD for model and app.
* Model registry and approval workflow.
* Automated data quality checks.
* Alerting on monitoring metrics (e.g., drift, performance degradation).
* Robust logging, observability, and backup/rollback mechanisms.



3\. Model selection strategy and trade‑offs

Strategy:

* Start with simple, interpretable baselines (e.g., logistic regression, decision tree) to understand feature relationships and set a performance floor.
* Move to tree‑based ensemble models (e.g., RandomForest, GradientBoosting) to better capture non‑linear interactions and handle mixed feature types.
* Use cross‑validation and metrics suited to class imbalance (e.g., ROC‑AUC, PR‑AUC, recall, F2‑score).



Trade‑offs:



Interpretability vs. performance:



* Logistic regression / single trees are easier to explain (coefficients, simple rules) but may underfit complex patterns.
* Ensembles (RandomForest, GradientBoosting) typically perform better on tabular, non‑linear data but are less transparent.



For maintenance engineers, we can balance this by:



* Using an ensemble model for better risk ranking.
* Providing feature importance and simple explanations (e.g., “High torque + high tool wear increased risk”) in documentation or dashboards.



Given the operational cost of missed failures, we favor a model with strong recall and ranking performance, while still being explainable enough via feature importance and example‑based explanations.



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&#x09;PART 2

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

1\. Reproducible, modular pipeline

Code is structured as a Python package (e.g., pdm/) with modules such as:

* data.py — data loading, prepare\_data, split\_train\_test.
* features.py — feature engineering, add\_derived\_features.
* model.py — build\_model, train\_model, evaluation helpers.
* serialize.py — save/load model and preprocessing artifacts.



A training script or notebook imports these functions, so the same logic is reused in both training and the web app.

Configuration (paths, model hyperparameters, random seeds) is centralized to keep runs reproducible.



2\. Training, evaluation, and metrics choice

Pipeline:

* Load raw data.
* Prepare features and target (prepare\_data).
* Split into train/test (split\_train\_test) with stratification on the failure label.
* Fit preprocessing (scaling/encoding/feature engineering) only on the training set, then apply to test.
* Train the model (build\_model + train\_model).
* Evaluate on the held‑out test set.



Metrics:



* The failure class is rare (class imbalance), so accuracy is misleading.
* We focus on:

&#x09;Recall (sensitivity) for the failure class — to minimize missed failures.

&#x09;Precision — to understand false alarm rate.

&#x09;F‑beta (e.g., F2) — to weight recall more than precision.

&#x09;ROC‑AUC / PR‑AUC — to assess ranking quality across thresholds.



In the operational context, we’d choose a threshold that achieves acceptable recall (e.g., ≥80–90%) while keeping false positives manageable.



Demonstrating correct preprocessing:

* Preprocessing steps (e.g., scalers, encoders, feature engineering) are implemented as part of a pipeline or fitted objects.
* They are fit on X\_train only, then applied to X\_test and to new data in the app, avoiding leakage.



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&#x09;PART 3

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



Web app deployment

1\. Output design

The app takes as input:

* Air temperature \[K]
* Process temperature \[K]
* Rotational speed \[rpm]
* Torque \[Nm]
* Tool wear \[min]



It outputs:



* Failure probability (0–1, shown as a metric).
* A risk level bucket (Very Low, Low, Medium, High, Very High) based on probability thresholds.



This gives engineers both a numeric score and a simple, actionable label for prioritization.



2\. Deployment

The app is implemented in Streamlit (app.py), using the pdm package for data prep and model loading.



It is deployed on Streamlit Community Cloud (or similar free tier).



The README includes:



The live URL (e.g., https://mfrapp-7dvj4atydsco8jmpc8mpov.streamlit.app/).



Basic usage instructions (what inputs to provide, how to interpret outputs).



Any environment notes (e.g., Python version, required packages).







