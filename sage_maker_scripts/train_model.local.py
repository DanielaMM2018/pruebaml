"""
042_supervised_model_xgboost_ordinal_local.py
--------------------------------------------

Versión local que simula:
- Entrenamiento en SageMaker
- Persistencia de artefactos en S3
- Registro de metadata para Model Manager
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from xgboost import XGBRegressor


# ------------------------------------------
# PATHS (LOCAL MODEL REGISTRY SIMULADO)
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_ROOT = BASE_DIR / "models" / "xgboost_ordinal_tecnologia"
MODEL_DIR = MODEL_ROOT / "model"
METRICS_DIR = MODEL_ROOT / "metrics"
METADATA_DIR = MODEL_ROOT / "metadata"

for d in [MODEL_DIR, METRICS_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------
# LOAD DATA
# ------------------------------------------
FEATURES_PATH = BASE_DIR / "results" / "features" / "features_supervised.csv"
df = pd.read_csv(FEATURES_PATH)


# ------------------------------------------
# TARGET ORDINAL
# ------------------------------------------
TARGET_RAW = "target_compra_tecnologia"
TARGET_ORD = "target_ordinal"
ID_COL = "id_cliente"

TARGET_MAP = {
    "sin_compra": 0,
    "baja": 1,
    "media": 2,
    "alta": 3
}

df[TARGET_ORD] = df[TARGET_RAW].map(TARGET_MAP)


# ------------------------------------------
# FEATURES / TARGET
# ------------------------------------------
X = df.drop(columns=[TARGET_RAW, TARGET_ORD, ID_COL])
y = df[TARGET_ORD]


# ------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=df[TARGET_RAW]
)


# ------------------------------------------
# MODEL
# ------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ------------------------------------------
# SAVE MODEL (equivalente a model.tar.gz en S3)
# ------------------------------------------
joblib.dump(model, MODEL_DIR / "model.joblib")


# ------------------------------------------
# PREDICTIONS
# ------------------------------------------
y_pred_reg = model.predict(X_test)

bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
labels = ["sin_compra", "baja", "media", "alta"]

y_pred_class = pd.cut(
    y_pred_reg,
    bins=bins,
    labels=labels
)

y_test_class = df.loc[y_test.index, TARGET_RAW]


# ------------------------------------------
# METRICS
# ------------------------------------------
clf_report = classification_report(
    y_test_class,
    y_pred_class,
    output_dict=True
)

conf_matrix = confusion_matrix(
    y_test_class,
    y_pred_class,
    labels=labels
)

accuracy = accuracy_score(y_test_class, y_pred_class)


# ------------------------------------------
# SAVE METRICS (simula metrics.json en S3)
# ------------------------------------------
metrics_payload = {
    "accuracy": accuracy,
    "num_train_samples": len(X_train),
    "num_test_samples": len(X_test),
    "target_type": "ordinal_multiclass",
    "labels": labels
}

with open(METRICS_DIR / "metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=4)

with open(METRICS_DIR / "classification_report.json", "w") as f:
    json.dump(clf_report, f, indent=4)

pd.DataFrame(
    conf_matrix,
    index=labels,
    columns=labels
).to_csv(METRICS_DIR / "confusion_matrix.csv")


# ------------------------------------------
# MODEL METADATA (Model Manager SIMULADO)
# ------------------------------------------
model_metadata = {
    "model_name": "xgboost-ordinal-tecnologia",
    "model_type": "XGBRegressor",
    "problem_type": "ordinal_regression",
    "target_mapping": TARGET_MAP,
    "training_params": model.get_params(),
    "features": list(X.columns),
    "framework": "xgboost",
    "framework_version": model.__class__.__module__,
    "artifact_path": str(MODEL_DIR / "model.joblib")
}

with open(METADATA_DIR / "model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=4)


print("Modelo XGBoost ordinal entrenado y registrado localmente")
print(f"Artefactos guardados en: {MODEL_ROOT}")
