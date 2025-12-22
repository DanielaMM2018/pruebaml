"""
043_supervised_model_huber_ordinal.py
------------------------------------

Modelo ordinal basado en Huber Regressor.
Robusto a outliers y adecuado para pocos datos.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------
# IMPORTS
# ------------------------------------------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import classification_report, confusion_matrix

from src.models.metrics import SupervisedMetrics


# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "features"


# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(RESULTS_DIR / "features_supervised.csv")


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
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("regressor", HuberRegressor(
        epsilon=1.35,
        alpha=0.0001
    ))
])

model.fit(X_train, y_train)


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
# EVALUATION
# ------------------------------------------
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

metrics = SupervisedMetrics(
    y_true=y_test_class,
    y_pred=y_pred_class,
    y_true_ord=y_test,
    y_pred_ord=y_pred_reg,
    script_name=__file__
)

metrics.save_metrics()
metrics.save_classification_report()
metrics.save_confusion_matrix()

print("Modelo Huber ordinal entrenado y evaluado correctamente")
