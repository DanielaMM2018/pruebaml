"""
042_supervised_model_xgboost_ordinal_sagemaker.py
-------------------------------------------------

Entrenamiento y registro en SageMaker:
- Lee features_supervised desde S3
- Entrena XGBoost ordinal (regresión)
- Guarda artefactos en S3
- Registra el modelo en SageMaker Model Registry
"""

import json
import tarfile
import tempfile
from pathlib import Path

import boto3
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

import sagemaker
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics


# -------------------------------------------------
# SAGEMAKER CONTEXT
# -------------------------------------------------
session = sagemaker.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()

s3_client = boto3.client("s3")
sm_client = boto3.client("sagemaker")


# -------------------------------------------------
# S3 PATHS
# -------------------------------------------------
BUCKET = "prueba-ml-dmelendez"

FEATURES_S3_URI = (
    f"s3://{BUCKET}/processed/features/features_supervised.csv"
)

ARTIFACTS_PREFIX = "models/xgboost_ordinal_tecnologia"
MODEL_S3_PREFIX = f"{ARTIFACTS_PREFIX}/model"
METRICS_S3_PREFIX = f"{ARTIFACTS_PREFIX}/metrics"
METADATA_S3_PREFIX = f"{ARTIFACTS_PREFIX}/metadata"


# -------------------------------------------------
# LOAD DATA FROM S3
# -------------------------------------------------
df = pd.read_csv(FEATURES_S3_URI)


# -------------------------------------------------
# TARGET ORDINAL
# -------------------------------------------------
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


# -------------------------------------------------
# FEATURES / TARGET
# -------------------------------------------------
X = df.drop(columns=[TARGET_RAW, TARGET_ORD, ID_COL])
y = df[TARGET_ORD]


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=df[TARGET_RAW]
)


# -------------------------------------------------
# MODEL
# -------------------------------------------------
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


# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------
y_pred_reg = model.predict(X_test)

bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
labels = ["sin_compra", "baja", "media", "alta"]

y_pred_class = pd.cut(
    y_pred_reg,
    bins=bins,
    labels=labels
)

y_test_class = df.loc[y_test.index, TARGET_RAW]


# -------------------------------------------------
# METRICS
# -------------------------------------------------
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

metrics_payload = {
    "accuracy": accuracy,
    "num_train_samples": len(X_train),
    "num_test_samples": len(X_test),
    "target_type": "ordinal_multiclass",
    "labels": labels
}


# -------------------------------------------------
# SAVE ARTIFACTS LOCALLY (TEMP)
# -------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    model_path = tmpdir / "model.joblib"
    joblib.dump(model, model_path)

    metrics_path = tmpdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    report_path = tmpdir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(clf_report, f, indent=4)

    cm_path = tmpdir / "confusion_matrix.csv"
    pd.DataFrame(
        conf_matrix,
        index=labels,
        columns=labels
    ).to_csv(cm_path)

    metadata_payload = {
        "model_name": "xgboost-ordinal-tecnologia",
        "model_type": "XGBRegressor",
        "problem_type": "ordinal_regression",
        "target_mapping": TARGET_MAP,
        "training_params": model.get_params(),
        "features": list(X.columns),
        "framework": "xgboost"
    }

    metadata_path = tmpdir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_payload, f, indent=4)

    # -------------------------------------------------
    # CREATE model.tar.gz (SageMaker compatible)
    # -------------------------------------------------
    tar_path = tmpdir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.joblib")

    # -------------------------------------------------
    # UPLOAD TO S3
    # -------------------------------------------------
    model_s3_uri = session.upload_data(
        path=str(tar_path),
        bucket=BUCKET,
        key_prefix=MODEL_S3_PREFIX
    )

    metrics_s3_uri = session.upload_data(
        path=str(metrics_path),
        bucket=BUCKET,
        key_prefix=METRICS_S3_PREFIX
    )

    report_s3_uri = session.upload_data(
        path=str(report_path),
        bucket=BUCKET,
        key_prefix=METRICS_S3_PREFIX
    )

    metadata_s3_uri = session.upload_data(
        path=str(metadata_path),
        bucket=BUCKET,
        key_prefix=METADATA_S3_PREFIX
    )


# -------------------------------------------------
# REGISTER MODEL IN MODEL MANAGER
# -------------------------------------------------
MODEL_PACKAGE_GROUP = "xgboost-ordinal-tecnologia"

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=metrics_s3_uri,
        content_type="application/json"
    )
)

sm_model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3",
        instance_type="ml.m5.large"
    ),
    model_data=model_s3_uri,
    role=role,
    sagemaker_session=session
)

model_package = sm_model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=MODEL_PACKAGE_GROUP,
    model_metrics=model_metrics,
    description="XGBoost ordinal para predicción de compra en tecnología"
)

print("Modelo registrado exitosamente en SageMaker Model Registry")
print(f"Model Package ARN: {model_package.model_package_arn}")
