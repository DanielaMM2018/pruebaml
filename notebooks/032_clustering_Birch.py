import sys
from pathlib import Path

# ------------------------------------------
# ADD PROJECT ROOT
# ------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR
sys.path.append(str(ROOT_DIR))

# ------------------------------------------
# IMPORTS
# ------------------------------------------
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import Birch
from src.clustering.metrics import ClusteringMetrics

# ------------------------------------------
# Preprocesing
# ------------------------------------------
from src.clustering.preprocessing import load_and_scale_features

X, X_scaled = load_and_scale_features(
    BASE_DIR / "results/features/features_clustering_numeric.csv"
)


# ------------------------------------------
# METRICS
# ------------------------------------------
metrics = ClusteringMetrics(
    X=X_scaled,
    script_name=__file__
)

# ------------------------------------------
# MODEL
# ------------------------------------------
birch = Birch(
    n_clusters=4,
    threshold=0.5
)

labels_birch = birch.fit_predict(X_scaled)

metrics.save_metrics(labels_birch)
print(metrics.evaluate(labels_birch))

