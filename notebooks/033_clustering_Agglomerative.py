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
from sklearn.cluster import AgglomerativeClustering
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
agg = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

labels_agg = agg.fit_predict(X_scaled)

metrics.save_metrics(labels_agg)
print(metrics.evaluate(labels_agg))
