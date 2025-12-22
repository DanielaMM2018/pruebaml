import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path("..")


# ------------------------------------------
# Preprocesing
# ------------------------------------------


from src.clustering.preprocessing import load_and_scale_features

X, X_scaled = load_and_scale_features(
    BASE_DIR / "results/features/features_clustering_numeric.csv"
)


from src.clustering.metrics import ClusteringMetrics

#metrics = ClusteringMetrics(X_scaled)
metrics = ClusteringMetrics(
    X=X_scaled,
    script_name=__file__
)


from sklearn.cluster import KMeans

metrics.plot_elbow(
    KMeans,
    k_range=range(2, 9),
    random_state=42,
    n_init=20
)

metrics.plot_silhouette_vs_k(
    KMeans,
    k_range=range(2, 9),
    random_state=42,
    n_init=20
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

labels_kmeans = kmeans.fit_predict(X_scaled)

metrics.save_metrics(labels_kmeans)
print(metrics.evaluate(labels_kmeans))

