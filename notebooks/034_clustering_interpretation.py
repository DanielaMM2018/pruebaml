# notebooks/034_clustering_interpretation.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path("..")
FEATURES_PATH = BASE_DIR / "results/features/features_clustering_numeric.csv"
OUTPUT_DIR = BASE_DIR / "results/clustering/interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
X = pd.read_csv(FEATURES_PATH)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------
# LOAD FINAL MODEL (KMeans)
# ------------------------------------------
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X_scaled)

# ------------------------------------------
# SAVE USERS + CLUSTER
# ------------------------------------------
df_clusters = X.copy()
df_clusters["cluster"] = labels

df_clusters.to_csv(
    OUTPUT_DIR / "users_with_clusters.csv",
    index=False
)

# ------------------------------------------
# TABLE SUMMARY PER CLUSTER
# ------------------------------------------
cluster_summary = (
    df_clusters
    .groupby("cluster")
    .mean()
    .round(2)
)

cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_summary.csv"
)

# ------------------------------------------
# PCA VISUALIZATION
# ------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    alpha=0.7
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA - Visualización de Clusters")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)

plt.savefig(OUTPUT_DIR / "pca_clusters.png")
plt.close()

print("[OK] Interpretación de clusters generada")
