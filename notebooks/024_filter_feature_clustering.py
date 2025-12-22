# ==========================================
# PREPARE DATA FOR CLUSTERING
# ==========================================

import pandas as pd
from pathlib import Path

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results/features"

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(results_dir / "features_sin_redundancia.csv")

print(f"Dataset original: {df.shape}")

# ------------------------------------------
# SELECT NUMERIC VARIABLES
# ------------------------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

# ------------------------------------------
# DROP IDENTIFIER
# ------------------------------------------
if "id_cliente" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["id_cliente"])

print(f"Dataset numérico para clustering: {numeric_df.shape}")

# ------------------------------------------
# SAVE
# ------------------------------------------
numeric_df.to_csv(
    results_dir / "features_clustering_numeric.csv",
    index=False
)

print("Dataset numérico listo para clustering")
