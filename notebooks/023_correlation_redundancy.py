# ==========================================
# FEATURE REDUNDANCY FILTER
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results/features"

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(results_dir / "features_usuarios_final.csv")
numeric_df = df.select_dtypes(include=["int64", "float64"])

# ------------------------------------------
# CORRELATION
# ------------------------------------------
corr = numeric_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

CORR_THRESHOLD = 0.90

# ------------------------------------------
# REDUNDANT FEATURES
# ------------------------------------------
redundant_features = [
    col for col in upper.columns if any(upper[col] >= CORR_THRESHOLD)
]

print("🔴 Features redundantes:")
for f in redundant_features:
    print(f"- {f}")

# ------------------------------------------
# EXPLAINABLE PAIRS
# ------------------------------------------
high_corr_pairs = (
    upper.stack()
    .reset_index()
    .rename(columns={
        "level_0": "feature_1",
        "level_1": "feature_2",
        0: "correlation"
    })
)

high_corr_pairs = high_corr_pairs[
    high_corr_pairs["correlation"] >= CORR_THRESHOLD
].sort_values("correlation", ascending=False)

high_corr_pairs.to_csv(
    results_dir / "high_correlation_pairs.csv",
    index=False
)

# ------------------------------------------
# FINAL DATASET
# ------------------------------------------
features_final = numeric_df.drop(columns=redundant_features)

features_final.to_csv(
    results_dir / "features_sin_redundancia.csv",
    index=False
)

print("Dataset final sin redundancia guardado")
