# ==========================================
# CORRELATION MATRIX - NUMERIC FEATURES
# ==========================================

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
results_dir = BASE_DIR / "results/features"
results_dir.mkdir(exist_ok=True)

# ------------------------------------------
# LOAD FEATURES
# ------------------------------------------
df = pd.read_csv(results_dir / "features_usuarios_final.csv")

# ------------------------------------------
# SELECT NUMERIC FEATURES
# ------------------------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

print(f"Total variables numéricas: {numeric_df.shape[1]}")

# ------------------------------------------
# CORRELATION MATRIX
# ------------------------------------------
corr_matrix = numeric_df.corr()

# ------------------------------------------
# SAVE MATRIX (CSV)
# ------------------------------------------
corr_matrix.to_csv(
    results_dir / "correlation_matrix_full.csv"
)

print("Matriz de correlación guardada")

# ------------------------------------------
# HEATMAP
# ------------------------------------------
plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1,
    cbar_kws={"shrink": 0.8}
)
plt.title("Matriz de correlación - Features numéricos", fontsize=16)
plt.tight_layout()
plt.savefig(
    results_dir / "correlation_matrix_full.png"
)
plt.close()

print("📊 Heatmap guardado")
