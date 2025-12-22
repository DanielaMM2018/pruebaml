"""
044_model_comparison_ordinal.py
-------------------------------

Objetivo:
Comparar modelos ordinales entrenados para predecir
la intensidad de compra en TECNOLOGÍA.

Métrica principal:
- MAE ordinal (menor es mejor)
"""

# ------------------------------------------
# IMPORTS
# ------------------------------------------
from pathlib import Path
import pandas as pd


# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "results" / "supervised" / "metrics"


# ------------------------------------------
# MODELS TO COMPARE
# ------------------------------------------
models = [
    "ElasticNet",
    "xgboost",
    "huber"
]

records = []


# ------------------------------------------
# LOAD MAE METRICS
# ------------------------------------------
for model in models:
    mae_path = METRICS_DIR / f"mae_ordinal_{model}.csv"

    if mae_path.exists():
        df = pd.read_csv(mae_path)

        records.append(
            {
                "model": model,
                "mae_ordinal": df.loc[0, "value"]
            }
        )
    else:
        print(f"[WARN] No se encontró MAE para {model}")


# ------------------------------------------
# COMPARISON TABLE
# ------------------------------------------
comparison_df = pd.DataFrame(records)

comparison_df = comparison_df.sort_values(
    by="mae_ordinal",
    ascending=True   # MENOR es mejor
)


# ------------------------------------------
# SAVE
# ------------------------------------------
output_path = METRICS_DIR / "model_comparison_ordinal.csv"
comparison_df.to_csv(output_path, index=False)


# ------------------------------------------
# OUTPUT
# ------------------------------------------
print("Comparación de modelos ordinales generada")
print(comparison_df)
print(f"Archivo guardado en: {output_path}")
