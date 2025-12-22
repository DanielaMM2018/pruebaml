"""
025_filter_feature_supervised.py
--------------------------------

Objetivo:
Preparar el dataset final para el modelo supervisado de predicción
de compra en la categoría TECNOLOGÍA.

Este script:
- Parte del dataset sin redundancia (salida de 023)
- Selecciona un subconjunto de variables relevantes para supervisado
- Mantiene el id_cliente para trazabilidad
- Define el target multiclase basado en transacciones recientes (TECNOLOGÍA)
"""

# ------------------------------------------
from pathlib import Path
import pandas as pd

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results/features"
DATA_DIR = BASE_DIR / "data"

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df = pd.read_csv(RESULTS_DIR / "features_sin_redundancia.csv")
df_tx = pd.read_csv(DATA_DIR / "muestra_transacciones.csv")
df_cust = pd.read_csv(DATA_DIR / "muestra_customers.csv")

df_tx["fecha"] = pd.to_datetime(df_tx["fecha"], errors="coerce")
df_cust["fecha_nacimiento"] = pd.to_datetime(df_cust["fecha_nacimiento"], errors="coerce")

# ------------------------------------------
# ÚLTIMA TRANSACCIÓN POR CLIENTE
# ------------------------------------------
last_tx = df_tx.groupby("id_cliente")["fecha"].max().reset_index(name="last_date")

# ------------------------------------------
# DEFINE FEATURES FOR SUPERVISED MODEL
# ------------------------------------------
FEATURES_SUPERVISED = [
    "num_tx_total",
    "total_gasto",
    "recencia_dias",
    "log_num_tx_total",
    "log_total_gasto",
    "gasto_TECNOLOGIA_12m",
    "pct_tecnologia_12m",
    "avg_gasto_tecnologia_3m",
    "avg_gasto_tecnologia_6m",
    "num_tx_1m",
    "puntos_redimidos_1m",
    "puntos_redimidos_3m",
    "edad",
    "estrato_social",
    "saldo_puntos",
]

# ------------------------------------------
# DEFINE TARGET MULTICLASE (TECNOLOGÍA, últimos 3 meses)
# ------------------------------------------
TARGET_MONTHS = 3

# Filtrar transacciones de los últimos 3 meses
target_tx = df_tx.merge(last_tx, on="id_cliente", how="left")
target_tx = target_tx[
    (target_tx["fecha"] >= target_tx["last_date"] - pd.DateOffset(months=TARGET_MONTHS)) &
    (target_tx["fecha"] <  target_tx["last_date"])
]

# Solo transacciones de TECNOLOGÍA
target_tx = target_tx[target_tx["categoria"] == "TECNOLOGIA"]

# Contar transacciones por cliente
target = target_tx.groupby("id_cliente").agg(
    num_tx_target=("id_transaccion", "count")
).reset_index()

# Asegurar que todos los clientes estén presentes
target = last_tx[["id_cliente"]].merge(target, on="id_cliente", how="left")
target["num_tx_target"] = target["num_tx_target"].fillna(0)

# Categorizar
def categorize_target(n):
    if n >= 4:
        return "alta"
    elif n >= 2:
        return "media"
    elif n >= 1:
        return "baja"
    else:
        return "sin_compra"

target["target_compra_tecnologia"] = target["num_tx_target"].apply(categorize_target)


# Merge con df original
df = df.merge(
    target[["id_cliente", "target_compra_tecnologia"]],
    on="id_cliente",
    how="left"
)

print(df["target_compra_tecnologia"].value_counts())
# Rellenar NaN
df["target_compra_tecnologia"].fillna("sin_compra", inplace=True)

print("Distribución target:")
print(df["target_compra_tecnologia"].value_counts())

# ------------------------------------------
# FINAL COLUMN SELECTION
# ------------------------------------------
final_columns = ["id_cliente"] + FEATURES_SUPERVISED + ["target_compra_tecnologia"]
df_supervised = df[final_columns].copy()

# ------------------------------------------
# BASIC VALIDATIONS
# ------------------------------------------
df_supervised = df_supervised.dropna(subset=["target_compra_tecnologia"])

# ------------------------------------------
# SAVE OUTPUT
# ------------------------------------------
output_path = RESULTS_DIR / "features_supervised.csv"
df_supervised.to_csv(output_path, index=False)

print("Dataset supervisado multiclase generado correctamente")
print(f"Archivo guardado en: {output_path}")
print(f"Shape final: {df_supervised.shape}")
print(df_supervised["target_compra_tecnologia"].value_counts())

