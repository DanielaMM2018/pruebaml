# ============================
# EDA - Exploratory Data Analysis
# ============================

# Backend NO interactivo (NO abre ventanas)
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")

# ----------------------------
# PATHS ROBUSTOS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"
results_dir = BASE_DIR / "results/eda"
results_dir.mkdir(exist_ok=True)

# ----------------------------
# CARGA DE DATOS
# ----------------------------
df_tx = pd.read_csv(data_dir / "muestra_transacciones.csv")
df_cust = pd.read_csv(data_dir / "muestra_customers.csv")

# ----------------------------
# INFO GENERAL
# ----------------------------
print(df_tx.info())
print(df_cust.info())

# ============================
# LIMPIEZA CLIENTES
# ============================

# Género válido
df_cust = df_cust[df_cust["genero"].isin(["F", "M"])]

# Edad válida
df_cust["fecha_nacimiento"] = pd.to_datetime(
    df_cust["fecha_nacimiento"], errors="coerce"
)

df_cust["edad"] = (
    pd.Timestamp.today() - df_cust["fecha_nacimiento"]
).dt.days // 365

df_cust = df_cust[(df_cust["edad"] >= 20) & (df_cust["edad"] <= 100)]

# ============================
# GÉNERO
# ============================
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cust, x="genero")
plt.title("Distribución de género (limpia)")
plt.tight_layout()
plt.savefig(results_dir / "01_genero.png")
plt.close()

# ============================
# EDAD
# ============================
plt.figure(figsize=(6, 4))
sns.histplot(df_cust["edad"], bins=25)
plt.title("Distribución de edad (20–100)")
plt.tight_layout()
plt.savefig(results_dir / "02_edad.png")
plt.close()

# ============================
# SALDO DE PUNTOS
# ============================
plt.figure(figsize=(6, 4))
sns.histplot(df_cust["saldo_puntos"], bins=50)
plt.xlim(0, 10000)
plt.title("Saldo de puntos")
plt.tight_layout()
plt.savefig(results_dir / "03_saldo_puntos.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(np.log10(df_cust["saldo_puntos"] + 1), bins=30)
plt.title("Saldo de puntos (escala log)")
plt.xlabel("log10(saldo_puntos + 1)")
plt.tight_layout()
plt.savefig(results_dir / "04_saldo_puntos_log.png")
plt.close()

# ============================
# TRANSACCIONES POR USUARIO
# ============================
tx_count = (
    df_tx.groupby("id_cliente")
    .size()
    .reset_index(name="num_transacciones")
)

plt.figure(figsize=(6, 4))
sns.histplot(tx_count["num_transacciones"], bins=50)
plt.xlim(0, 1000)
plt.title("Número de transacciones por usuario")
plt.tight_layout()
plt.savefig(results_dir / "05_num_transacciones.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(np.log10(tx_count["num_transacciones"] + 1), bins=30)
plt.title("Número de transacciones por usuario (log)")
plt.xlabel("log10(num_transacciones + 1)")
plt.tight_layout()
plt.savefig(results_dir / "06_num_transacciones_log.png")
plt.close()

# ============================
# TOTAL GASTADO POR USUARIO
# ============================
total_spent = (
    df_tx.groupby("id_cliente")["valor_transaccion"]
    .sum()
    .reset_index(name="total_gastado")
)

plt.figure(figsize=(6, 4))
sns.histplot(total_spent["total_gastado"], bins=50)
plt.xlim(0, 5e7)
plt.title("Total gastado por usuario")
plt.tight_layout()
plt.savefig(results_dir / "07_total_gastado.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(np.log10(total_spent["total_gastado"] + 1), bins=30)
plt.title("Total gastado por usuario (log)")
plt.xlabel("log10(total_gastado + 1)")
plt.tight_layout()
plt.savefig(results_dir / "08_total_gastado_log.png")
plt.close()

print("EDA finalizado correctamente. Gráficos guardados en /results/eda")
