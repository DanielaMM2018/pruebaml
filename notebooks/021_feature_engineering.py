# ==========================================
# FEATURE ENGINEERING - USER LEVEL
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------
# PATHS
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data"
results_dir = BASE_DIR / "results/features"
results_dir.mkdir(exist_ok=True)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
df_tx = pd.read_csv(data_dir / "muestra_transacciones.csv")
df_cust = pd.read_csv(data_dir / "muestra_customers.csv")

df_tx["fecha"] = pd.to_datetime(df_tx["fecha"], errors="coerce")

# ------------------------------------------
# REFERENCE DATE
# ------------------------------------------
ref_date = df_tx["fecha"].max()

# ------------------------------------------
# LIMPIEZA CATEGORÍAS
# ------------------------------------------
category_map = {
    "COMIDAS": "COMIDA",
    "DEPORTE": "DEPORTE",
    "DEPORTES": "DEPORTE",
    "LICORES Y CIGARRILLOS": "LICORES",
    "SALUD Y BELLEZA": "SALUD_BELLEZA"
}

df_tx["categoria"] = df_tx["categoria"].replace(category_map)

# ------------------------------------------
# CONTROL DE CALIDAD
# ------------------------------------------
df_tx["tx_negativa"] = (
    (df_tx["tipo_transaccion"] == "Sale") &
    ((df_tx["valor_transaccion"] < 0) | (df_tx["puntos"] < 0))
)

# ------------------------------------------
# FUNCIONES
# ------------------------------------------
def features_por_ventana(df, meses):
    fecha_min = ref_date - pd.DateOffset(months=meses)
    df_w = df[df["fecha"] >= fecha_min]

    return df_w.groupby("id_cliente").agg(
        **{
            f"num_tx_{meses}m": ("id_transaccion", "count"),
            f"gasto_{meses}m": ("valor_transaccion", "sum"),
            f"puntos_ganados_{meses}m": ("puntos", lambda x: x[x > 0].sum()),
            f"puntos_redimidos_{meses}m": ("puntos", lambda x: -x[x < 0].sum())
        }
    )

def avg_monthly_tech_spend(df, meses):
    fecha_min = ref_date - pd.DateOffset(months=meses)

    df_w = df[
        (df["fecha"] >= fecha_min) &
        (df["categoria"] == "TECNOLOGIA")
    ].copy()

    df_w["valor_transaccion"] = pd.to_numeric(
        df_w["valor_transaccion"], errors="coerce"
    )

    return (
        df_w.groupby("id_cliente", as_index=False)["valor_transaccion"]
        .sum()
        .assign(**{
            f"avg_gasto_tecnologia_{meses}m":
                lambda x: x["valor_transaccion"] / meses
        })
        .drop(columns="valor_transaccion")
    )


# ------------------------------------------
# FEATURES GENERALES
# ------------------------------------------
user_base = df_tx.groupby("id_cliente").agg(
    num_tx_total=("id_transaccion", "count"),
    total_gasto=("valor_transaccion", "sum"),
    puntos_ganados_total=("puntos", lambda x: x[x > 0].sum()),
    puntos_redimidos_total=("puntos", lambda x: -x[x < 0].sum()),
    recencia_dias=("fecha", lambda x: (ref_date - x.max()).days),
    num_tx_negativas=("tx_negativa", "sum")
)

# ------------------------------------------
# FEATURES TEMPORALES
# ------------------------------------------
for m in [1, 3, 6, 12]:
    user_base = user_base.merge(
        features_por_ventana(df_tx, m),
        on="id_cliente",
        how="left"
    )

# ------------------------------------------
# FEATURES POR CATEGORÍA (12m)
# ------------------------------------------
df_12m = df_tx[df_tx["fecha"] >= ref_date - pd.DateOffset(months=12)]

cat_pivot = df_12m.pivot_table(
    index="id_cliente",
    columns="categoria",
    values="valor_transaccion",
    aggfunc="sum",
    fill_value=0
)

cat_pivot.columns = [f"gasto_{c}_12m" for c in cat_pivot.columns]

cat_pivot["pct_tecnologia_12m"] = (
    cat_pivot.get("gasto_TECNOLOGIA_12m", 0) /
    (cat_pivot.sum(axis=1) + 1)
)

cat_pivot["compra_tecnologia"] = (
    cat_pivot.get("gasto_TECNOLOGIA_12m", 0) > 0
).astype(int)

# ------------------------------------------
# PROMEDIOS TECNOLOGÍA
# ------------------------------------------
avg_tech = (
    avg_monthly_tech_spend(df_tx, 3)
    .merge(avg_monthly_tech_spend(df_tx, 6), on="id_cliente", how="outer")
    .merge(avg_monthly_tech_spend(df_tx, 12), on="id_cliente", how="outer")
)

# ------------------------------------------
# MERGE FINAL
# ------------------------------------------
df_users = (
    df_cust
    .merge(user_base, on="id_cliente", how="left")
    .merge(cat_pivot, on="id_cliente", how="left")
    .merge(avg_tech, on="id_cliente", how="left")
)

# ------------------------------------------
# EDAD
# ------------------------------------------
df_users["fecha_nacimiento"] = pd.to_datetime(
    df_users["fecha_nacimiento"], errors="coerce"
)

df_users["edad"] = (
    ref_date - df_users["fecha_nacimiento"]
).dt.days // 365

# ------------------------------------------
# LOG TRANSFORMS
# ------------------------------------------
df_users["log_num_tx_total"] = np.log10(df_users["num_tx_total"] + 1)
df_users["log_total_gasto"] = np.log10(df_users["total_gasto"] + 1)
df_users["log_puntos_ganados"] = np.log10(df_users["puntos_ganados_total"] + 1)


# ------------------------------------------
# FILL NA
# ------------------------------------------
df_users.fillna(0, inplace=True)

# ------------------------------------------
# REDONDEO NUMÉRICO (EVITA ERRORES EN EXCEL)
# ------------------------------------------
num_cols = df_users.select_dtypes(include=["float64", "float32"]).columns
df_users[num_cols] = df_users[num_cols].round(2)

# ------------------------------------------
# SAVE
# ------------------------------------------
df_users.to_csv(
    results_dir / "features_usuarios_final.csv",
    index=False
)

