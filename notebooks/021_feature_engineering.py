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
df_tx["fecha"] = pd.to_datetime(df_tx["fecha"], errors="coerce")
df_cust["fecha_nacimiento"] = pd.to_datetime(df_cust["fecha_nacimiento"], errors="coerce")

# ------------------------------------------
# ÚLTIMA TRANSACCIÓN POR CLIENTE
# ------------------------------------------
last_tx = df_tx.groupby("id_cliente")["fecha"].max().reset_index(name="last_date")

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
# FUNCIONES DE VENTANA BASADAS EN last_date
# ------------------------------------------
def window_features(df, last_tx, m_start, m_end):
    d = df.merge(last_tx, on="id_cliente", how="left")
    d = d[
        (d["fecha"] >= d["last_date"] - pd.DateOffset(months=m_start)) &
        (d["fecha"] <  d["last_date"] - pd.DateOffset(months=m_end))
    ]
    months = m_start - m_end
    return d.groupby("id_cliente").agg(
        **{
            f"num_tx_{months}m": ("id_transaccion", "count"),
            f"gasto_{months}m": ("valor_transaccion", "sum"),
            f"puntos_ganados_{months}m": ("puntos", lambda x: x[x > 0].sum()),
            f"puntos_redimidos_{months}m": ("puntos", lambda x: -x[x < 0].sum())
        }
    )

def avg_tech(df, last_tx, m_start, m_end):
    d = df.merge(last_tx, on="id_cliente", how="left")
    d = d[
        (d["fecha"] >= d["last_date"] - pd.DateOffset(months=m_start)) &
        (d["fecha"] <  d["last_date"] - pd.DateOffset(months=m_end))
    ]
    d = d[d["categoria"] == "TECNOLOGIA"]
    months = m_start - m_end
    return (
        d.groupby("id_cliente")["valor_transaccion"]
        .sum()
        .reset_index(name=f"avg_gasto_tecnologia_{months}m")
        .assign(**{f"avg_gasto_tecnologia_{months}m":
                       lambda x: x[f"avg_gasto_tecnologia_{months}m"] / months})
    )


# ------------------------------------------
# FEATURES GENERALES
# ------------------------------------------
user_base = df_tx.groupby("id_cliente").agg(
    num_tx_total=("id_transaccion", "count"),
    total_gasto=("valor_transaccion", "sum"),
    puntos_ganados_total=("puntos", lambda x: x[x > 0].sum()),
    puntos_redimidos_total=("puntos", lambda x: -x[x < 0].sum()),
    recencia_dias=("fecha", lambda x: (x.max() - x).min().days),
    num_tx_negativas=("tx_negativa", "sum")
)

# ------------------------------------------
# FEATURES TEMPORALES
# ------------------------------------------
# ------------------------------------------
# FEATURES TEMPORALES - renombrando correctamente
# ------------------------------------------

# tx_1m
tx_1m = window_features(df_tx, last_tx, 3, 0).rename(columns={
    "num_tx_3m": "num_tx_1m",
    "gasto_3m": "gasto_1m",
    "puntos_ganados_3m": "puntos_ganados_1m",
    "puntos_redimidos_3m": "puntos_redimidos_1m"
}).reset_index()  # <--- asegura id_cliente como columna

# tx_3m
tx_3m = window_features(df_tx, last_tx, 6, 3).rename(columns={
    "num_tx_3m": "num_tx_3m",
    "gasto_3m": "gasto_3m",
    "puntos_ganados_3m": "puntos_ganados_3m",
    "puntos_redimidos_3m": "puntos_redimidos_3m"
}).reset_index()

# tx_6m
tx_6m = window_features(df_tx, last_tx, 9, 3).rename(columns={
    "num_tx_6m": "num_tx_6m",
    "gasto_6m": "gasto_6m",
    "puntos_ganados_6m": "puntos_ganados_6m",
    "puntos_redimidos_6m": "puntos_redimidos_6m"
}).reset_index()

# tx_12m
tx_12m = window_features(df_tx, last_tx, 15, 3).rename(columns={
    "num_tx_12m": "num_tx_12m",
    "gasto_12m": "gasto_12m",
    "puntos_ganados_12m": "puntos_ganados_12m",
    "puntos_redimidos_12m": "puntos_redimidos_12m"
}).reset_index()

# ------------------------------------------
# MERGE CON user_base
# ------------------------------------------
for tx, meses in zip([tx_1m, tx_3m, tx_6m, tx_12m], [1, 3, 6, 12]):
    cols_to_merge = ["id_cliente",
                     f"num_tx_{meses}m",
                     f"gasto_{meses}m",
                     f"puntos_ganados_{meses}m",
                     f"puntos_redimidos_{meses}m"]
    user_base = user_base.merge(tx[cols_to_merge], on="id_cliente", how="left")
# ------------------------------------------
# FEATURES POR CATEGORÍA (12m)
# ------------------------------------------
df_12m = df_tx.merge(last_tx, on="id_cliente", how="left")
df_12m = df_12m[df_12m["fecha"] >= df_12m["last_date"] - pd.DateOffset(months=12)]

cat_pivot = df_12m.pivot_table(
    index="id_cliente",
    columns="categoria",
    values="valor_transaccion",
    aggfunc="sum",
    fill_value=0
)
cat_pivot.columns = [f"gasto_{c}_12m" for c in cat_pivot.columns]

cat_pivot["pct_tecnologia_12m"] = (
    cat_pivot.get("gasto_TECNOLOGIA_12m", 0) / (cat_pivot.sum(axis=1) + 1)
)
cat_pivot["compra_tecnologia"] = (cat_pivot.get("gasto_TECNOLOGIA_12m", 0) > 0).astype(int)

# ------------------------------------------
# PROMEDIOS TECNOLOGÍA
# ------------------------------------------
# Generamos los promedios directamente y reemplazamos lo viejo
avg_tech_3m = avg_tech(df_tx, last_tx, 6, 3)
avg_tech_6m = avg_tech(df_tx, last_tx, 9, 3)
avg_tech_12m = avg_tech(df_tx, last_tx, 15, 3)

# En lugar de merge por merge, concatenamos horizontalmente y usamos set_index para id_cliente
avg_tech_df = pd.concat(
    [avg_tech_3m.set_index("id_cliente"),
     avg_tech_6m.set_index("id_cliente"),
     avg_tech_12m.set_index("id_cliente")],
    axis=1
).reset_index()

# ------------------------------------------
# MERGE FINAL
# ------------------------------------------
df_users = (
    df_cust
    .merge(last_tx, on="id_cliente", how="left")
    .merge(user_base, on="id_cliente", how="left")
    .merge(cat_pivot, on="id_cliente", how="left")
    .merge(avg_tech_df, on="id_cliente", how="left") 
)

# ------------------------------------------
# EDAD
# ------------------------------------------
df_users["edad"] = (df_users["last_date"] - df_users["fecha_nacimiento"]).dt.days // 365

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