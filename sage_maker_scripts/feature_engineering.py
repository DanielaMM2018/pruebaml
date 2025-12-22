# ==========================================
# FEATURE ENGINEERING - USER LEVEL
# ==========================================

import pandas as pd
import numpy as np
import awswrangler as wr

print("pandas:", pd.__version__)
print("awswrangler:", wr.__version__)

S3_BUCKET = "prueba-ml-dmelendez"

S3_TX_PATH = (
    f"s3://{S3_BUCKET}/raw/transacciones/muestra_transacciones.csv"
)
S3_CUSTOMERS_PATH = (
    f"s3://{S3_BUCKET}/raw/customers/muestra_customers.csv"
)
S3_FEATURES = (
    f"s3://{S3_BUCKET}/processed/features/features_supervised.csv"
)
# ------------------------------------------
# LOAD DATA FROM S3
# ------------------------------------------
df_tx = wr.s3.read_csv(S3_TX_PATH)
df_cust = wr.s3.read_csv(S3_CUSTOMERS_PATH)

df_tx["fecha"] = pd.to_datetime(df_tx["fecha"], errors="coerce")
df_cust["fecha_nacimiento"] = pd.to_datetime(
    df_cust["fecha_nacimiento"], errors="coerce"
)

# ------------------------------------------
# LAST TRANSACTION DATE PER USER
# ------------------------------------------

last_tx = (
    df_tx.groupby("id_cliente")["fecha"]
    .max()
    .reset_index(name="last_date")
)

# ------------------------------------------
# BASE FEATURES (GLOBAL HISTORY)
# ------------------------------------------
base = (
    df_tx.merge(last_tx, on="id_cliente", how="left")
    .groupby("id_cliente")
    .agg(
        num_tx_total=("id_transaccion", "count"),
        total_gasto=("valor_transaccion", "sum"),
        recencia_dias=("fecha", lambda x: (x.max() - x).min().days),
    )
    .reset_index()
)

base["log_num_tx_total"] = np.log10(base["num_tx_total"] + 1)
base["log_total_gasto"] = np.log10(base["total_gasto"] + 1)

# ------------------------------------------
# FUNCTION: WINDOW FEATURES (FIXED)
# ------------------------------------------
def window_features(df, last_tx, m_start, m_end):
    d = df.merge(last_tx, on="id_cliente", how="left")

    d = d[
        (d["fecha"] >= d["last_date"] - pd.DateOffset(months=m_start)) &
        (d["fecha"] <  d["last_date"] - pd.DateOffset(months=m_end))
    ]

    return d.groupby("id_cliente").agg(
        **{
            f"num_tx_{m_start-m_end}m": ("id_transaccion", "count"),
            f"gasto_{m_start-m_end}m": ("valor_transaccion", "sum"),
            f"puntos_ganados_{m_start-m_end}m": (
                "puntos", lambda x: x[x > 0].sum()
            ),
            f"puntos_redimidos_{m_start-m_end}m": (
                "puntos", lambda x: -x[x < 0].sum()
            ),
        }
    ).reset_index()

# ------------------------------------------
# TEMPORAL FEATURES
# ------------------------------------------
# 1M = último mes (solo para features agregadas)
tx_1m = window_features(df_tx, last_tx, m_start=3, m_end=0)

# 3M = meses t-4 a t-1
tx_3m = window_features(df_tx, last_tx, m_start=6, m_end=3)

# 6M = meses t-7 a t-1
tx_6m = window_features(df_tx, last_tx, m_start=9, m_end=3)


# ------------------------------------------
# TECNOLOGÍA 12M
# ------------------------------------------

df_12m = df_tx.merge(last_tx, on="id_cliente", how="left")

df_12m = df_12m[
    df_12m["fecha"] >= df_12m["last_date"] - pd.DateOffset(months=12)
]


tech_12m = (
    df_12m[df_12m["categoria"] == "TECNOLOGIA"]
    .groupby("id_cliente")["valor_transaccion"]
    .sum()
    .reset_index(name="gasto_TECNOLOGIA_12m")
)

total_12m = (
    df_12m.groupby("id_cliente")["valor_transaccion"]
    .sum()
    .reset_index(name="gasto_total_12m")
)

tech_12m = tech_12m.merge(total_12m, on="id_cliente", how="right").fillna(0)

tech_12m["pct_tecnologia_12m"] = (
    tech_12m["gasto_TECNOLOGIA_12m"] /
    (tech_12m["gasto_total_12m"] + 1)
)

tech_12m["compra_tecnologia"] = (
    tech_12m["gasto_TECNOLOGIA_12m"] > 0
).astype(int)

tech_12m = tech_12m[
    ["id_cliente", "gasto_TECNOLOGIA_12m",
     "pct_tecnologia_12m", "compra_tecnologia"]
]

# ------------------------------------------
# TECNOLOGÍA 3M y 6M (PROMEDIO MENSUAL)
# ------------------------------------------
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
        .assign(
            **{
                f"avg_gasto_tecnologia_{months}m":
                    lambda x: x[f"avg_gasto_tecnologia_{months}m"] / months
            }
        )
    )


avg_tech_3m = avg_tech(df_tx, last_tx, 6, 3)
avg_tech_6m = avg_tech(df_tx, last_tx, 9, 3)

# ------------------------------------------
# TARGET – TECNOLOGÍA (LAST MONTH)
# ------------------------------------------
TARGET_MONTHS = 3

# Filtrar transacciones del target
target_tx = df_tx.merge(last_tx, on="id_cliente", how="left")

target_tx = target_tx[
    (target_tx["fecha"] >= target_tx["last_date"] - pd.DateOffset(months=TARGET_MONTHS)) &
    (target_tx["fecha"] <  target_tx["last_date"])
]

target_tx = target_tx[target_tx["categoria"] == "TECNOLOGIA"]

# Contar transacciones por cliente
target = target_tx.groupby("id_cliente").agg(
    num_tx_target=("id_transaccion", "count")
).reset_index()

# Asegurar todos los clientes estén en target
target = last_tx[["id_cliente"]].merge(target, on="id_cliente", how="left")

target["num_tx_target"] = target["num_tx_target"].fillna(0)


def categorize_target(n):
    if n >= 5:
        return "alta"
    elif n >= 3:
        return "media"
    elif n >= 1:
        return "baja"
    else:
        return "sin_compra"
        
target["target_compra_tecnologia"] = target["num_tx_target"].apply(categorize_target)

target = target[["id_cliente", "target_compra_tecnologia"]]

target["target_compra_tecnologia"].value_counts()


# ------------------------------------------
# CUSTOMER FEATURES
# ------------------------------------------
cust = df_cust[
    ["id_cliente", "estrato_social", "saldo_puntos", "fecha_nacimiento"]
].copy()

cust = cust.merge(last_tx, on="id_cliente", how="left")

cust["edad"] = (
    cust["last_date"] - cust["fecha_nacimiento"]
).dt.days // 365

cust = cust.drop(columns=["fecha_nacimiento", "last_date"])

# ------------------------------------------
# FINAL MERGE
# ------------------------------------------

df_final = (
    cust
    .merge(base, on="id_cliente", how="right")
    # Ventana 3 meses previos al target (t-6 a t-3)
    .merge(tx_3m[["id_cliente", "num_tx_3m", "gasto_3m", 
                  "puntos_ganados_3m", "puntos_redimidos_3m"]],
           on="id_cliente", how="left")
    # Ventana 6 meses previos al target (t-9 a t-3)
    .merge(tx_6m[["id_cliente", "num_tx_6m", "gasto_6m", 
                  "puntos_ganados_6m", "puntos_redimidos_6m"]],
           on="id_cliente", how="left")
    # 12M historial tecnología
    .merge(tech_12m, on="id_cliente", how="left")
    # Promedios mensuales de tecnología
    .merge(avg_tech_3m, on="id_cliente", how="left")
    .merge(avg_tech_6m, on="id_cliente", how="left")
    # Target multiclase (últimos 3 meses)
    .merge(target, on="id_cliente", how="left")
)

# Reemplazar NaN solo en columnas numéricas
num_cols = df_final.select_dtypes(include=np.number).columns
df_final[num_cols] = df_final[num_cols].fillna(0)


# ------------------------------------------
# SAVE TO S3
# ------------------------------------------
wr.s3.to_csv(
    df=df_final,
    path=S3_FEATURES,
    index=False
)

print("Features guardadas en:", S3_FEATURES)