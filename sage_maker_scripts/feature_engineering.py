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
# REFERENCE DATE
# ------------------------------------------
ref_date = df_tx["fecha"].max()

# ------------------------------------------
# BASE FEATURES (GLOBAL)
# ------------------------------------------
base = df_tx.groupby("id_cliente").agg(
    num_tx_total=("id_transaccion", "count"),
    total_gasto=("valor_transaccion", "sum"),
    recencia_dias=("fecha", lambda x: (ref_date - x.max()).days),
).reset_index()

base["log_num_tx_total"] = np.log10(base["num_tx_total"] + 1)
base["log_total_gasto"] = np.log10(base["total_gasto"] + 1)

# ------------------------------------------
# FUNCTION: WINDOW FEATURES
# ------------------------------------------
def window_features(df, months):
    fmin = ref_date - pd.DateOffset(months=months)
    d = df[df["fecha"] >= fmin]

    return d.groupby("id_cliente").agg(
        **{
            f"num_tx_{months}m": ("id_transaccion", "count"),
            f"gasto_{months}m": ("valor_transaccion", "sum"),
            f"puntos_ganados_{months}m": (
                "puntos", lambda x: x[x > 0].sum()
            ),
            f"puntos_redimidos_{months}m": (
                "puntos", lambda x: -x[x < 0].sum()
            ),
        }
    ).reset_index()

# ------------------------------------------
# TEMPORAL FEATURES
# ------------------------------------------
tx_1m = window_features(df_tx, 1)
tx_3m = window_features(df_tx, 3)
tx_6m = window_features(df_tx, 6)

# ------------------------------------------
# TECNOLOGÍA 12M
# ------------------------------------------
df_12m = df_tx[df_tx["fecha"] >= ref_date - pd.DateOffset(months=12)]

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
# AVG TECNOLOGÍA
# ------------------------------------------
def avg_tech(df, months):
    fmin = ref_date - pd.DateOffset(months=months)
    d = df[
        (df["fecha"] >= fmin) &
        (df["categoria"] == "TECNOLOGIA")
    ]

    return (
        d.groupby("id_cliente")["valor_transaccion"]
        .sum()
        .reset_index(name=f"avg_gasto_tecnologia_{months}m")
        .assign(**{
            f"avg_gasto_tecnologia_{months}m":
                lambda x: x[f"avg_gasto_tecnologia_{months}m"] / months
        })
    )

avg_tech_3m = avg_tech(df_tx, 3)
avg_tech_6m = avg_tech(df_tx, 6)

# ------------------------------------------
# CUSTOMER FEATURES
# ------------------------------------------
cust = df_cust[
    ["id_cliente", "estrato_social", "saldo_puntos", "fecha_nacimiento"]
].copy()


cust["fecha_nacimiento"] = pd.to_datetime(
    cust["fecha_nacimiento"], errors="coerce"
)

cust["edad"] = (
    ref_date - cust["fecha_nacimiento"]
).dt.days // 365

cust = cust.drop(columns="fecha_nacimiento")

# ------------------------------------------
# FINAL MERGE
# ------------------------------------------
df = (
    cust
    .merge(base, on="id_cliente", how="left")
    .merge(tx_1m[["id_cliente", "num_tx_1m", "gasto_1m", "puntos_ganados_1m"]],
           on="id_cliente", how="left")
    .merge(tx_3m[["id_cliente", "puntos_redimidos_3m"]],
           on="id_cliente", how="left")
    .merge(tx_6m[["id_cliente", "puntos_redimidos_6m"]],
           on="id_cliente", how="left")
    .merge(tech_12m, on="id_cliente", how="left")
    .merge(avg_tech_3m, on="id_cliente", how="left")
    .merge(avg_tech_6m, on="id_cliente", how="left")
)

df.fillna(0, inplace=True)

# ------------------------------------------
# SAVE TO S3
# ------------------------------------------
wr.s3.to_csv(
    df=df,
    path=S3_FEATURES,
    index=False
)

print("Features guardadas en:", S3_FEATURES)