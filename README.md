# Prueba Puntos Colombia

Descripción de los archivos y carpetas más relevantes del proyecto.

## Estructura y explicación

- data/
  - `muestra_transacciones.csv` - muestra de transacciones (id_cliente, fecha, valor_transaccion, categoria, puntos, tipo_transaccion, id_transaccion, ...).
  - `muestra_customers.csv` - información de clientes (id_cliente, fecha_nacimiento, genero, saldo_puntos, ...).

- notebooks/
  - `01_eda.py` - script de EDA que genera gráficos en `results/eda` (distribuciones, recencia, histogramas, etc.).
  - `021_feature_engineering.py` - genera features a nivel usuario y guarda `results/features/features_usuarios_final.csv`.
  - `022_correlation_matrix.py` - calcula y guarda la matriz de correlación (`results/features/correlation_matrix_full.csv`) y un heatmap (`correlation_matrix_full.png`).
  - `023_correlation_redundancy.py` - detecta features redundantes (umbral 0.90), guarda `features_sin_redundancia.csv` y `high_correlation_pairs.csv`.

- results/
  - `eda/` - gráficos resultantes del EDA.
  - `features/` - features intermedios y finales, matrices de correlación y pares de alta correlación.

- `requirements.txt` - listado de dependencias; usar para instalar el entorno.

## Flujo de trabajo recomendado

1. Preparar entorno y dependencias.
2. Ejecutar `01_eda.py` para inspeccionar los datos.
3. Ejecutar `021_feature_engineering.py` para crear `features_usuarios_final.csv`.
4. Ejecutar `022_correlation_matrix.py` y `023_correlation_redundancy.py` para identificar redundancias y limpiar features.
5. Ejecutar notebooks de clustering (`03x`) y modelado (`04x`) para evaluar y comparar modelos.

## Notas

- Los artefactos (CSVs, PNGs) se guardan en `results/` para mantener reproducibilidad.
