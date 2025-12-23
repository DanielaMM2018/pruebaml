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

---

## Documentación & Guías 📚

- `notebooks/FEATURE.MD` - Guía detallada de la ingeniería de features: transformaciones, selección, variables finales y recomendaciones para escalado y reducción de dimensionalidad.
- `notebooks/SUPERVISED.MD` - Documentación del modelado supervisado (objetivo ordinal para categoría TECNOLOGÍA), algoritmos evaluados y métricas de evaluación.
- `notebooks/CLUSTERING.MD` - Reporte de los experimentos de clustering (K-Means, Birch, Agglomerative), selección de K y perfiles resultantes.
- `sage_maker_scripts/SAGEMAKER.MD` - Diseño y flujo para entrenamiento y registro en SageMaker, estructura de artefactos en S3 y pasos de despliegue.

---

## Revisión general del proyecto ✅

**Alcance:** Construcción de un pipeline reproducible para generación de features, segmentación (clustering) y modelado ordinal de la intensidad de compra en la categoría *TECNOLOGÍA*.

**Fortalezas:**
- Pipeline modular y reproducible; artefactos bien organizados (`results/`, `artifacts/`, `models/`).
- Documentación técnica en múltiples MDs que facilitan replicación y revisión.
- Enfoque técnico sólido: features temporales, tratamiento de outliers y objetivo ordinal apropiado.

**Áreas de mejora / próximos pasos:**
- Añadir tests automáticos (unitarios para transformaciones, integraciones para pipelines).
- Automatizar ejecución (Makefile / CI) para reproducibilidad continua.
- Incluir ejemplos de uso y notebooks de inferencia/serving.
- Registrar más claramente los contratos de entrada/salida de cada script (schemas de CSV).

**Cómo empezar a contribuir:**
1. Instalar dependencias: `pip install -r requirements.txt`.
2. Reproducir feature engineering: `python notebooks/021_feature_engineering.py`.
3. Ejecutar modelos: `python notebooks/042_supervised_xgboost_ordinal.py` y notebooks de clustering en `03x`.
4. Abrir los MDs mencionados para entender decisiones y parámetros.


