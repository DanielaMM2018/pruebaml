# Proyecto de Machine Learning  

---

## 1. Objetivo General
Construir un **pipeline de Machine Learning en AWS** que permita:

- Analizar el comportamiento histórico de los clientes.
- Construir variables agregadas a nivel cliente (*feature engineering*).
- Modelar de forma **supervisada y ordinal** la intensidad de compra en la categoría **TECNOLOGÍA**.
- Almacenar datasets, métricas y artefactos exclusivamente en **Amazon S3**.
- Registrar el modelo entrenado en **SageMaker Model Registry**.


---

## 2. Arquitectura General (AWS)
El flujo completo se ejecuta en **Amazon SageMaker**, utilizando **S3 como capa central de datos**:

- Datos crudos → `s3://prueba-ml-dmelendez/raw/`
- Features procesadas → `s3://prueba-ml-dmelendez/processed/features/`
- Modelos y métricas → `s3://prueba-ml-dmelendez/models/`
- Registro de modelos → SageMaker Model Registry

No se utilizan archivos locales persistentes.

---

## 3. Origen de Datos
Los datos se leen directamente desde S3 mediante `awswrangler`:

- **Transacciones** → `s3://prueba-ml-dmelendez/raw/transacciones/muestra_transacciones.csv`
- **Clientes**→ `s3://prueba-ml-dmelendez/raw/customers/muestra_customers.csv`


Cada transacción contiene información de fecha, categoría, monto y puntos; el dataset de clientes aporta variables demográficas y de contexto.

---

## 4. Análisis Exploratorio de Datos (EDA)

### Objetivo del EDA
- Validar calidad y estructura de los datos.
- Entender patrones generales de consumo.
- Identificar variables relevantes para *feature engineering* y modelado.

### Principales Hallazgos
- El comportamiento de compra presenta **alta persistencia temporal**.
- La mayoría de clientes no compra tecnología en ventanas cortas, generando fuerte desbalance.
- Variables monetarias y de frecuencia muestran alta dispersión → se justifican agregaciones y transformaciones logarítmicas.
- Un enfoque binario (compra / no compra) produciría resultados artificialmente altos y poco informativos.

---

## 5. Feature Engineering (Nivel Cliente)

### Enfoque Temporal
Las variables se construyen usando información **previa al período objetivo**, tomando como referencia la **última transacción de cada cliente**.  
Esto evita *data leakage* y asegura coherencia temporal.

---

### 5.1 Features Globales
Derivadas del historial completo del cliente:

- Número total de transacciones
- Gasto total acumulado
- Recencia (días desde última compra)
- Transformaciones logarítmicas:
- `log_num_tx_total`
- `log_total_gasto`

---

### 5.2 Features por Ventanas Temporales
Ventanas relativas al último evento:

- Ventana **3 meses**
- Ventana **6 meses**

Para cada ventana:
- Número de transacciones
- Gasto total
- Puntos ganados
- Puntos redimidos

---

### 5.3 Features de Tecnología
- Gasto total en tecnología (12 meses)
- Proporción de gasto en tecnología
- Indicador de compra en tecnología
- Promedio mensual de gasto en tecnología (3 y 6 meses)

---

### 5.4 Features de Cliente
- Edad (calculada desde fecha de nacimiento)
- Estrato social
- Saldo de puntos

---

## 6. Construcción del Target (Ordinal)

### Definición
El target se define usando **los últimos 3 meses posteriores** al período de features, exclusivamente para la categoría **TECNOLOGÍA**.

### Regla de Negocio

| Número de compras | Clase |
|------------------|-------|
| 0 | sin_compra |
| 1 | baja |
| 2–3 | media |
| ≥4 | alta |

Se obtiene una variable **ordinal multiclase**, no binaria.

---

## 7. Dataset Final
- Cada fila representa un cliente.
- Variables numéricas agregadas + target ordinal.
- Valores faltantes imputados con **0**.
- Dataset almacenado en S3:
`s3://prueba-ml-dmelendez/processed/features/features_supervised.csv`


---

## 8. Formulación del Problema Supervisado
El problema se modela como **regresión ordinal**, no como clasificación directa.

### Mapeo Ordinal
| Clase original | Valor ordinal |
|---------------|---------------|
| sin_compra | 0 |
| baja | 1 |
| media | 2 |
| alta | 4 |


Este enfoque permite:
- Penalizar errores según distancia ordinal.
- Evitar métricas engañosas como ROC-AUC.
- Capturar gradientes de intención de compra.

---

## 9. Entrenamiento del Modelo (SageMaker)

### Partición de Datos
- Train / Test split: 75% / 25%
- Estratificación según la clase original del target.

---

### Modelo Seleccionado
**XGBoost Regressor**

Parámetros principales:
- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `objective = reg:squarederror`

El modelo predice un valor continuo ordinal, luego discretizado a las clases originales.

---

## 10. Evaluación
Se calculan métricas sobre el conjunto de test:

- Accuracy (post-discretización)
- Classification Report
- Confusion Matrix

El foco está en evaluar **errores ordinales**, no en métricas binarias.

---

## 11. Artefactos Generados
Durante el entrenamiento se guardan en S3:

- Modelo serializado (`model.tar.gz`)
- Métricas en JSON
- Métricas compatibles con SageMaker UI
- Classification report
- Confusion matrix
- Metadata del modelo (features, parámetros, tipo de problema)

Ruta base:
`s3://prueba-ml-dmelendez/models/xgboost_ordinal_tecnologia/`


---

## 12. Registro del Modelo
El modelo se registra en **SageMaker Model Registry**:

- Model Package Group: `xgboost-ordinal-tecnologia`
- Framework: XGBoost
- Tipo de problema: Regresión ordinal
- Métricas asociadas para trazabilidad y aprobación

Esto habilita despliegues futuros en endpoints o procesos batch.

---

## 13. Conclusión
El proyecto implementa un **pipeline productivo y reproducible en AWS**, donde:

- S3 centraliza datos y artefactos.
- SageMaker gestiona entrenamiento y registro.
- El enfoque ordinal evita soluciones triviales y mejora la utilidad del modelo.

> **Conclusión final:**  
> Modelar la intensidad de compra como un problema ordinal transforma un escenario trivialmente separable en un problema predictivo más realista, robusto y accionable para negocio.



