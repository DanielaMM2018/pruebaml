# 🧩 Ingeniería de Features y Selección de Variables

## 1. Objetivo de la ingeniería de features
El objetivo fue transformar los datos transaccionales y sociodemográficos en un conjunto de **variables numéricas a nivel usuario**, capaces de:
- Capturar intensidad de uso
- Reflejar recencia y frecuencia
- Identificar preferencias por categoría
- Ser aptas para **clustering no supervisado** y modelos predictivos posteriores

---

## 2. Construcción de features a nivel usuario
La agregación se realizó por `id_cliente`, combinando:
- Información transaccional histórica
- Ventanas temporales
- Atributos demográficos

---

## 3. Features generadas desde transacciones

### 3.1 Actividad y comportamiento general
Estas variables capturan el **comportamiento global** del usuario:

| Columna               | Descripción                    | Propósito             |
|-----------------------|--------------------------------|----------------------|
| num_tx_total          | Total de transacciones históricas | Intensidad de uso    |
| total_gasto           | Gasto monetario total           | Valor del cliente    |
| puntos_ganados_total  | Total de puntos acumulados      | Engagement           |
| puntos_redimidos_total| Total de puntos redimidos       | Uso de beneficio     |
| recencia_dias         | Días desde la última transacción | Actividad reciente  |
| num_tx_negativas      | Transacciones inconsistentes    | Control de calidad   |

> Permiten diferenciar usuarios activos vs inactivos, frecuentes vs ocasionales y detectar posibles anomalías.

---

### 3.2 Features por ventanas temporales
Métricas agregadas para 1, 3, 6 y 12 meses:

| Columna base          | Significado          |
|-----------------------|--------------------|
| num_tx_{m}m           | Número de transacciones |
| gasto_{m}m            | Gasto acumulado     |
| puntos_ganados_{m}m   | Puntos ganados     |
| puntos_redimidos_{m}m | Puntos redimidos   |

> Capturan tendencias recientes que no se observan en totales históricos y son claves para segmentación.

---

### 3.3 Features por categoría (últimos 12 meses)

| Columna               | Descripción                     |
|-----------------------|---------------------------------|
| gasto_<CATEGORIA>_12m | Gasto total por categoría       |
| pct_tecnologia_12m    | Proporción del gasto en TECNOLOGÍA |
| compra_tecnologia     | Indicador binario de compra en TECNOLOGÍA |

> Permiten identificar preferencias de consumo y usuarios con afinidad tecnológica, clave para clustering y forecasting.

---

### 3.4 Promedios de gasto en TECNOLOGÍA

| Columna                    | Descripción           |
|-----------------------------|--------------------|
| avg_gasto_tecnologia_3m     | Promedio mensual 3 meses |
| avg_gasto_tecnologia_6m     | Promedio mensual 6 meses |
| avg_gasto_tecnologia_12m    | Promedio mensual 12 meses |

> Evaluar estabilidad del gasto tecnológico en el tiempo.

---

## 4. Features demográficas

| Columna          | Descripción             |
|------------------|-----------------------|
| edad             | Edad del cliente       |
| estrato_social   | Nivel socioeconómico   |
| saldo_puntos     | Puntos acumulados actuales |

> Complementan el comportamiento transaccional con contexto demográfico.

---

## 5. Transformaciones aplicadas
Se aplicó **log-transform** a variables asimétricas:

| Columna original  | Transformada         |
|------------------|--------------------|
| num_tx_total      | log_num_tx_total    |
| total_gasto       | log_total_gasto     |
| puntos_ganados_total | log_puntos_ganados |

> Reducir skewness y evitar que valores extremos dominen el clustering.

---

## 6. Análisis de correlación
- Matriz de correlación de **Pearson** sobre 52 variables numéricas  
- Entregables:
  - `correlation_matrix_full.csv`
  - `correlation_matrix_full.png`

> Identificar redundancias, evitar multicolinealidad y reducir dimensionalidad.

---

## 7. Identificación y eliminación de redundancia
- Se eliminaron pares con |ρ| ≥ 0.9  
- Patrones observados:
  - Totales vs ventanas largas
  - Ventanas temporales anidadas
  - Promedios vs acumulados
  - Transformaciones logarítmicas altamente correlacionadas  

📄 Archivo: `high_correlation_pairs.csv`

---

## 8. Resultado de la reducción de features

| Métrica           | Valor |
|------------------|-------|
| Features iniciales | 52    |
| Features eliminadas | 16   |
| Features finales   | 36    |
| Reducción         | ~31%  |

📄 Dataset final: `features_sin_redundancia.csv`  

> Eliminadas por no aportar información adicional, generar ruido o sesgar el clustering.

---

## 9. Recomendaciones para un clustering adecuado

### 9.1 Escalado
- **Obligatorio**: RobustScaler recomendado por outliers

### 9.2 Selección final orientada a distancia
- Revisar varianza post-escalado
- Evitar dominancia de variables acumulativas

### 9.3 Reducción dimensional (opcional)
- PCA para clustering y visualización (80–90% varianza)

### 9.4 Evaluación de modelos
- KMeans
- Gaussian Mixture Models
- DBSCAN  

**Métricas:**
- Silhouette
- Davies–Bouldin
- Calinski–Harabasz

---

## 10. Conclusión
Se realizó un proceso completo de **ingeniería de features**, agregación por usuario y selección de variables:
- Desde datos crudos hasta un dataset numérico optimizado
- Eliminación de redundancias mejora interpretabilidad y estabilidad
- Dataset final listo para modelos **no supervisados y supervisados**
