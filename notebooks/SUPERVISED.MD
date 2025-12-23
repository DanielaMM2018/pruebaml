# Modelado Supervisado de Pronóstico de Compra  
## Categoría: TECNOLOGÍA

## Objetivo
El objetivo de esta etapa fue construir un **modelo supervisado ordinal** para pronosticar la **intensidad de compra en la categoría TECNOLOGÍA**, utilizando información histórica del comportamiento transaccional de los clientes.

En lugar de formular el problema como una predicción binaria (compra / no compra), se adoptó un **enfoque ordinal**, que permite modelar distintos niveles de intensidad de compra y evita soluciones triviales derivadas de la fuerte persistencia temporal del comportamiento de consumo.

---

## Formulación del Problema
El comportamiento de compra en tecnología presenta una alta dependencia del historial previo. Un enfoque binario conduciría a una separación casi determinística de los clientes, produciendo métricas artificialmente altas (por ejemplo, accuracy o ROC-AUC cercanos a 1.0), sin aportar valor real al análisis.

Para evitar este problema, la variable objetivo se definió como una **variable ordinal**, con un orden natural que representa distintos niveles de intensidad de compra:

`sin_compra < baja < media < alta`

Este enfoque permite:

- Modelar gradientes de intención de compra.
- Penalizar errores según su severidad ordinal.
- Evaluar modelos con métricas más informativas y realistas.

---

## Conjunto de Datos
El dataset supervisado se construyó a partir de **variables numéricas agregadas por cliente**, derivadas del historial transaccional y del gasto por categoría.

- Cada fila representa un cliente.
- Cada columna describe su comportamiento histórico hasta el período de referencia.
- El identificador del cliente (`id_cliente`) fue excluido del modelado.

### Variable Objetivo
La variable objetivo es:

- `target_compra_tecnologia`, transformada a una escala ordinal numérica respetando el orden natural de las clases:

| Clase original | Valor ordinal |
|---------------|---------------|
| sin_compra | 0 |
| baja | 1 |
| media | 2 |
| alta | 4 |

---

## Variables Explicativas
Las variables explicativas incluyen grupos conceptuales de comportamiento:

**Actividad transaccional**
- Frecuencia de transacciones
- Recencia
- Transformaciones logarítmicas de volumen

**Valor económico**
- Gasto total y por categoría
- Puntos ganados y redimidos

**Patrones de gasto en tecnología**
- Gasto total en tecnología
- Proporción de gasto en tecnología
- Promedios de gasto en ventanas temporales recientes

**Variables sociodemográficas**
- Edad
- Estrato social

---

## Construcción de Features y Enfoque Temporal
No se crearon nuevas variables adicionales para el modelado supervisado.  
Se utilizaron directamente las variables resultantes del proceso previo de *feature engineering*.

Esta decisión se fundamenta en que:

- Las variables de gasto y actividad capturan de forma directa el comportamiento histórico del cliente.
- El objetivo es un **modelo de pronóstico por cliente**, no un modelo de series de tiempo individuales.
- Se adopta un **dataset estático con información temporal implícita**, adecuado para el enfoque ordinal propuesto.

---

## Preprocesamiento
- Para los **modelos lineales** (ElasticNet y Huber Regressor), se aplicó **StandardScaler** para asegurar estabilidad numérica y correcta optimización.
- Para el modelo basado en árboles (**XGBoost**), no se aplicó escalado, dado que este tipo de algoritmos no es sensible a la escala de las variables.
- La partición *train/test* se realizó de forma **estratificada** según la variable objetivo original.

---

## Algoritmos Evaluados
Se evaluaron tres modelos ordinales basados en regresión:

- **ElasticNet ordinal**  
  Modelo lineal regularizado (L1 + L2), utilizado como baseline interpretable.

- **Huber Regressor ordinal**  
  Modelo lineal robusto, menos sensible a outliers.

- **XGBoost ordinal**  
  Modelo de boosting basado en árboles, capaz de capturar relaciones no lineales e interacciones complejas.

En todos los casos, el modelo predice una **variable ordinal continua**, que posteriormente se discretiza en las clases originales respetando su orden natural.

---

## Evaluación de Modelos
La métrica principal utilizada fue el **Mean Absolute Error (MAE) ordinal**, adecuada para problemas con estructura ordenada, ya que penaliza los errores según su distancia ordinal.

Adicionalmente, se analizaron:

- Reportes de clasificación
- Matrices de confusión

Estas métricas permiten evaluar el desempeño por clase sin recurrir a métricas inapropiadas como ROC-AUC, que no son adecuadas para problemas ordinales.

---

## Selección del Modelo Final
La comparación entre modelos se realizó utilizando el **MAE ordinal**, seleccionando como modelo final aquel con menor error promedio.

El modelo **XGBoost ordinal** presentó el mejor desempeño global, mostrando mayor capacidad para capturar diferencias entre niveles de intensidad de compra, especialmente en escenarios no lineales.

Los modelos lineales se mantuvieron como referencia interpretable y punto de comparación.

---

## Conclusión
El enfoque ordinal permitió construir un modelo supervisado de pronóstico de compra **más realista y robusto** que una formulación binaria.

El modelo no solo identifica si un cliente comprará, sino que estima la **intensidad esperada de dicha compra**, proporcionando una base sólida para priorización de clientes y toma de decisiones de negocio.


