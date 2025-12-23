# Modelos de Clustering No Supervisado

## Objetivo
El objetivo de esta etapa fue segmentar a los usuarios en grupos homogéneos mediante técnicas de clustering no supervisado, con el fin de identificar perfiles de comportamiento similares a partir de variables numéricas derivadas de información transaccional y sociodemográfica.  

Esta segmentación permite descubrir **estructuras latentes en los datos** sin asumir categorías predefinidas y constituye un insumo clave para posteriores análisis de negocio, personalización de estrategias y toma de decisiones.

---

## Conjunto de Datos
El dataset utilizado contiene aproximadamente **500 usuarios**, donde:

- Cada fila representa un cliente.
- Cada columna corresponde a una variable numérica que describe su comportamiento.
- Se utilizaron **34 variables numéricas** relacionadas con:

**Actividad transaccional:** `num_tx_total`, `num_tx_1m`, `num_tx_3m`, `recencia_dias`, `log_num_tx_total`  
**Valor económico:** `total_gasto`, `gasto_1m`, `log_total_gasto`  
**Uso y redención de puntos:** `saldo_puntos`, `puntos_ganados_total`, `puntos_redimidos_total`, `puntos_ganados_1m`, `puntos_redimidos_3m`, `puntos_redimidos_6m`  
**Gasto por categoría:** `gasto_HOGAR_12m`, `gasto_TECNOLOGIA_12m`, `gasto_MERCADO_12m`, `gasto_ENTRETENIMIENTO_12m`, `gasto_TRANSPORTE_12m`  
**Patrones tecnológicos:** `pct_tecnologia_12m`, `compra_tecnologia`, `avg_gasto_tecnologia_3m`, `avg_gasto_tecnologia_6m`  
**Variables sociodemográficas:** `edad`, `estrato_social`  

> **Nota:** El identificador del usuario (ID) no fue incluido en el modelo, ya que no aporta información para medir similitud y podría introducir ruido.

---

## Preprocesamiento
Antes de aplicar los modelos de clustering:

- Se seleccionaron únicamente variables numéricas relevantes.
- Las variables fueron escaladas utilizando **RobustScaler**, para:
  - Reducir el impacto de valores atípicos.
  - Garantizar que todas las variables contribuyan de forma equilibrada a las métricas de distancia.
- El escalado se centralizó en un módulo reutilizable para asegurar consistencia entre todos los modelos evaluados.

---

## Algoritmos Evaluados
Se probaron tres algoritmos de clustering no supervisado:

1. **K-Means**  
   Basado en distancia euclidiana, asigna cada usuario al centroide más cercano. Requiere definir el número de clusters \(K\).  
   *Eficiente e interpretable.*

2. **Birch**  
   Algoritmo jerárquico que construye micro-clusters y luego los agrupa.  
   *Eficiente para datasets pequeños y medianos.*

3. **Agglomerative Clustering**  
   Algoritmo jerárquico ascendente, fusiona progresivamente clusters hasta alcanzar el número deseado.  

---

## Selección del Número de Clusters
- Se evaluaron valores de K entre 2 y 8 utilizando **K-Means** como modelo exploratorio, aplicando:
  - **Elbow Method:** reducción de la varianza interna.
  - **Silhouette Score:** medición de cohesión y separación entre clusters.

> Se determinó **K = 4**, al observar un punto de inflexión claro en el método del codo y un máximo local del Silhouette Score, ofreciendo un equilibrio entre separación y simplicidad interpretativa.

---

## Evaluación de Modelos
| Modelo        | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---------------|------------|----------------|------------------|
| K-Means       | 0.894      | 0.844          | 682.36           |
| Birch         | 0.872      | 0.922          | 646.48           |
| Agglomerative | 0.872      | 0.922          | 646.48           |

> **Modelo seleccionado:** **K-Means**, por mayor cohesión interna, menor solapamiento y mejor separación global.

---

## Interpretación de los Clusters
- El modelo final segmentó a ~500 usuarios en **4 clusters**, representando perfiles de comportamiento similares.  
- Se utilizó la tabla `cluster_summary` para analizar los centroides y valores promedio de las variables.

**Indicadores Clave para la interpretación:**

- **Actividad:** `num_tx_total`, `num_tx_1m`, `num_tx_3m`, `recencia_dias`  
- **Valor:** `total_gasto`, `gasto_1m`, `gasto_HOGAR_12m`, `gasto_TECNOLOGIA_12m`  
- **Engagement:** `saldo_puntos`, `puntos_ganados_total`, `puntos_redimidos_total`  
- **Patrones temporales:** ventanas 1, 3, 6, 12 meses  
- **Perfil demográfico:** `edad`, `estrato_social`  

---

## Perfiles Identificados

1. **Usuarios de bajo valor / ocasionales:** bajos `num_tx_total` y `total_gasto`, alta `recencia_dias`.  
2. **Usuarios activos de valor medio:** actividad constante, gasto distribuido en múltiples categorías.  
3. **Usuarios de alto valor / premium:** alta frecuencia, alto gasto, fuerte uso de puntos, baja recencia.  
4. **Usuarios atípicos / extremos:** valores excepcionalmente altos en gasto y transacciones.

> La visualización mediante **PCA** confirmó la separación entre estos grupos.

---

## Conclusión
El uso de clustering no supervisado permitió:

- Identificar segmentos de usuarios con características similares.
- Obtener una visión estructurada, objetiva y reproducible.
- Proporcionar insumos para **personalización de campañas, priorización de clientes y estrategias comerciales**.

> 🔹 **Frase final:** El clustering descubre estructuras latentes, no clases preexistentes, asegurando segmentación objetiva y accionable.
