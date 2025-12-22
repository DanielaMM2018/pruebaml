from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class ClusteringMetrics:
    """
    Clase para evaluar y guardar métricas de clustering no supervisado.
    """

    def __init__(self, X, script_name):
        """
        Parameters
        ----------
        X : array-like
            Matriz de features ya escaladas
        script_name : str
            __file__ del script que ejecuta el modelo
        """
        self.X = X

        # Root del proyecto (pruebaml/)
        self.project_root = Path(__file__).resolve().parents[2]

        # Carpeta de salida
        self.output_dir = (
            self.project_root / "results" / "clustering" / "metrics"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extraer nombre del modelo desde el script
        # Ej: 071_clustering_KMeans.py -> KMeans
        self.model_name = (
            Path(script_name)
            .stem
            .split("_")[-1]
        )

    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------
    def evaluate(self, labels):
        """
        Calcula métricas estándar de clustering.
        """
        return {
            "silhouette": silhouette_score(self.X, labels),
            "davies_bouldin": davies_bouldin_score(self.X, labels),
            "calinski_harabasz": calinski_harabasz_score(self.X, labels),
        }

    def save_metrics(self, labels):
        """
        Guarda cada métrica en un CSV independiente.
        """
        metrics = self.evaluate(labels)

        for metric_name, value in metrics.items():
            df = pd.DataFrame(
                {
                    "metric": [metric_name],
                    "value": [value],
                }
            )

            filename = f"{metric_name}_{self.model_name}.csv"
            path = self.output_dir / filename
            df.to_csv(path, index=False)

        print(f"[OK] Métricas guardadas en: {self.output_dir}")
        return metrics

    # ------------------------------------------------------------------
    # GRÁFICAS
    # ------------------------------------------------------------------
    def plot_elbow(self, model_class, k_range, **model_kwargs):
        """
        Método del codo (solo modelos con inertia_)
        """
        inertias = []

        for k in k_range:
            model = model_class(n_clusters=k, **model_kwargs)
            model.fit(self.X)
            inertias.append(model.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, marker="o")
        plt.xlabel("Número de clusters (K)")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Method ({self.model_name})")
        plt.grid(True)

        path = self.output_dir / f"elbow_{self.model_name}.png"
        plt.savefig(path)
        plt.close()

        print(f"[OK] Gráfica guardada: {path}")

    def plot_silhouette_vs_k(self, model_class, k_range, **model_kwargs):
        """
        Silhouette Score vs K
        """
        scores = []

        for k in k_range:
            model = model_class(n_clusters=k, **model_kwargs)
            labels = model.fit_predict(self.X)
            scores.append(silhouette_score(self.X, labels))

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, scores, marker="o")
        plt.xlabel("Número de clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title(f"Silhouette vs K ({self.model_name})")
        plt.grid(True)

        path = self.output_dir / f"silhouette_vs_k_{self.model_name}.png"
        plt.savefig(path)
        plt.close()

        print(f"[OK] Gráfica guardada: {path}")
