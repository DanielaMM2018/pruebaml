from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    mean_absolute_error
)


class SupervisedMetrics:
    """
    Clase para evaluar y guardar métricas de modelos supervisados
    (binarios, multiclase y ordinales).
    """

    def __init__(
        self,
        y_true,
        y_pred,
        script_name,
        y_proba=None,
        y_true_ord=None,
        y_pred_ord=None
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.y_true_ord = y_true_ord
        self.y_pred_ord = y_pred_ord

        # Root del proyecto
        self.project_root = Path(__file__).resolve().parents[2]

        # Carpeta de salida
        self.output_dir = (
            self.project_root / "results" / "supervised" / "metrics"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Nombre del modelo (ElasticNet / XGBoost / RandomForest)
        parts = Path(script_name).stem.split("_")

        # penúltimo bloque
        self.model_name = parts[-2]


    # ------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------
    def compute_metrics(self):
        """
        Calcula métricas según el tipo de modelo.
        """
        metrics = {}

        # Métrica ORDINAL
        if self.y_true_ord is not None and self.y_pred_ord is not None:
            metrics["mae_ordinal"] = mean_absolute_error(
                self.y_true_ord,
                self.y_pred_ord
            )

        # Métrica BINARIA (solo si hay probabilidades)
        if self.y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(
                self.y_true,
                self.y_proba
            )

        return metrics

    def save_metrics(self):
        """
        Guarda métricas numéricas en CSV (una por modelo).
        """
        metrics = self.compute_metrics()

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
    # REPORTES
    # ------------------------------------------------------------------
    def save_classification_report(self):
        """
        Guarda classification report como CSV.
        """
        report = classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True
        )

        df = pd.DataFrame(report).transpose()
        path = self.output_dir / f"classification_report_{self.model_name}.csv"
        df.to_csv(path)

        print(f"[OK] Classification report guardado: {path}")

    def save_confusion_matrix(self):
        """
        Guarda confusion matrix como CSV.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        labels = sorted(self.y_true.unique())

        df = pd.DataFrame(
            cm,
            index=[f"Real_{l}" for l in labels],
            columns=[f"Pred_{l}" for l in labels]
        )

        path = self.output_dir / f"confusion_matrix_{self.model_name}.csv"
        df.to_csv(path)

        print(f"[OK] Confusion matrix guardada: {path}")

    # ------------------------------------------------------------------
    # GRÁFICAS (solo binario)
    # ------------------------------------------------------------------
    def plot_roc_curve(self):
        """
        Guarda curva ROC (solo si aplica).
        """
        if self.y_proba is None:
            return

        RocCurveDisplay.from_predictions(
            self.y_true,
            self.y_proba
        )

        path = self.output_dir / f"roc_curve_{self.model_name}.png"
        plt.savefig(path)
        plt.close()

        print(f"[OK] Curva ROC guardada: {path}")
