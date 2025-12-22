# src/clustering/preprocessing.py

import pandas as pd
from sklearn.preprocessing import RobustScaler


def load_and_scale_features(path):
    """
    Carga features numéricas y aplica RobustScaler.

    Parameters
    ----------
    path : str or Path
        Ruta al CSV con las features numéricas

    Returns
    -------
    X : pd.DataFrame
        Features originales
    X_scaled : np.ndarray
        Features escaladas
    """
    X = pd.read_csv(path)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled
