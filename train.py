# train.py

import datetime as dt
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "data.csv"
PLOT_FILE = "rf_maxtemp_pred_vs_real.png"
METRICS_FILE = "rf_maxtemp_metrics.txt"


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Eliminar columnas que sobran si existen
    for col in ["Unnamed: 0", "cloud"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Fecha -> número de días
    if "Date" not in df.columns:
        raise ValueError("El dataset debe tener una columna 'Date'.")

    # Intenta parsear formato tipo 01.04.2017 o similar
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    origin = dt.datetime(2017, 4, 1)
    df["Date_num"] = (df["Date"] - origin).dt.days

    # Codificar variable categórica "weather"
    if "weather" not in df.columns:
        raise ValueError("El dataset debe tener una columna 'weather'.")

    weather_map = {
        "Fog": 0,
        "Haze": 1,
        "Light Rain": 2,
        "Light rain": 2,  # por si viene con minúscula
        "Mist": 3,
        "Rain": 4,
        "Smoke": 5,
    }
    df["weather_code"] = df["weather"].map(weather_map)

    # Quitar filas donde no se pudo mapear o falta algo
    df = df.dropna(subset=["Date_num", "weather_code"])

    # Asegurar tipo numérico
    df["weather_code"] = df["weather_code"].astype(int)

    return df


def train_plot_and_metrics(df: pd.DataFrame, plot_path: str, metrics_path: str):
    if "maxtemp" not in df.columns:
        raise ValueError("El dataset debe tener una columna 'maxtemp'.")

    features = ["Date_num", "pressure", "humidity", "mean wind speed", "weather_code"]

    for f in features:
        if f not in df.columns:
            raise ValueError(f"Falta la columna '{f}' en el dataset.")

    X = df[features]
    y = df["maxtemp"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=201,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ====== MÉTRICAS ======
    mae = mean_absolute_error(y_test, y_pred)

    # En algunas versiones de scikit-learn no existe "squared" en mean_squared_error,
    # así que calculamos RMSE a partir del MSE para máxima compatibilidad.
    mse = mean_squared_error(y_test, y_pred)  # por defecto squared=True
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # Mostrar en los logs (GitHub Actions los muestra aquí)
    print("===== Métricas Random Forest (maxtemp) =====")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # Guardar también en un archivo de texto
    with open(metrics_path, "w") as f:
        f.write("Métricas Random Forest (maxtemp)\n")
        f.write(f"MAE : {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²  : {r2:.3f}\n")

    # ====== GRÁFICO REAL vs PREDICTO ======
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicciones RF")
    min_temp = min(y_test.min(), y_pred.min())
    max_temp = max(y_test.max(), y_pred.max())
    plt.plot([min_temp, max_temp], [min_temp, max_temp], "r--", label="Línea ideal")
    plt.xlabel("Temperatura real (maxtemp)")
    plt.ylabel("Temperatura predicha (maxtemp)")
    plt.title("Random Forest: real vs predicho (maxtemp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado en: {plot_path}")
    print(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontró '{DATA_FILE}' en la carpeta actual.")

    df = load_and_preprocess(DATA_FILE)
    train_plot_and_metrics(df, PLOT_FILE, METRICS_FILE)
