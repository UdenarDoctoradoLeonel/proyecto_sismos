import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/clean/earthquakes_prepared.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")


def load_data(path, max_rows=50_000):
    """
    Carga el dataset preparado.
    - Usa una muestra aleatoria si hay demasiadas filas.
    - Se queda solo con columnas numéricas.
    """
    print("Cargando datos desde:", path)
    df = pd.read_csv(path)

    print(f"Filas totales en el archivo: {len(df)}")

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        print(f"Usando una muestra aleatoria de {max_rows} filas para entrenar.")

    # Nos aseguramos de que 'significant' exista
    if "significant" not in df.columns:
        raise ValueError("La columna 'significant' no está en el archivo preparado.")

    # Solo columnas numéricas
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if "significant" not in numeric_cols:
        raise ValueError("'significant' no es numérica en el archivo preparado.")

    # y es la etiqueta
    y = df["significant"]

    # X son todas las numéricas menos la etiqueta
    feature_cols = [c for c in numeric_cols if c != "significant"]
    X = df[feature_cols].copy()

    # Por si quedara algún NaN
    X = X.fillna(0)

    print("Columnas usadas como features:", feature_cols)
    print("Tamaño de X:", X.shape)
    print("Tamaño de y:", y.shape)

    return X, y


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Modelo base: regresión logística.
    """
    print("\nEntrenando modelo base (Regresión Logística)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print("\nResultado modelo base (Regresión Logística)")
    print(classification_report(y_test, y_pred))

    return model, f1


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Modelo mejorado: Random Forest con parámetros moderados.
    """
    print("\nEntrenando modelo mejorado (Random Forest)...")
    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 60)
    mlflow.log_param("max_depth", 12)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print("\nResultado modelo mejorado (Random Forest)")
    print(classification_report(y_test, y_pred))

    return model, f1


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Iniciando flujo de entrenamiento...")
    X, y = load_data(DATA_PATH)

    print("\nDividiendo en train y test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Tamaño train:", X_train.shape, " - Tamaño test:", X_test.shape)

    mlflow.set_experiment("earthquakes_significant")

    best_model = None
    best_f1 = -1.0

    # Modelo base
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        model_lr, f1_lr = train_baseline_model(X_train, y_train, X_test, y_test)
        if f1_lr > best_f1:
            best_f1 = f1_lr
            best_model = model_lr

    # Modelo mejorado
    with mlflow.start_run(run_name="random_forest"):
        model_rf, f1_rf = train_random_forest(X_train, y_train, X_test, y_test)
        if f1_rf > best_f1:
            best_f1 = f1_rf
            best_model = model_rf

    # Guardamos el modelo campeón
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nMejor modelo guardado en: {MODEL_PATH}")
    print(f"Mejor F1: {best_f1:.4f}")
    print("Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()