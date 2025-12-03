import os
import pandas as pd

CLEAN_PATH = "data/clean/earthquakes_clean.csv"
PREPARED_DIR = "data/clean"
PREPARED_PATH = os.path.join(PREPARED_DIR, "earthquakes_prepared.csv")

def main():
    os.makedirs(PREPARED_DIR, exist_ok=True)

    df = pd.read_csv(CLEAN_PATH)

    # Variable objetivo
    df["significant"] = (df["significance"] >= 100).astype(int)

    # Usamos valores numéricos principales
    numeric_features = ["magnitudo", "tsunami", "longitude", "latitude", "depth"]

    # Simplificamos la columna 'state': solo los 20 más frecuentes
    if "state" in df.columns:
        top_states = df["state"].value_counts().head(20).index
        df["state_simple"] = df["state"].apply(lambda x: x if x in top_states else "other")
        df = df.drop(columns=["state"])
        df = pd.get_dummies(df, columns=["state_simple"], drop_first=True)

    # Creamos dataset final con y y las columnas numéricas
    base_cols = numeric_features + ["significant"]
    other_cols = [c for c in df.columns if c not in base_cols]
    final_cols = numeric_features + other_cols + ["significant"]

    df = df[final_cols]

    df.to_csv(PREPARED_PATH, index=False)
    print(f"Archivo preparado guardado en: {PREPARED_PATH}")

if __name__ == "__main__":
    main()