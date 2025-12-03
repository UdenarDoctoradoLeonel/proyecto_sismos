import os
import pandas as pd


RAW_PATH = "data/raw/Eartquakes-1990-2023.csv"
CLEAN_DIR = "data/clean"
CLEAN_PATH = os.path.join(CLEAN_DIR, "earthquakes_clean.csv")


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # Nos quedamos con las columnas que vamos a usar
    cols = [
        "time",
        "place",
        "status",
        "tsunami",
        "significance",
        "data_type",
        "magnitudo",
        "state",
        "longitude",
        "latitude",
        "depth",
        "date",
    ]
    df = df[cols].copy()

    # Eliminamos filas con datos esenciales faltantes
    df = df.dropna(
        subset=["significance", "magnitudo", "longitude", "latitude", "depth"]
    )

    # Aseguramos tipos numéricos donde corresponde
    df["tsunami"] = pd.to_numeric(df["tsunami"], errors="coerce").fillna(0).astype(int)
    df["significance"] = pd.to_numeric(df["significance"], errors="coerce")
    df["magnitudo"] = pd.to_numeric(df["magnitudo"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")

    # Limpieza simple de valores extremos de profundidad
    df = df[(df["depth"] > -5) & (df["depth"] < 700)]

    # Guardamos el resultado
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Archivo limpio guardado en: {CLEAN_PATH}")


if __name__ == "__main__":
    main()