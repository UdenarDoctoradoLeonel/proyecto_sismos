import os
import pickle
from datetime import datetime, time

import pandas as pd
import streamlit as st

# Ruta al modelo entrenado
MODEL_PATH = "models/model.pkl"

# Columnas que usa el modelo (las mismas que en el entrenamiento)
FEATURE_COLS = [
    "magnitudo",
    "tsunami",
    "longitude",
    "latitude",
    "depth",
    "time",
    "significance",
]


@st.cache_resource
def load_model(path: str):
    """
    Carga el modelo entrenado desde disco.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de modelo en: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def construir_registro(
    magnitudo: float,
    tsunami: int,
    longitud: float,
    latitud: float,
    profundidad: float,
    fecha_hora: datetime,
    significance: float,
) -> pd.DataFrame:
    """
    Construye un DataFrame de una sola fila con las columnas en el
    mismo orden que se usaron durante el entrenamiento.
    """

    # El dataset original almacena el tiempo como timestamp en milisegundos
    timestamp_ms = int(fecha_hora.timestamp() * 1000)

    data = {
        "magnitudo": [magnitudo],
        "tsunami": [tsunami],
        "longitude": [longitud],
        "latitude": [latitud],
        "depth": [profundidad],
        "time": [timestamp_ms],
        "significance": [significance],
    }

    df = pd.DataFrame(data, columns=FEATURE_COLS)
    return df


def main():
    st.set_page_config(
        page_title="Clasificación de Sismos Significativos",
        layout="centered",
    )

    st.title("Clasificación de Sismos Significativos")

    st.write(
        """
        Esta aplicación permite estimar si un evento sísmico puede considerarse
        **significativo** a partir de un modelo de clasificación entrenado con
        registros históricos de sismos entre 1990 y 2023.

        La idea no es reemplazar el análisis de un especialista, sino ofrecer
        una herramienta de apoyo basada en datos para explorar distintos
        escenarios de magnitud, profundidad, ubicación y nivel de significancia.
        """
    )

    # Cargar el modelo
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.subheader("1. Ingreso de la información del evento sísmico")

    col1, col2 = st.columns(2)

    with col1:
        magnitudo = st.slider(
            "Magnitud",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Valor de magnitud del sismo.",
        )

        profundidad = st.slider(
            "Profundidad (km)",
            min_value=0.0,
            max_value=700.0,
            value=20.0,
            step=1.0,
            help="Profundidad aproximada del hipocentro.",
        )

        tsunami_opcion = st.selectbox(
            "¿El evento estuvo asociado a un tsunami?",
            options=["No", "Sí"],
            index=0,
            help="Seleccione 'Sí' solo si existe evidencia de tsunami asociado al evento.",
        )
        tsunami = 1 if tsunami_opcion == "Sí" else 0

    with col2:
        latitud = st.slider(
            "Latitud",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=0.1,
        )
        longitud = st.slider(
            "Longitud",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.1,
        )

        st.markdown("**Fecha y hora del sismo (UTC)**")
        fecha = st.date_input(
            "Fecha",
            value=datetime(2015, 1, 1).date(),
        )
        hora = st.time_input(
            "Hora",
            value=time(0, 0),
        )

    fecha_hora = datetime.combine(fecha, hora)

    st.subheader("2. Índice de significancia del evento")

    st.write(
        """
        El conjunto de datos original incluye un índice numérico de significancia
        del evento. En esta versión de la aplicación, este valor se ingresa de forma
        aproximada para explorar distintos escenarios y observar cómo responde el modelo.
        """
    )

    significance = st.slider(
        "Índice de significancia (valor de referencia)",
        min_value=0.0,
        max_value=2000.0,
        value=100.0,
        step=10.0,
    )

    # Construimos el registro que se enviará al modelo
    entrada_df = construir_registro(
        magnitudo=magnitudo,
        tsunami=tsunami,
        longitud=longitud,
        latitud=latitud,
        profundidad=profundidad,
        fecha_hora=fecha_hora,
        significance=significance,
    )

    st.subheader("3. Datos que se envían al modelo")
    st.write(
        """
        A continuación se muestra la fila que se construye internamente para
        alimentar el modelo de clasificación. Esto permite verificar que las
        variables se están interpretando correctamente.
        """
    )
    st.dataframe(entrada_df)

    st.subheader("4. Resultado de la clasificación")

    if st.button("Calcular clasificación"):
        try:
            prob_significativo = model.predict_proba(entrada_df)[0, 1]
            prediccion = int(model.predict(entrada_df)[0])
        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {e}")
            st.stop()

        if prediccion == 1:
            st.success(
                f"Según el modelo, el evento se clasifica como **SIGNIFICATIVO**.\n\n"
                f"Probabilidad estimada de la clase significativa: {prob_significativo:.2%}."
            )
        else:
            st.info(
                f"Según el modelo, el evento se clasifica como **NO significativo**.\n\n"
                f"Probabilidad estimada de la clase significativa: {prob_significativo:.2%}."
            )

        st.write(
            """
            Este resultado debe interpretarse como una estimación probabilística
            basada en patrones presentes en el conjunto de datos utilizado para
            el entrenamiento. No reemplaza el análisis detallado que puede hacer
            un especialista en sismología, pero ofrece una referencia útil para
            la toma de decisiones y la exploración de escenarios.
            """
        )

        st.write(
            f"\nProbabilidad numérica (clase significativa = 1): {prob_significativo:.4f}"
        )


if __name__ == "__main__":
    main()