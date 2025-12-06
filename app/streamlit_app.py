import streamlit as st

st.set_page_config(page_title="Prueba despliegue", page_icon="🌎")

st.title("Prueba de despliegue en Streamlit Cloud")
st.write(
    """
Esta es una prueba mínima para verificar que el entorno de Streamlit Cloud
funciona correctamente con Python 3.13 y la versión de Streamlit instalada.
Si puedes ver esta página, el problema NO es de infraestructura,
sino de las dependencias adicionales (pandas, numpy, scikit-learn, etc.).
"""
)