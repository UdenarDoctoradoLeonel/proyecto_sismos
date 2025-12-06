# Clasificación de Sismos Significativos (1990–2023)

Este proyecto desarrolla un proceso completo de análisis, preparación de datos y modelación predictiva para determinar si un evento sísmico puede considerarse significativo a partir de variables físicas asociadas al fenómeno. El trabajo combina exploración de datos, construcción de un pipeline reproducible y entrenamiento de modelos supervisados. El resultado final se integra en una aplicación web interactiva.

---

## 1. Objetivo del proyecto

El propósito es entrenar un clasificador capaz de estimar la probabilidad de que un sismo pertenezca a la categoría “significativo”, utilizando mediciones disponibles en registros históricos. El proyecto está orientado a la comprensión del comportamiento de los datos y a la construcción de un flujo de trabajo reproducible y verificable.

---

## 2. Fuente de datos

Los datos proceden del conjunto público:

**The Ultimate Earthquake Dataset (1990–2023)**  
Disponibles en Kaggle:  
https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023

El archivo original fue depurado y transformado antes del entrenamiento del modelo.  
Las columnas empleadas finalmente como variables predictoras incluyen:

- magnitudo  
- tsunami  
- longitude  
- latitude  
- depth  
- time (en milisegundos Unix)  
- significance  

La etiqueta binaria `significant` se construyó a partir del indicador disponible en el dataset.

---

## 3. Metodología

El proceso seguido se divide en cuatro etapas principales:

### a. Exploración y comprensión de los datos  
Se revisó la distribución de magnitudes, profundidades, ubicación geográfica de los eventos y existencia de valores atípicos. También se estudió la relación entre magnitud, profundidad y significancia del sismo.

### b. Limpieza y preparación  
Se implementaron dos etapas automatizadas mediante DVC:
- **clean_data**: lectura del archivo original, normalización de columnas y eliminación de registros incompletos.
- **prepare_data**: selección de variables, transformación de marcas de tiempo y construcción de la variable objetivo.

### c. Entrenamiento del modelo  
Se entrenaron dos modelos de referencia:
- Regresión Logística (modelo base).  
- Random Forest (modelo final seleccionado).

La comparación se realizó a partir de las métricas de precisión y F1.  
El Random Forest obtuvo el mejor rendimiento y fue almacenado como modelo final.

### d. Implementación de una aplicación web  
Se construyó una interfaz que permite ingresar valores de un sismo hipotético y obtener la clasificación estimada por el modelo.

---

## 4. Arquitectura del pipeline

El proyecto utiliza **Git**, **DVC** y **MLflow**, permitiendo reproducibilidad completa:


---

## 5. Resultados y métrica principal

El modelo Random Forest alcanzó un desempeño superior en la clasificación, con valores altos de precisión y F1 en la clase minoritaria. Aunque la interpretación de un modelo de este tipo requiere cautela, su rendimiento permite usarlo como herramienta de apoyo exploratorio.

---

## 6. Aplicación web del proyecto

La aplicación está disponible en el siguiente enlace:

**https://proyectosismos-ikfskcmjs4gje3ajwbewc3.streamlit.app**

Permite ajustar los valores de magnitud, profundidad, ubicación, presencia de tsunami y nivel de significancia para evaluar la probabilidad de que el sismo sea clasificado como significativo.

---


---

## 5. Resultados y métrica principal

El modelo Random Forest alcanzó un desempeño superior en la clasificación, con valores altos de precisión y F1 en la clase minoritaria. Aunque la interpretación de un modelo de este tipo requiere cautela, su rendimiento permite usarlo como herramienta de apoyo exploratorio.

---

## 6. Cómo ejecutar el proyecto localmente

        # Crear y activar entorno virtual
        python3 -m venv .venv
        source .venv/bin/activate

        # Instalar dependencias
        pip install -r requirements.txt

        # Reproducir pipeline
        dvc repro

        # Ejecutar la aplicación
        streamlit run app/streamlit_app.py

## 7. Observaciones finales

- El modelo no sustituye el trabajo de análisis sísmico profesional; únicamente genera una estimación basada en patrones numéricos.

- La finalidad principal es académica y orientada al aprendizaje de técnicas de ciencia de datos y flujo reproducible de experimentos.

- La interpretación de los resultados debe hacerse con criterio técnico y atendiendo al contexto geológico particular.

## 8. Licencia: 

Este proyecto se distribuye con fines académicos y puede ser reutilizado citando la fuente original de los datos y este repositorio.

## 9. Autores

- **Leonel Delgado Eraso**  
- **Ayda Lucía Patiño Chaves**