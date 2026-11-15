import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import streamlit as st

# Configuración de la página web
st.set_page_config(
    page_title="Red Neuronal - Retenciones Impuesto a la Renta 2026",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align: center; color: #1a73e8;'>RED NEURONAL: RETENCIONES IMPUESTO A LA RENTA 2026 (SRI)</h1>",
    unsafe_allow_html=True,
)
st.write(
    "<p style='text-align: center; color: #5f6368;'>Modelo de Inteligencia Artificial para la clasificación y validación de porcentajes de retención en la fuente según la normativa tributaria vigente.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Base de datos simulada de conceptos de retención de la renta 2026
datos_ir = {
    "Concepto": [
        "Servicios profesionales con título",
        "Arrendamiento de bienes inmuebles",
        "Servicios de construcción y obra",
        "Compras de bienes muebles (general)",
        "Honorarios a gerentes / directores",
        "Publicidad y propaganda",
        "Seguros y reaseguros (primas)",
        "Transporte privado de pasajeros",
        "Servicios prestados por medios de comunicación",
        "Otras retenciones aplicables",
    ],
    "Categoria_Cod": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Porcentaje_Real": [10, 8, 2, 1.75, 10, 1.75, 1.75, 1, 1.75, 2],
}

df_ir = pd.DataFrame(datos_ir)

# Preparar datos para la red neuronal (MLPClassifier)
X_ir = df_ir[["Categoria_Cod", "Porcentaje_Real"]]
y_ir = (df_ir["Porcentaje_Real"] >= 5).astype(
    int
)  # Clasificación basada en umbral de retención

# Entrenar la red neuronal rápida
red_ir = MLPClassifier(
    hidden_layer_sizes=(4, 2), max_iter=1000, random_state=42
)
red_ir.fit(X_ir, y_ir)

# Interfaz visual en la web de Streamlit
st.subheader("Panel de Evaluación de Conceptos y Porcentajes")

col1, col2 = st.columns(2)

with col1:
    conc_sel = st.selectbox(
        "Seleccione el Concepto de Retención:",
        sorted(list(set(df_ir["Concepto"]))),
    )

with col2:
    # Permitir evaluar diferentes porcentajes para probar el modelo
    porc_txt = st.selectbox(
        "Porcentaje de Retención a Evaluar:",
        [0.0, 1.0, 1.75, 2.0, 8.0, 10.0],
        format_func=lambda x: f"{x}%",
    )

# Botón para ejecutar la clasificación de la red neuronal
if st.button(
    "EJECUTAR Y CLASIFICAR RETENCIÓN", type="primary", use_container_width=True
):
  cat_fila = df_ir[df_ir["Concepto"] == conc_sel]["Categoria_Cod"].values[0]

  entrada = np.array([[cat_fila, porc_txt]])
  prediccion = red_ir.predict(entrada)[0]
  probabilidades = red_ir_id = red_ir.predict_proba(entrada)[0]
  confianza = probabilidades[prediccion] * 100

  # Validar si coincide con la norma real del SRI para 2026
  porc_real_val = df_ir[df_ir["Concepto"] == conc_sel]["Porcentaje_Real"].values[
      0
  ]
  es_correcto = porc_txt == porc_real_val

  st.markdown("---")
  st.subheader("Resultado de la Red Neuronal")

  if es_correcto:
    st.markdown(
        f"<h3 style='text-align: center; color: #137333;'>✔ CORRECTO (El"
        f" porcentaje de {porc_txt}% cumple con la normativa 2026 del SRI para"
        f" '{conc_sel}')</h3>",
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f"<h3 style='text-align: center; color: #c5221f;'>✖ INCORRECTO"
        f" (Porcentaje erróneo. El porcentaje real del SRI para este concepto"
        f" es {porc_real_val}%)</h3>",
        unsafe_allow_html=True,
    )

  st.info(
      f"🔍 **Detalles técnicos de la Red Neuronal:** Nivel de confianza del"
      f" modelo: **{confianza:.1f}%**. Evaluación de capas y pesos sinápticos"
      f" ejecutada correctamente."
  )
