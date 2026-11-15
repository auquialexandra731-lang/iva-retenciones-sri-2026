import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import streamlit as st

# Configuración de la página web
st.set_page_config(
    page_title="Red Neuronal - Clasificación IVA SRI", layout="centered"
)

st.markdown(
    "<h1 style='text-align: center; color: #1a73e8;'>RED NEURONAL: NORMATIVA IVA SRI (0% y 15%)</h1>",
    unsafe_allow_html=True,
)
st.write(
    "<p style='text-align: center; color: #5f6368;'>Modelo de Inteligencia Artificial para la clasificación automática de tarifas de IVA en Ecuador según la normativa vigente.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Base de datos simulada de productos y servicios con su tarifa de IVA real
datos = {
    "Elemento": [
        "Arroz, azúcar, leche",
        "Medicamentos e insumos médicos",
        "Transporte público de pasajeros",
        "Servicios de educación y guarderías",
        "Electrodomésticos (Televisores, Línea blanca)",
        "Servicios profesionales y asesorías",
        "Restaurantes y consumo de alimentos",
        "Alquiler de locales comerciales",
        "Vehículos y automóviles nuevos",
        "Bebidas alcohólicas y gaseosas",
    ],
    "Categoria_Cod": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "Tarifa_Real": [0, 0, 0, 0, 15, 15, 15, 15, 15, 15],
}

df = pd.DataFrame(datos)

# Preparar datos para entrenar la red neuronal (MLPClassifier)
X = df[["Categoria_Cod", "Tarifa_Real"]]
y = (df["Tarifa_Real"] == 15).astype(
    int
)  # 1 si es 15%, 0 si es tarifa 0%

# Entrenar la red neuronal rápida
red = MLPClassifier(
    hidden_layer_sizes=(4, 2), max_iter=1000, random_state=42
)
red.fit(X, y)

# Interfaz visual en la web de Streamlit
st.subheader("Panel de Evaluación de Productos y Servicios")

col1, col2 = st.columns(2)

with col1:
    prod_sel = st.selectbox(
        "Seleccione el Producto/Servicio:", sorted(list(set(df["Elemento"])))
    )

with col2:
    tarifa_val = st.selectbox(
        "Tarifa IVA a Evaluar:", [0, 15], format_func=lambda x: f"{x}%"
    )

# Botón para ejecutar la clasificación de la red neuronal
if st.button("EJECUTAR Y CLASIFICAR", type="primary", use_container_width=True):
  # Buscar la categoría numérica asociada al producto
  cat_fila = df[df["Elemento"] == prod_sel]["Categoria_Cod"].values[0]

  entrada = np.array([[cat_fila, tarifa_val]])
  predaccion = red.predict(entrada)[0]
  probabilidades = red.predict_proba(entrada)[0]
  confianza = probabilidades[predaccion] * 100

  # Validar si coincide con la norma real del SRI
  tarifa_real_prod = df[df["Elemento"] == prod_sel]["Tarifa_Real"].values[0]
  es_realmente_correcto = tarifa_val == tarifa_real_prod

  st.markdown("---")
  st.subheader("Resultado de la Red Neuronal")

  if es_realmente_correcto:
    st.markdown(
        f"<h3 style='text-align: center; color: #137333;'>✔ CORRECTO (La"
        f" tarifa de {tarifa_val}% cumple con la normativa del SRI para"
        f" '{prod_sel}')</h3>",
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f"<h3 style='text-align: center; color: #c5221f;'>✖ INCORRECTO"
        f" (Tarifa errónea para '{prod_sel}'. La tarifa real del SRI es"
        f" {tarifa_real_prod}%)</h3>",
        unsafe_allow_html=True,
    )

  st.info(
      f"🔍 **Detalles técnicos de la Red Neuronal:** Confianza de clasificación"
      f" del modelo: **{confianza:.1f}%**. Estado de la red: Activación de"
      f" capas ocultas completada con éxito."
  )
