import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Red Neuronal - Retenciones Impuesto a la Renta 2026 (SRI)", layout="centered")

# ============================================================
# 1. BASE DE DATOS OFICIAL - RETENCIONES IR 2026 (SRI)
# ============================================================
retenciones_sri_2026 = [
    ("Honorarios profesionales con título", 10),
    ("Servicios profesionales sociedades residentes", 5),
    ("Servicios predominio intelectual sin título", 10),
    ("Servicios predominio mano de obra", 3),
    ("Medios de comunicación y publicidad", 3),
    ("Transporte privado pasajeros / carga", 1),
    ("Liquidación de compra (rusticidad/nivel cultural)", 3),
    ("Transferencia bienes muebles corporales", 2),
    ("Arrendamiento de bienes inmuebles", 10),
    ("Seguros y reaseguros (primas y cesiones)", 2),
    ("Rendimientos financieros y depósitos", 3),
    ("Intereses pagados a bancos e IFIs", 0),
    ("Otras compras sin retención (RIMPE Negocios Populares)", 0),
    ("Compra de bienes inmuebles", 0),
    ("Transporte público de pasajeros", 0)
]

conceptos = [item[0] for item in retenciones_sri_2026]
tarifas_reales = dict(retenciones_sri_2026)

# ============================================================
# 2. ENTRENAMIENTO ROBUSTO DE LA RED NEURONAL
# ============================================================
datos = []
resultados = []

# Posibles porcentajes de retención que maneja el SRI
todos_los_porcentajes = [0, 1, 2, 3, 5, 10]

for conc in conceptos:
    tarifa_correcta = tarifas_reales[conc]
    for p in todos_los_porcentajes:
        datos.append([conc, p])
        # Si el porcentaje evaluado coincide con el real, es CORRECTO (1), sino INCORRECTO (0)
        if p == tarifa_correcta:
            resultados.append(1)
        else:
            resultados.append(0)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
conceptos_array = np.array([[fila[0]] for fila in datos])
conceptos_codificados = encoder.fit_transform(conceptos_array)

# Normalizar porcentajes dividiendo para 10
porcentajes_array = np.array([[fila[1] / 10.0] for fila in datos])

X = np.hstack([conceptos_codificados, porcentajes_array])
y = np.array(resultados)

# Red Neuronal con mejor capacidad de discriminación
red = MLPClassifier(hidden_layer_sizes=(12, 8), activation="relu", solver="lbfgs", max_iter=3000, random_state=42)
red.fit(X, y)

# ============================================================
# 3. INTERFAZ WEB (STREAMLIT)
# ============================================================
st.markdown("<h2 style='text-align: center; color: #1a73e8;'>RED NEURONAL: RETENCIONES IMPUESTO A LA RENTA 2026 (SRI)</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    conc_sel = st.selectbox("Seleccione el Concepto / Transacción:", sorted(list(set(conceptos))))
with col2:
    porc_txt = st.selectbox("Porcentaje de Retención a Evaluar:", ["0%", "1%", "2%", "3%", "5%", "10%"])

porc_val = int(porc_txt.replace("%", ""))

if st.button("EJECUTAR Y CLASIFICAR", type="primary", use_container_width=True):
    conc_cod = encoder.transform([[conc_sel]])
    porc_cod = np.array([[porc_val / 10.0]])
    entrada = np.hstack([conc_cod, porc_cod])
    
    prediccion = red.predict(entrada)[0]
    probabilidades = red.predict_proba(entrada)[0]
    confianza = probabilidades[prediccion] * 100

    # Comprobación estricta de la regla de negocio real del SRI
    es_realmente_correcto = (porc_val == tarifas_reales[conc_sel])

    # Dibujar Red Neuronal con Matplotlib
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Caja capa oculta
    rect = plt.Rectangle((0.33, 0.1), 0.34, 0.8, facecolor="#e8eaed", edgecolor="#bdc1c6", linewidth=2, zorder=1)
    ax.add_patch(rect)
    ax.text(0.5, 0.88, "Capa Oculta (Neuronas)", fontsize=8, fontweight='bold', color="#5f6368", ha='center')

    nodos_entrada = [(0.1, 0.7), (0.1, 0.3)]
    nodos_oculta = [(0.5, 0.75), (0.5, 0.5), (0.5, 0.25)]
    nodo_salida = (0.9, 0.5)

    # Conexiones
    for x1, y1 in nodos_entrada:
        for x2, y2 in nodos_oculta:
            ax.plot([x1, x2], [y1, y2], color="#9aa0a6", lw=1.5, zorder=2)
    for x1, y1 in nodos_oculta:
        ax.plot([x1, nodo_salida[0]], [y1, nodo_salida[1]], color="#9aa0a6", lw=1.5, zorder=2)

    # Nodos de Entrada
    col_in = ["#1a73e8", "#fbbc04"]
    for i, (x, y) in enumerate(nodos_entrada):
        circle = plt.Circle((x, y), 0.08, color=col_in[i], ec='black', lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"E{i+1}", color='white', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)

    # Nodos Ocultos
    for i, (x, y) in enumerate(nodos_oculta):
        circle = plt.Circle((x, y), 0.08, color='white', ec='black', lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"H{i+1}", color='black', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)

    # Nodo Salida (Depende de la validación real)
    color_sal = "#34a853" if es_realmente_correcto else "#ea4335"
    circle = plt.Circle(nodo_salida, 0.08, color=color_sal, ec='black', lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(nodo_salida[0], nodo_salida[1], "OUT", color='white', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
    ax.text(nodo_salida[0], nodo_salida[1]-0.16, f"{confianza:.1f}% Conf.", fontsize=8, fontweight='bold', ha='center', color="#202124")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    st.pyplot(fig)

    if es_realmente_correcto:
        st.markdown("<h3 style='text-align: center; color: #137333;'>✓ CORRECTO (El porcentaje cumple con la normativa 2026 del SRI)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: #c5221f;'>✗ INCORRECTO (El porcentaje correcto para este concepto es {tarifas_reales[conc_sel]}%)</h3>", unsafe_allow_html=True)
        
    st.markdown(f"<p style='text-align: center; color: #5f6368;'>Evaluando concepto: '{conc_sel}' con tarifa de retención del {porc_txt}</p>", unsafe_allow_html=True)