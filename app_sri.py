import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# 1. BASE DE DATOS OFICIAL (NORMATIVA SRI - ECUADOR)
# ============================================================
# Clasificación basada en la LORTI (Art. 55 y 56) y guías del SRI.
elementos_sri = [
    # --- TARIFA 0% ---
    ("manzana natural", 0),
    ("arroz en estado natural", 0),
    ("leche en polvo nacional", 0),
    ("leche pasteurizada nacional simple", 0),
    ("queso fresco", 0),
    ("pan común (NTE INEN 2945)", 0),
    ("atún enlatado nacional", 0),
    ("medicamentos de uso humano", 0),
    ("papel bond y libros", 0),
    ("transporte público terrestre", 0),
    ("servicios de educación", 0),
    ("servicios de salud", 0),
    
    # --- TARIFA 15% ---
    ("jugo de fruta procesado", 15),
    ("leche saborizada o importada", 15),
    ("queso maduro procesado", 15),
    ("pan de pascua o pastel", 15),
    ("galletas y golosinas", 15),
    ("atún enlatado importado", 15),
    ("bebidas gaseosas y energizantes", 15),
    ("teléfonos celulares y computadoras", 15),
    ("prendas de vestir y calzado", 15),
    ("servicios profesionales (abogados/ingenieros)", 15),
    ("arrendamiento de local comercial", 15),
    ("restaurantes y hoteles", 15)
]

nombres = [item[0] for item in elementos_sri]
tarifas_reales = [item[1] for item in elementos_sri]

# ============================================================
# 2. ENTRENAMIENTO DE LA RED NEURONAL
# ============================================================
datos = []
resultados = []

for elem, tarifa_corr in zip(nombres, tarifas_reales):
    # Caso 1: Combinación CORRECTA (Salida 1)
    datos.append([elem, tarifa_corr])
    resultados.append(1)
    
    # Caso 2: Combinación INCORRECTA (Salida 0)
    tarifa_inc = 15 if tarifa_corr == 0 else 0
    datos.append([elem, tarifa_inc])
    resultados.append(0)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
nombres_array = np.array([[fila[0]] for fila in datos])
elementos_codificados = encoder.fit_transform(nombres_array)

tarifas_array = np.array([[fila[1] / 15] for fila in datos])

X = np.hstack([elementos_codificados, tarifas_array])
y = np.array(resultados)

# Red Neuronal con capas ocultas (8, 5)
red = MLPClassifier(hidden_layer_sizes=(8, 5), activation="relu", solver="lbfgs", max_iter=2000, random_state=42)
red.fit(X, y)

# ============================================================
# 3. INTERFAZ GRÁFICA (VENTANA DE WINDOWS)
# ============================================================
ventana = tk.Tk()
ventana.title("Red Neuronal - Clasificación IVA SRI (Ecuador)")
ventana.geometry("900x700")
ventana.configure(bg="white")

titulo = tk.Label(ventana, text="RED NEURONAL: NORMATIVA IVA SRI (0% y 15%)", font=("Arial", 16, "bold"), bg="white", fg="#1a73e8")
titulo.pack(pady=15)

# Marco de Selección
marco_inputs = tk.Frame(ventana, bg="white")
marco_inputs.pack(pady=5)

tk.Label(marco_inputs, text="Seleccione el Producto/Servicio:", font=("Arial", 11, "bold"), bg="white").grid(row=0, column=0, padx=10, sticky="w")
combo_elemento = ttk.Combobox(marco_inputs, values=sorted(list(set(nombres))), state="readonly", width=38, font=("Arial", 11))
combo_elemento.grid(row=0, column=1, padx=10)
if nombres: combo_elemento.current(0)

tk.Label(marco_inputs, text="Tarifa IVA a Evaluar:", font=("Arial", 11, "bold"), bg="white").grid(row=1, column=0, padx=10, pady=10, sticky="w")
combo_tarifa = ttk.Combobox(marco_inputs, values=["0%", "15%"], state="readonly", width=38, font=("Arial", 11))
combo_tarifa.grid(row=1, column=1, padx=10, pady=10)
combo_tarifa.current(0)

# Lienzo para dibujar la Red Neuronal visualmente
canvas = tk.Canvas(ventana, width=820, height=220, bg="#f8f9fa", highlightthickness=1, highlightbackground="#dadce0")
canvas.pack(pady=10)

etiqueta_resultado = tk.Label(ventana, text="", font=("Arial", 15, "bold"), bg="white")
etiqueta_resultado.pack(pady=5)

etiqueta_detalle = tk.Label(ventana, text="", font=("Arial", 10), bg="white", justify="center")
etiqueta_detalle.pack(pady=5)

def evaluar_y_dibujar():
    elem_sel = combo_elemento.get()
    tarifa_txt = combo_tarifa.get()
    
    if not elem_sel or not tarifa_txt:
        messagebox.showwarning("Aviso", "Seleccione un elemento y una tarifa.")
        return
        
    tarifa_val = int(tarifa_txt.replace("%", ""))
    
    # Procesar predicción en la red neuronal
    elem_cod = encoder.transform([[elem_sel]])
    tarifa_cod = np.array([[tarifa_val / 15]])
    entrada = np.hstack([elem_cod, tarifa_cod])
    
    prediccion = red.predict(entrada)[0]
    probabilidades = red.predict_proba(entrada)[0]
    confianza = probabilidades[prediccion] * 100
    
    # Dibujar Red Neuronal en el Canvas
    canvas.delete("all")
    
    # Capa Oculta (Caja de fondo gris)
    canvas.create_rectangle(340, 15, 480, 205, fill="#e8eaed", outline="#bdc1c6", width=2)
    canvas.create_text(410, 25, text="Capa Oculta (Neuronas)", font=("Arial", 8, "bold"), fill="#5f6368")
    
    # Coordenadas de los nodos
    nodos_entrada = [(80, 60), (80, 140)]
    nodos_oculta = [(410, 60), (410, 110), (410, 160)]
    nodo_salida = (740, 110)
    
    # Dibujar Líneas de Conexión
    for x1, y1 in nodos_entrada:
        for x2, y2 in nodos_oculta:
            canvas.create_line(x1+25, y1, x2-25, y2, fill="#9aa0a6", width=1.5)
            
    for x1, y1 in nodos_oculta:
        canvas.create_line(x1+25, y1, nodo_salida[0]-25, nodo_salida[1], fill="#9aa0a6", width=1.5)
        
    # Dibujar Nodos de Entrada
    col_in = ["#1a73e8", "#fbbc04"]
    val_in_txt = ["Elemento", f"{tarifa_val}% IVA"]
    for i, (x, y) in enumerate(nodos_entrada):
        canvas.create_oval(x-25, y-25, x+25, y+25, fill=col_in[i], outline="black", width=2)
        canvas.create_text(x, y, text=f"E{i+1}", fill="white", font=("Arial", 9, "bold"))
        canvas.create_text(x, y+35, text=val_in_txt[i], fill="#202124", font=("Arial", 8, "bold"))
        
    # Dibujar Nodos de Capa Oculta
    for i, (x, y) in enumerate(nodos_oculta):
        canvas.create_oval(x-25, y-25, x+25, y+25, fill="#ffffff", outline="black", width=2)
        canvas.create_text(x, y, text=f"H{i+1}", fill="black", font=("Arial", 9, "bold"))
        
    # Dibujar Nodo de Salida
    color_salida = "#34a853" if prediccion == 1 else "#ea4335"
    sx, sy = nodo_salida
    canvas.create_oval(sx-25, sy-25, sx+25, sy+25, fill=color_salida, outline="black", width=2)
    canvas.create_text(sx, sy, text="OUT", fill="white", font=("Arial", 9, "bold"))
    canvas.create_text(sx, sy+35, text=f"{confianza:.1f}% Conf.", fill="#202124", font=("Arial", 8, "bold"))
    
    # Textos de Resultado basados en la normativa
    if prediccion == 1:
        etiqueta_resultado.config(text="✓ CORRECTO (La tarifa cumple con el SRI)", fg="#137333")
    else:
        etiqueta_resultado.config(text="✗ INCORRECTO (Tarifa errónea según ley del SRI)", fg="#c5221f")
        
    etiqueta_detalle.config(text=f"Evaluando: '{elem_sel}' asignado a la tarifa del {tarifa_txt}")

# Botón de ejecución
boton = tk.Button(ventana, text="EJECUTAR Y CLASIFICAR", command=evaluar_y_dibujar, bg="#1a73e8", fg="white", font=("Arial", 11, "bold"), padx=15, pady=8)
boton.pack(pady=10)

ventana.mainloop()