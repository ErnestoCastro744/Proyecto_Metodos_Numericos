import subprocess
import sys
import importlib

def instalar(paquete):
    try:
        importlib.import_module(paquete)
    except ImportError:
        print(f"Instalando {paquete}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])
        print(f"{paquete} instalado correctamente.")

instalar("matplotlib")
instalar("numpy")

import math
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
import numpy as np

BG      = "#ffffff"
PANEL   = "#f5f5f5"
CARD    = "#ebebeb"
CARD2   = "#e0e0e0"
ACCENT  = "#4a3fcf"
ACCENT2 = "#e0005e"
TEXT    = "#1a1a2e"
SUBTEXT = "#555577"
SUCCESS = "#007a4d"
BORDER  = "#ccccdd"

rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   "#fafafa",
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  SUBTEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#ddddee",
    "grid.alpha":       0.6,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT
})

permitidas = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "ln": math.log, "log": math.log10,
    "pi": math.pi, "e": math.e, "abs": abs
}

def evaluar(expr):
    expr = expr.strip().lower().replace("^", "**")
    return eval(expr, {"__builtins__": {}}, permitidas)

def resolver_sistema_gauss(A, b):
    n = len(b)
    for i in range(n):
        mayor = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[mayor][i]):
                mayor = k
        A[i], A[mayor] = A[mayor], A[i]
        b[i], b[mayor] = b[mayor], b[i]
        if A[i][i] == 0:
            return None
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - suma) / A[i][i]
    return x

def spline_natural(x, y, valor):
    n = len(x)
    if n < 3:
        return None, None, None, None, "Se necesitan al menos 3 puntos."
    h = []
    for i in range(n - 1):
        hi = x[i + 1] - x[i]
        if hi == 0:
            return None, None, None, None, "No puede haber x repetidos."
        h.append(hi)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    A[0][0] = 1
    A[n - 1][n - 1] = 1
    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i]     = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 6 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    M = resolver_sistema_gauss(A, b)
    if M is None:
        return None, None, None, None, "No se pudo resolver."
    pos = -1
    for i in range(n - 1):
        if x[i] <= valor <= x[i + 1]:
            pos = i
            break
    if pos == -1:
        return None, None, None, None, "Valor fuera del rango."
    i  = pos
    x1 = x[i]; x2 = x[i+1]
    y1 = y[i]; y2 = y[i+1]
    hi = h[i]; M1 = M[i]; M2 = M[i+1]
    resultado = (
        M1*(x2-valor)**3/(6*hi) + M2*(valor-x1)**3/(6*hi)
        + (y1 - M1*hi**2/6)*(x2-valor)/hi
        + (y2 - M2*hi**2/6)*(valor-x1)/hi
    )
    datos = {
        "x": x, "y": y, "h": h, "M": M,
        "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        "hi": hi, "valor": valor, "resultado": resultado,
        "M1": M1, "M2": M2
    }
    return resultado, M, h, datos, None

def evaluar_spline(x_pts, y_pts, M, h, t):
    for i in range(len(x_pts) - 1):
        if x_pts[i] <= t <= x_pts[i + 1]:
            x1=x_pts[i]; x2=x_pts[i+1]
            y1=y_pts[i]; y2=y_pts[i+1]
            hi=h[i]; M1=M[i]; M2=M[i+1]
            return (
                M1*(x2-t)**3/(6*hi) + M2*(t-x1)**3/(6*hi)
                + (y1-M1*hi**2/6)*(x2-t)/hi
                + (y2-M2*hi**2/6)*(t-x1)/hi
            )
    return None

def mostrar_explicacion():
    ventana_exp = tk.Toplevel(ventana)
    ventana_exp.title("Explicación del Procedimiento")
    ventana_exp.geometry("1200x750")
    ventana_exp.configure(bg=BG)
    tk.Label(ventana_exp, text="Explicación Completa del Método",
             bg=BG, fg=ACCENT, font=("Segoe UI", 24, "bold")).pack(pady=(20,10))
    frame_principal = tk.Frame(ventana_exp, bg=BG)
    frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    canvas_exp = tk.Canvas(frame_principal, bg=BG, highlightthickness=0)
    scrollbar  = tk.Scrollbar(frame_principal, orient="vertical", command=canvas_exp.yview)
    canvas_exp.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_exp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    contenido      = tk.Frame(canvas_exp, bg=BG)
    ventana_canvas = canvas_exp.create_window((0,0), window=contenido, anchor="nw")
    canvas_exp.bind("<Configure>", lambda e: canvas_exp.itemconfig(ventana_canvas, width=e.width))
    contenido.bind("<Configure>", lambda e: canvas_exp.configure(scrollregion=canvas_exp.bbox("all")))
    canvas_exp.bind_all("<MouseWheel>", lambda e: canvas_exp.yview_scroll(int(-1*(e.delta/120)), "units"))

    def tarjeta(titulo, texto, color):
        card = tk.Frame(contenido, bg="#ffffff", highlightbackground=color, highlightthickness=1)
        card.pack(fill=tk.X, expand=True, padx=20, pady=10)
        header = tk.Frame(card, bg=color, height=40)
        header.pack(fill=tk.X)
        tk.Label(header, text="  "+titulo, bg=color, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)
        body = tk.Frame(card, bg="#ffffff", padx=25, pady=20)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(body, text=texto, bg="#ffffff", fg=TEXT, justify="left",
                 anchor="w", wraplength=1000, font=("Consolas", 11)).pack(anchor="w", fill=tk.X)

    tarjeta("📘 INTRODUCCIÓN", """
Un spline cúbico natural es un método de interpolación.
La interpolación estima valores intermedios entre puntos conocidos.
El spline construye una curva suave que pasa exactamente por todos los puntos.
Cada tramo usa un polinomio cúbico, lo que garantiza suavidad sin cambios bruscos.
""", "#4d8cff")
    tarjeta("△ DISTANCIAS h", """
h representa la distancia horizontal entre dos valores consecutivos de x.

Fórmula:   h = x(i+1) - x(i)

Estos valores construyen el sistema de ecuaciones para los momentos.
""", "#9c88ff")
    tarjeta("◇ MOMENTOS Mi", """
Los momentos Mi son las segundas derivadas del spline en cada nodo.
Indican la curvatura: Mi grande → más doblez; Mi pequeño → más suave.

Condición natural:   M0 = 0   y   Mn = 0
(curvatura cero en los extremos)
""", ACCENT2)
    tarjeta("🧮 MÉTODO DE GAUSS", """
Para obtener los momentos Mi se construye un sistema de ecuaciones
que se resuelve con eliminación de Gauss:

1. Organizar la matriz.
2. Eliminar valores bajo la diagonal.
3. Obtener matriz triangular.
4. Sustituir hacia atrás.
""", "#00aacc")
    tarjeta("📐 FÓRMULA DEL SPLINE", """
S(x) =  M1·(x2-x)³ / 6h
      + M2·(x-x1)³ / 6h
      + (y1 - M1·h²/6)·(x2-x)/h
      + (y2 - M2·h²/6)·(x-x1)/h

x1, x2 : extremos del intervalo
M1, M2 : momentos del intervalo
h      : distancia entre puntos
x      : valor a interpolar
""", "#007a4d")
    tarjeta("📚 FUNCIONES DISPONIBLES", """
sin(x)   cos(x)   tan(x)   sqrt(x)   ln(x)   log(x)   abs(x)
Constantes: pi,  e
Potencias:  2^3  ó  2**3

Ejemplos:  sin(pi/2),  cos(0),  sqrt(16),  ln(e),  3^2+1
Todas las trigonométricas trabajan en radianes.
""", "#cc7700")

    tk.Button(ventana_exp, text="Cerrar", command=ventana_exp.destroy,
              bg=ACCENT, fg="white", relief="flat",
              font=("Segoe UI", 11, "bold"), padx=25, pady=10,
              cursor="hand2").pack(pady=15)

def mostrar_procedimiento(d):
    for w in frame_proc.winfo_children():
        w.destroy()
    texto    = tk.Text(frame_proc, bg="#ffffff", fg=TEXT, font=("Consolas", 10),
                       wrap="word", relief="flat", padx=10, pady=10)
    scrollbar = tk.Scrollbar(frame_proc, command=texto.yview)
    texto.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    texto.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    contenido = "==============================\nPROCEDIMIENTO PASO A PASO\n==============================\n\nPUNTOS INGRESADOS\n\n"
    for i in range(len(d["x"])):
        contenido += f"P{i} = ({d['x'][i]}, {d['y'][i]})\n"
    contenido += "\nDISTANCIAS h\n\n"
    for i in range(len(d["h"])):
        contenido += f"h{i} = {d['x'][i+1]} - {d['x'][i]}\nh{i} = {d['h'][i]}\n\n"
    contenido += "MOMENTOS Mi\n\n"
    for i in range(len(d["M"])):
        contenido += f"M{i} = {round(d['M'][i], 6)}\n"
    contenido += f"""
VALOR A INTERPOLAR

x = {d['valor']}

INTERVALO USADO

[{d['x1']}, {d['x2']}]

DATOS DEL INTERVALO

x1 = {d['x1']}
x2 = {d['x2']}
y1 = {d['y1']}
y2 = {d['y2']}
h  = {d['hi']}
M1 = {d['M1']}
M2 = {d['M2']}

FORMULA

S(x) = M1(x2-x)^3 / 6h
     + M2(x-x1)^3 / 6h
     + (y1-M1h^2/6)(x2-x)/h
     + (y2-M2h^2/6)(x-x1)/h

RESULTADO FINAL

S({d['valor']}) = {round(d['resultado'], 6)}
"""
    texto.insert("1.0", contenido)
    texto.config(state="disabled")

def graficar(x_pts, y_pts, M, h, valor, resultado):
    fig.clf()
    colores = ["#4a3fcf", "#e0005e", "#007a4d", "#cc7700", "#0099bb"]
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(x_pts) - 1):
        xs = np.linspace(x_pts[i], x_pts[i+1], 200)
        ys = [evaluar_spline(x_pts, y_pts, M, h, t) for t in xs]
        c  = colores[i % len(colores)]
        ax.plot(xs, ys, color=c, lw=2.5, label=f"Tramo {i}  [{x_pts[i]}, {x_pts[i+1]}]")
        ax.fill_between(xs, ys, alpha=0.07, color=c)

    ax.scatter(x_pts, y_pts, s=90, color="#ffffff", edgecolors=ACCENT,
               linewidths=2, zorder=5, label="Nodos")
    for i in range(len(x_pts)):
        ax.annotate(f"({x_pts[i]}, {y_pts[i]})", xy=(x_pts[i], y_pts[i]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=SUBTEXT)

    ax.scatter([valor], [resultado], color=ACCENT2, s=220, marker="*", zorder=7,
               label=f"S({valor}) = {round(resultado, 4)}")
    ax.axvline(valor,     color=ACCENT2, ls="--", alpha=0.35)
    ax.axhline(resultado, color=ACCENT2, ls="--", alpha=0.35)

    ax.set_title("Spline Cúbico Natural", fontsize=16, fontweight="bold", color=TEXT)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout(pad=2.5)
    canvas.draw()

def calcular():
    try:
        texto = entrada_puntos.get("1.0", tk.END).strip()
        valor = float(evaluar(entrada_valor.get()))
        x, y  = [], []
        for fila in texto.split("\n"):
            datos = fila.split(",")
            if len(datos) != 2:
                messagebox.showerror("Error", "Use x,y por línea")
                return
            x.append(float(evaluar(datos[0])))
            y.append(float(evaluar(datos[1])))
        pares = sorted(zip(x, y))
        x = [p[0] for p in pares]
        y = [p[1] for p in pares]
        res, M, h, datos, err = spline_natural(x, y, valor)
        if err:
            messagebox.showerror("Error", err)
            return
        lbl_res.config(text=f"S({valor}) = {round(res, 6)}", fg=SUCCESS)
        mostrar_procedimiento(datos)
        graficar(x, y, M, h, valor, res)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def limpiar():
    entrada_puntos.delete("1.0", tk.END)
    entrada_valor.delete(0, tk.END)
    lbl_res.config(text="Ingrese datos y calcule", fg=SUBTEXT)
    for w in frame_proc.winfo_children():
        w.destroy()
    fig.clf()
    canvas.draw()

# ── ventana principal ────────────────────────────────────────────────────────
ventana = tk.Tk()
ventana.title("Spline Cubico Natural")
ventana.state("zoomed")
ventana.configure(bg=BG)

panel = tk.Frame(ventana, bg=PANEL, width=380)
panel.pack(side=tk.LEFT, fill=tk.Y)
panel.pack_propagate(False)

tk.Label(panel, text="Spline Cubico Natural", bg=PANEL, fg=ACCENT,
         font=("Segoe UI", 20, "bold")).pack(pady=(20, 10))

tk.Label(panel, text="Puntos (x,y)", bg=PANEL, fg=SUBTEXT,
         font=("Segoe UI", 10)).pack(anchor="w", padx=15)

entrada_puntos = tk.Text(panel, height=7, bg="#ffffff", fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=("Consolas", 11))
entrada_puntos.pack(fill=tk.X, padx=15, pady=5)
entrada_puntos.insert(tk.END, "3.0,-4.240058\n3.1,-3.496909\n3.2,-2.596792\n\n")

tk.Label(panel, text="Valor a interpolar", bg=PANEL, fg=SUBTEXT,
         font=("Segoe UI", 10)).pack(anchor="w", padx=15)

entrada_valor = tk.Entry(panel, bg="#ffffff", fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Consolas", 12))
entrada_valor.pack(fill=tk.X, padx=15, pady=5)
entrada_valor.insert(0, "3.1")

def crear_boton(texto, comando, color):
    return tk.Button(panel, text=texto, command=comando, bg=color, fg="white",
                     relief="flat", font=("Segoe UI", 11, "bold"), pady=10, cursor="hand2")

crear_boton("Calcular",                       calcular,            ACCENT).pack(fill=tk.X, padx=15, pady=5)
crear_boton("Limpiar",                        limpiar,             "#888899").pack(fill=tk.X, padx=15, pady=5)
crear_boton("Teoria",  mostrar_explicacion, "#5a4fcf").pack(fill=tk.X, padx=15, pady=5)

lbl_res = tk.Label(panel, text="Ingrese datos y calcule", bg=CARD, fg=SUBTEXT,
                   font=("Segoe UI", 11, "bold"), pady=12)
lbl_res.pack(fill=tk.X, padx=15, pady=10)

tk.Label(panel, text="Procedimiento paso a paso", bg=PANEL, fg=TEXT,
         font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15)

frame_proc = tk.Frame(panel, bg=PANEL)
frame_proc.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

frame_der = tk.Frame(ventana, bg=BG)
frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig    = plt.Figure(figsize=(9, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_der)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

ventana.mainloop()