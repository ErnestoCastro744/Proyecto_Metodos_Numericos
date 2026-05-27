import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# ─────────────────────────────────────────────
#  RESOLUCIÓN DEL SISTEMA (Gauss con pivoteo)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  ALGORITMO PRINCIPAL
# ─────────────────────────────────────────────
def spline_natural(x, y, valor):
    n = len(x)
    intervalos = n - 1

    if n < 3:
        return None, None, None, "Se necesitan al menos 3 puntos."

    h = []
    for i in range(intervalos):
        hi = x[i + 1] - x[i]
        if hi == 0:
            return None, None, None, "No puede haber valores de x repetidos."
        h.append(hi)

    # Construir sistema tridiagonal
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    # Condición natural: M₀ = 0, Mₙ = 0
    A[0][0] = 1.0
    b[0] = 0.0
    A[n - 1][n - 1] = 1.0
    b[n - 1] = 0.0

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i]     = 2.0 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 6.0 * (
            (y[i + 1] - y[i]) / h[i] -
            (y[i]     - y[i - 1]) / h[i - 1]
        )

    M = resolver_sistema_gauss(A, b)
    if M is None:
        return None, None, None, "No se pudo resolver el sistema."

    # Localizar el valor en su intervalo
    posicion = -1
    for i in range(intervalos):
        if x[i] <= valor <= x[i + 1]:
            posicion = i
            break
    if posicion == -1:
        return None, None, None, "El valor está fuera del rango."

    # Evaluar S(valor)
    i   = posicion
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    h_i = h[i]
    M1, M2 = M[i], M[i + 1]

    t1 = M1 * (x2 - valor) ** 3 / (6 * h_i)
    t2 = M2 * (valor - x1) ** 3 / (6 * h_i)
    t3 = (y1 - M1 * h_i ** 2 / 6) * (x2 - valor) / h_i
    t4 = (y2 - M2 * h_i ** 2 / 6) * (valor - x1) / h_i
    resultado = t1 + t2 + t3 + t4

    # Texto de procedimiento
    proc = "SPLINE CÚBICO NATURAL\n"
    proc += "Condición natural: S''(x₀)=0  y  S''(xₙ)=0\n\n"
    proc += "Puntos:\n"
    for k in range(n):
        proc += f"  ({x[k]}, {y[k]})\n"
    proc += "\nValores h:\n"
    for k in range(intervalos):
        proc += f"  h{k} = {round(h[k], 4)}\n"
    proc += "\nSegundas derivadas M (momentos):\n"
    for k in range(n):
        proc += f"  M{k} = {round(M[k], 6)}\n"
    proc += f"\nIntervalo usado: [{x1}, {x2}]\n"
    proc += f"  y1={y1},  y2={y2},  h={h_i}\n"
    proc += f"  M1={round(M1,4)},  M2={round(M2,4)}\n"
    proc += "\nFórmula:\n"
    proc += "  S(x) = M1(x2-x)³/6h  +  M2(x-x1)³/6h\n"
    proc += "       + (y1-M1h²/6)(x2-x)/h\n"
    proc += "       + (y2-M2h²/6)(x-x1)/h\n"
    proc += f"\nResultado:\n  S({valor}) ≈ {round(resultado, 6)}"

    return resultado, M, h, proc


# ─────────────────────────────────────────────
#  FUNCIÓN DE EVALUACIÓN (para graficar)
# ─────────────────────────────────────────────
def evaluar_spline(x_pts, y_pts, M, h, t):
    """Evalúa el spline en un escalar t."""
    n = len(x_pts)
    for i in range(n - 1):
        if x_pts[i] <= t <= x_pts[i + 1]:
            x1, x2 = x_pts[i], x_pts[i + 1]
            y1, y2 = y_pts[i], y_pts[i + 1]
            hi = h[i]
            M1, M2 = M[i], M[i + 1]
            t1 = M1 * (x2 - t) ** 3 / (6 * hi)
            t2 = M2 * (t - x1) ** 3 / (6 * hi)
            t3 = (y1 - M1 * hi ** 2 / 6) * (x2 - t) / hi
            t4 = (y2 - M2 * hi ** 2 / 6) * (t - x1) / hi
            return t1 + t2 + t3 + t4
    return None


# ─────────────────────────────────────────────
#  ACCIÓN DEL BOTÓN CALCULAR
# ─────────────────────────────────────────────
def calcular():
    try:
        texto = entrada_puntos.get("1.0", tk.END).strip()
        valor = float(entrada_valor.get())

        x, y = [], []
        for fila in texto.split("\n"):
            datos = fila.strip().split(",")
            if len(datos) != 2:
                messagebox.showerror("Error", "Formato incorrecto. Use x,y por línea.")
                return
            x.append(float(datos[0]))
            y.append(float(datos[1]))

        if len(x) < 3:
            messagebox.showerror("Error", "Se necesitan al menos 3 puntos.")
            return

        # Ordenar por x
        pares = sorted(zip(x, y))
        x = [p[0] for p in pares]
        y = [p[1] for p in pares]

        for i in range(len(x) - 1):
            if x[i] == x[i + 1]:
                messagebox.showerror("Error", "No puede haber x repetidos.")
                return

        resultado, M, h, procedimiento = spline_natural(x, y, valor)

        if resultado is None:
            messagebox.showerror("Error", procedimiento)
            return

        # Mostrar procedimiento
        salida.config(state="normal")
        salida.delete("1.0", tk.END)
        salida.insert(tk.END, procedimiento)
        salida.config(state="disabled")

        # Generar gráficas
        graficar(x, y, M, h, valor, resultado)

    except ValueError:
        messagebox.showerror("Error", "Use números válidos.")


# ─────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────
def graficar(x_pts, y_pts, M, h, valor, resultado):
    fig.clf()
    n = len(x_pts)

    # ── Gráfica 1: Spline completo ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor("#f8f9fa")

    # Curva spline por tramos
    colores_tramo = plt.cm.tab10(np.linspace(0, 0.8, n - 1))
    for i in range(n - 1):
        xs = np.linspace(x_pts[i], x_pts[i + 1], 120)
        ys = [evaluar_spline(x_pts, y_pts, M, h, t) for t in xs]
        ax1.plot(xs, ys, color=colores_tramo[i], lw=2.2,
                 label=f"Tramo {i}: [{x_pts[i]}, {x_pts[i+1]}]")

    # Puntos de control
    ax1.scatter(x_pts, y_pts, color="black", zorder=5, s=70, label="Nodos")
    for k in range(n):
        ax1.annotate(f"({x_pts[k]}, {y_pts[k]})",
                     xy=(x_pts[k], y_pts[k]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8, color="#333333")

    # Punto interpolado
    ax1.scatter([valor], [resultado], color="crimson", zorder=6,
                s=110, marker="*", label=f"S({valor}) = {round(resultado, 4)}")
    ax1.axvline(valor, color="crimson", lw=0.8, ls="--", alpha=0.5)
    ax1.axhline(resultado, color="crimson", lw=0.8, ls="--", alpha=0.5)

    ax1.set_title("Spline cúbico natural por tramos", fontsize=11, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=7.5, loc="best")
    ax1.grid(True, alpha=0.35)

    # ── Gráfica 2: Segundas derivadas (momentos M) ──────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#f8f9fa")

    indices = list(range(n))
    colores_m = ["#2196F3" if abs(m) < 1e-10 else "#E91E63" for m in M]
    bars = ax2.bar(indices, M, color=colores_m, edgecolor="white", width=0.55)

    for bar, val in zip(bars, M):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.02 * (max(M) - min(M) + 0.001)),
                 f"{round(val, 4)}",
                 ha="center", va="bottom", fontsize=8.5)

    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f"M{i}\n(x={x_pts[i]})" for i in indices], fontsize=8)
    ax2.set_title("Segundas derivadas Mᵢ (momentos)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Valor de M")
    ax2.set_xlabel("Nodo i")

    patch_nat = mpatches.Patch(color="#2196F3", label="M = 0 (condición natural)")
    patch_int = mpatches.Patch(color="#E91E63", label="M ≠ 0 (nodo interior)")
    ax2.legend(handles=[patch_nat, patch_int], fontsize=8)
    ax2.grid(axis="y", alpha=0.35)

    fig.tight_layout(pad=2.5)
    canvas.draw()


# ─────────────────────────────────────────────
#  LIMPIAR
# ─────────────────────────────────────────────
def limpiar():
    entrada_puntos.delete("1.0", tk.END)
    entrada_valor.delete(0, tk.END)
    salida.config(state="normal")
    salida.delete("1.0", tk.END)
    salida.config(state="disabled")
    fig.clf()
    canvas.draw()


# ─────────────────────────────────────────────
#  INTERFAZ
# ─────────────────────────────────────────────
ventana = tk.Tk()
ventana.title("Spline Cúbico Natural — con Gráficas")
ventana.state("zoomed")

# Panel izquierdo (controles)
frame_izq = tk.Frame(ventana, width=320)
frame_izq.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=8)
frame_izq.pack_propagate(False)

tk.Label(frame_izq, text="Spline Cúbico Natural",
         font=("Arial", 14, "bold")).pack(pady=(8, 2))

tk.Label(frame_izq, text="Puntos (x,y — uno por línea):",
         font=("Arial", 10)).pack(anchor="w")

entrada_puntos = tk.Text(frame_izq, width=30, height=7, font=("Courier", 10))
entrada_puntos.pack(pady=3)
entrada_puntos.insert(tk.END, "1,2\n2,3\n3,5\n4,4\n5,6")

tk.Label(frame_izq, text="Valor a interpolar (x):",
         font=("Arial", 10)).pack(anchor="w", pady=(6, 0))

entrada_valor = tk.Entry(frame_izq, width=15, font=("Arial", 11))
entrada_valor.pack(anchor="w")
entrada_valor.insert(0, "2.5")

frame_botones = tk.Frame(frame_izq)
frame_botones.pack(pady=8)

tk.Button(frame_botones, text="Calcular", font=("Arial", 10, "bold"),
          width=12, bg="#1976D2", fg="white",
          command=calcular).grid(row=0, column=0, padx=4)

tk.Button(frame_botones, text="Limpiar", font=("Arial", 10),
          width=12, command=limpiar).grid(row=0, column=1, padx=4)

tk.Button(frame_botones, text="Salir", font=("Arial", 10),
          width=12, command=ventana.destroy).grid(row=1, column=0, columnspan=2, pady=4)

tk.Label(frame_izq, text="Procedimiento:",
         font=("Arial", 10, "bold")).pack(anchor="w", pady=(4, 0))

salida = tk.Text(frame_izq, width=36, height=22, font=("Courier", 8),
                 wrap="word", state="disabled", bg="#f5f5f5")
salida.pack(fill=tk.BOTH, expand=True, pady=2)

# Panel derecho (gráficas)
frame_der = tk.Frame(ventana)
frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=8)

fig = plt.Figure(figsize=(10, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_der)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

ventana.mainloop()