"""Costo en RAM y CPU de la misma maqueta de interfaz en Tkinter, CustomTkinter y PySide6.

Uso:  python tools\\bench_gui.py            (con un Python que tenga customtkinter, PySide6 y psutil;
                                            no hace falta el modelo ni la GPU)
Cada framework corre en un proceso aparte con la misma maqueta: ventana principal con barra
de controles, ~200 líneas de texto, dos medidores de nivel animados a 20 Hz e indicador de
estado, más la franja de subtítulos (sin bordes, semitransparente, siempre encima y que no
captura clics). Se mide la RAM tras 5 s y la CPU durante 10 s.
"""

import json
import subprocess
import sys
import time

LINEAS = 200
HZ_MEDIDORES = 20
TEXTO = "Hoy vamos a revisar el presupuesto del proyecto y las fechas de entrega con el cliente."


def click_through_windows(hwnd):
    """Estilo extendido de Windows para que la franja no capture clics. Devuelve si quedó aplicado."""
    import ctypes
    GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
    u = ctypes.windll.user32
    u.SetWindowLongW(hwnd, GWL_EXSTYLE, u.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    return bool(u.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TRANSPARENT)


def maqueta_tk(ctk=False):
    import random
    import tkinter as tk
    if ctk:
        import customtkinter as c
        c.set_appearance_mode("dark")
        root = c.CTk()
        Boton = lambda p, t: c.CTkButton(p, text=t, width=90)
        Interruptor = lambda p, t: c.CTkSwitch(p, text=t)
        texto = c.CTkTextbox(root, font=("Segoe UI", 14))
        estado = c.CTkLabel(root, text="Escuchando · atraso 0.0 s")
        Toplevel = c.CTkToplevel
    else:
        root = tk.Tk()
        root.configure(bg="#1e1f22")
        Boton = lambda p, t: tk.Button(p, text=t, bg="#2b2d31", fg="#e8e8e8", relief="flat", padx=10, pady=4)
        Interruptor = lambda p, t: tk.Checkbutton(p, text=t, bg="#1e1f22", fg="#e8e8e8", selectcolor="#2b2d31")
        texto = tk.Text(root, font=("Segoe UI", 14), bg="#1e1f22", fg="#e8e8e8", relief="flat")
        estado = tk.Label(root, text="Escuchando · atraso 0.0 s", bg="#1e1f22", fg="#8a8d93")
        Toplevel = tk.Toplevel
    root.geometry("760x480")
    barra = tk.Frame(root, bg="#1e1f22")
    barra.pack(fill="x", padx=8, pady=6)
    for t in ("Iniciar", "Limpiar", "Guardar", "Reconectar", "Subtítulos"):
        Boton(barra, t).pack(side="left", padx=2)
    Interruptor(barra, "Micrófono").pack(side="left", padx=6)
    medidores = tk.Canvas(root, height=36, bg="#1e1f22", highlightthickness=0)
    medidores.pack(fill="x", padx=8)
    barras = [medidores.create_rectangle(70, y, 70, y + 10, fill=c_, width=0)
              for y, c_ in ((6, "#6ea8fe"), (22, "#5cc98a"))]
    medidores.create_text(8, 11, text="Otros", anchor="w", fill="#8a8d93")
    medidores.create_text(8, 27, text="Yo", anchor="w", fill="#8a8d93")
    texto.pack(fill="both", expand=True, padx=8)
    for i in range(LINEAS):
        texto.insert("end", f"12:{i // 60:02d}:{i % 60:02d}  {'Yo' if i % 3 else 'Otros'}: {TEXTO}\n")
    estado.pack(fill="x", padx=8, pady=4)

    franja = Toplevel(root)
    franja.overrideredirect(True)
    franja.attributes("-topmost", True)
    franja.attributes("-alpha", 0.85)
    franja.geometry("900x110+200+700")
    sub = tk.Label(franja, text=TEXTO, font=("Segoe UI", 22), bg="#000000", fg="#ffffff", wraplength=880)
    sub.pack(fill="both", expand=True)
    root.update()
    clics_ok = click_through_windows(int(franja.wm_frame(), 16))

    def animar():
        for b in barras:
            medidores.coords(b, 70, medidores.coords(b)[1], 70 + random.random() * 600, medidores.coords(b)[3])
        root.after(1000 // HZ_MEDIDORES, animar)

    def cambiar_sub(n=[0]):
        n[0] += 1
        sub.configure(text=f"{TEXTO} ({n[0]})")
        root.after(1000, cambiar_sub)

    animar()
    cambiar_sub()
    return root.mainloop, clics_ok


def maqueta_qt():
    import random
    from PySide6 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    win.resize(760, 480)
    win.setStyleSheet("background:#1e1f22; color:#e8e8e8;")
    lay = QtWidgets.QVBoxLayout(win)
    barra = QtWidgets.QHBoxLayout()
    for t in ("Iniciar", "Limpiar", "Guardar", "Reconectar", "Subtítulos"):
        barra.addWidget(QtWidgets.QPushButton(t))
    barra.addWidget(QtWidgets.QCheckBox("Micrófono"))
    lay.addLayout(barra)
    medidores = [QtWidgets.QProgressBar(maximum=100, textVisible=False) for _ in range(2)]
    for m in medidores:
        lay.addWidget(m)
    texto = QtWidgets.QPlainTextEdit()
    texto.setFont(QtGui.QFont("Segoe UI", 14))
    texto.setPlainText("\n".join(f"12:{i // 60:02d}:{i % 60:02d}  {'Yo' if i % 3 else 'Otros'}: {TEXTO}"
                                 for i in range(LINEAS)))
    lay.addWidget(texto)
    lay.addWidget(QtWidgets.QLabel("Escuchando · atraso 0.0 s"))
    win.show()

    franja = QtWidgets.QLabel(TEXTO)
    franja.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
                          | QtCore.Qt.WindowTransparentForInput)
    franja.setWindowOpacity(0.85)
    franja.setStyleSheet("background:#000; color:#fff; font: 22pt 'Segoe UI';")
    franja.setWordWrap(True)
    franja.setGeometry(200, 700, 900, 110)
    franja.show()
    clics_ok = bool(franja.windowFlags() & QtCore.Qt.WindowTransparentForInput)

    t1 = QtCore.QTimer()
    t1.timeout.connect(lambda: [m.setValue(random.randint(0, 100)) for m in medidores])
    t1.start(1000 // HZ_MEDIDORES)
    n = [0]
    t2 = QtCore.QTimer()

    def cambiar_sub():
        n[0] += 1
        franja.setText(f"{TEXTO} ({n[0]})")

    t2.timeout.connect(cambiar_sub)
    t2.start(1000)
    app._timers = (t1, t2)
    return app.exec, clics_ok


def hijo(framework):
    """Proceso medido: arma la maqueta y reporta su propia RAM y CPU."""
    import threading
    import psutil
    p = psutil.Process()
    ram_inicio = p.memory_info().rss / 2**20
    bucle, clics_ok = {"tk": maqueta_tk, "ctk": lambda: maqueta_tk(ctk=True), "qt": maqueta_qt}[framework]()

    def medir():
        time.sleep(5)
        ram = p.memory_info().rss / 2**20
        p.cpu_percent(None)
        time.sleep(10)
        cpu = p.cpu_percent(None)
        print(json.dumps({"ram_vacio": ram_inicio, "ram": ram, "cpu": cpu, "clics_ok": clics_ok}), flush=True)
        import os
        os._exit(0)

    threading.Thread(target=medir, daemon=True).start()
    bucle()


def main():
    nombres = {"tk": "Tkinter", "ctk": "CustomTkinter", "qt": "PySide6 (Qt)"}
    print(f"{'framework':16s} {'RAM total':>10s} {'RAM de la UI':>13s} {'CPU % 1 núcleo':>15s}  franja sin clics")
    for fw, nombre in nombres.items():
        r = []
        for _ in range(3):              # 3 corridas: se informa la mediana
            salida = subprocess.run([sys.executable, __file__, "--hijo", fw], capture_output=True, text=True,
                                    timeout=120).stdout.strip().splitlines()
            r.append(json.loads(salida[-1]))
        med = lambda k: sorted(x[k] for x in r)[1]
        print(f"{nombre:16s} {med('ram'):8.0f} MB {med('ram') - med('ram_vacio'):10.0f} MB {med('cpu'):14.1f}   "
              f"{'sí' if all(x['clics_ok'] for x in r) else 'NO'}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--hijo":
        hijo(sys.argv[2])
    else:
        main()
