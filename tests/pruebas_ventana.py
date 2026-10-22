"""Pruebas de la ventana (App) que no necesitan el modelo ni audio: corren en segundos.

Uso:  venv\\Scripts\\python.exe tests\\pruebas_ventana.py
La batería completa (regresion.py) también las corre. Cada prueba devuelve una lista
de fallos (vacía si todo está bien) y usa una carpeta temporal: nunca toca las
transcripciones ni la configuración reales.
"""

import json
import os
import sys
import time
import tkinter as tk

from comun import T, app_aislada


def _esperar(root, segundos):
    fin = time.time() + segundos
    while time.time() < fin:
        root.update()
        time.sleep(0.02)


def orden_cronologico():
    """Frases que llegan fuera de orden: la ventana, Guardar y el .txt quedan cronológicos,
    y Limpiar no borra nada del .txt."""
    with app_aislada():
        root = tk.Tk()
        app = T.App(root)
        base = time.time()
        for fuente, texto, seg in [("Otros", "Segunda.", 10), ("Yo", "Primera.", 5),
                                   ("Otros", "Cuarta.", 30), ("Yo", "Tercera.", 20)]:
            app.cola_ui.put(("final", (fuente, texto, base + seg)))
        app.cola_ui.put(("parcial", ("Otros", "provisional")))
        app._procesar_cola()
        ventana = app.texto.get("1.0", "end-1c").splitlines()
        with open(app.archivo_auto, encoding="utf-8") as f:
            archivo = f.read().splitlines()
        app.limpiar()
        app.cola_ui.put(("final", ("Yo", "Quinta.", base + 40)))
        app._procesar_cola()
        with open(app.archivo_auto, encoding="utf-8") as f:
            tras_limpiar = f.read().splitlines()
        root.destroy()
    orden = ["Primera.", "Segunda.", "Tercera.", "Cuarta."]
    hora = time.strftime("%H:%M:%S", time.localtime(base + 5))
    fallos = []
    if [l.split(": ", 1)[-1] for l in ventana[:4]] != orden or ventana[4] != "Otros: provisional":
        fallos.append(("ventana", ventana))
    if [l.split(": ", 1)[-1] for l in archivo] != orden or not archivo[0].startswith(f"[{hora}] Yo:"):
        fallos.append(("archivo", archivo))
    if len(tras_limpiar) != 5:
        fallos.append(("Limpiar borró líneas del .txt", tras_limpiar))
    return fallos


def timer_vram():
    """La ventana pide liberar la VRAM tras LIBERAR_VRAM_TRAS detenido, y no si se inicia antes."""
    espera = T.LIBERAR_VRAM_TRAS
    T.LIBERAR_VRAM_TRAS = 0.2
    try:
        with app_aislada():
            root = tk.Tk()
            app = T.App(root)
            llamadas = []
            app.motor.liberar_vram = lambda: llamadas.append(time.time()) or True
            app.cola_ui.put(("modelo_listo", None))
            app._procesar_cola()
            _esperar(root, 0.5)
            tras_espera = len(llamadas)
            app._programar_liberar_vram()   # Vuelve a quedar detenido...
            app._cancelar_timer_vram()      # ...pero se inicia antes de que venza
            _esperar(root, 0.5)
            tras_cancelar = len(llamadas)
            root.destroy()
    finally:
        T.LIBERAR_VRAM_TRAS = espera
    return [] if (tras_espera, tras_cancelar) == (1, 1) else [("timer VRAM", tras_espera, tras_cancelar)]


def configuracion():
    """Se guarda y se recupera entre sesiones; un archivo dañado o con tipos raros no rompe nada."""
    fallos = []
    with app_aislada() as carpeta:
        ruta = T.CONFIG_ARCHIVO
        if dict(T.Configuracion()) != T.Configuracion.INICIAL:
            fallos.append(("sin archivo: valores iniciales", dict(T.Configuracion())))

        # Sesión 1: el usuario cambia idioma, "siempre encima" y la ventana; cierra
        root = tk.Tk()
        app = T.App(root)
        app.idioma.set("English")
        app._cambiar_idioma("English")
        app.encima.set(False)
        app._encima()
        root.geometry("820x500+60+70")
        root.update()
        app.cerrar()
        with open(ruta, encoding="utf-8") as f:
            guardada = json.load(f)
        if (guardada.get("idioma"), guardada.get("siempre_encima"), guardada.get("geometria")) != \
                ("en", False, "820x500+60+70"):
            fallos.append(("guardada", guardada))

        # Sesión 2: arranca como quedó, y el micrófono sigue apagado
        root = tk.Tk()
        app = T.App(root)
        root.update()
        recuperado = (app.idioma.get(), app.motor.contexto[0], app.encima.get(),
                      bool(root.attributes("-topmost")), root.geometry(), app.mic.get())
        root.destroy()
        if recuperado != ("English", "en", False, False, "820x500+60+70", False):
            fallos.append(("recuperada", recuperado))

        # Archivo dañado, claves desconocidas y tipos equivocados: se ignoran sin error
        for contenido in ("{no es json", json.dumps({"idioma": 3, "siempre_encima": "sí", "otra": 1}),
                          json.dumps(["lista"])):
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(contenido)
            if dict(T.Configuracion()) != T.Configuracion.INICIAL:
                fallos.append(("contenido inválido aceptado", contenido, dict(T.Configuracion())))
        if not os.path.isdir(carpeta):
            fallos.append(("carpeta temporal", carpeta))

    # Ventana guardada en un monitor que ya no está: se conserva el tamaño, no la posición
    if T._geometria_visible("800x500+99999+99999") != "800x500":
        fallos.append(("fuera de pantalla", T._geometria_visible("800x500+99999+99999")))
    if T._geometria_visible("800x500+40+40") != "800x500+40+40" or T._geometria_visible("basura") != "760x440":
        fallos.append(("geometría", T._geometria_visible("800x500+40+40"), T._geometria_visible("basura")))
    return fallos


PRUEBAS = [orden_cronologico, timer_vram, configuracion]


def todas():
    """Corre todas las pruebas de la ventana: (cantidad, lista de fallos)."""
    fallos = []
    for prueba in PRUEBAS:
        fallos += [(prueba.__name__,) + tuple(f) for f in prueba()]
    return len(PRUEBAS), fallos


if __name__ == "__main__":
    n, fallos = todas()
    print(f"ventana: {n - len({f[0] for f in fallos})}/{n} OK")
    for f in fallos:
        print("  FALLO:", f)
    sys.exit(1 if fallos else 0)
