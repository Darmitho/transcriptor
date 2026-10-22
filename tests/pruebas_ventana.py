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


def letra_y_colores():
    """A−/A+ cambian la letra con límites y se recuerda; cada hablante lleva su color;
    un color inválido guardado vuelve al inicial; Ajustes no se abre dos veces."""
    fallos = []
    with app_aislada():
        with open(T.CONFIG_ARCHIVO, "w", encoding="utf-8") as f:
            json.dump({"color_yo": "verdoso", "color_otros": "#ffcc00"}, f)
        root = tk.Tk()
        app = T.App(root)
        app.cola_ui.put(("final", ("Yo", "Hola.", time.time())))
        app.cola_ui.put(("final", ("Otros", "Buenas.", time.time() + 1)))
        app._procesar_cola()
        color = lambda fuente: app.texto.tag_cget(f"hablante_{fuente}", "foreground")
        etiquetas = [app.texto.tag_names(f"{n}.12") for n in (1, 2)]
        if color("Yo") != T.Configuracion.INICIAL["color_yo"] or color("Otros") != "#ffcc00":
            fallos.append(("colores", color("Yo"), color("Otros")))
        if "hablante_Yo" not in etiquetas[0] or "hablante_Otros" not in etiquetas[1]:
            fallos.append(("etiquetas de las líneas", etiquetas))
        app.cambiar_letra(+1)
        if (app.letra.cget("size"), app.letra_hora.cget("size")) != (15, 11):
            fallos.append(("A+", app.letra.cget("size"), app.letra_hora.cget("size")))
        app.cambiar_letra(tamano=100)
        tope = app.letra.cget("size")
        app.cambiar_letra(tamano=1)
        piso = app.letra.cget("size")
        if (tope, piso) != (T.Configuracion.LETRA_MAX, T.Configuracion.LETRA_MIN):
            fallos.append(("límites", tope, piso))
        app.cambiar_letra(tamano=20)
        app.abrir_ajustes()
        primera = app.ajustes
        app.abrir_ajustes()
        if app.ajustes is not primera:
            fallos.append(("Ajustes abierto dos veces",))
        app.config["color_otros"] = "#00aaff"   # Lo que deja elegir_color tras el selector
        app._aplicar_colores()
        app.config.guardar()
        root.destroy()

        root = tk.Tk()
        app = T.App(root)
        recuperado = (app.letra.cget("size"), app.texto.tag_cget("hablante_Otros", "foreground"))
        root.destroy()
        if recuperado != (20, "#00aaff"):
            fallos.append(("recuperado", recuperado))
    return fallos


def medidores_y_estado():
    """Barras según el nivel de cada fuente (vacías si no hay audio reciente o si la fuente
    no existe) y estado: Detenido, Escuchando, Transcribiendo, Atrasado, Error."""
    from types import SimpleNamespace
    fallos = []
    with app_aislada():
        root = tk.Tk()
        app = T.App(root)
        ancho = lambda n: (lambda c: c[2] - c[0])(app.medidores.coords(app.barras[n][0]))
        estado = lambda: (app.texto_estado.cget("text").split(" ·")[0], app.punto_estado.cget("fg"))

        app._actualizar_indicadores()
        if estado() != ("Detenido", app.GRIS):
            fallos.append(("detenido", estado()))

        ahora = time.time()
        otros = SimpleNamespace(nombre="Otros", retirar=False, nivel=0.1, t_nivel=ahora)    # -20 dBFS
        app.motor.fuentes = [otros]          # Sin micrófono: la barra de "Yo" queda vacía
        app.activo = True
        app.motor.atraso = lambda: 0.2
        app._actualizar_indicadores()
        esperado = (60 - 20) / 60 * (app.ANCHO_MEDIDOR - 2)
        if abs(ancho("Otros") - esperado) > 1 or ancho("Yo") != 0:
            fallos.append(("barras", ancho("Otros"), esperado, ancho("Yo")))
        if estado() != ("Escuchando", app.VERDE) or "atraso 0.2 s" not in app.texto_estado.cget("text"):
            fallos.append(("escuchando", app.texto_estado.cget("text")))

        otros.t_nivel = ahora - 5            # Sin audio reciente (WASAPI calla en silencio): cae
        antes = ancho("Otros")
        app._actualizar_indicadores()
        if not (0 < ancho("Otros") < antes):
            fallos.append(("caída suave", antes, ancho("Otros")))
        for _ in range(15):
            app._actualizar_indicadores()
        if ancho("Otros") != 0:
            fallos.append(("cae a cero", ancho("Otros")))

        app.motor.transcribiendo = True
        app._actualizar_indicadores()
        t = estado()
        app.motor.atraso = lambda: 2.5
        app._actualizar_indicadores()
        a = estado()
        app.cola_ui.put(("estado", "Error: algo falló"))
        app._procesar_cola()
        app._actualizar_indicadores()
        e = estado()
        if (t, a, e) != (("Transcribiendo", app.ACENTO), ("Atrasado", app.AMARILLO), ("Error", app.ROJO)):
            fallos.append(("estados", t, a, e))
        app.activo = False
        root.destroy()
    return fallos


def atajos():
    """Pulsaciones reales sobre la ventana: Ctrl+Espacio inicia y detiene (no mientras el
    botón está deshabilitado), Ctrl+M alterna el micrófono, Ctrl +/= y − cambian la letra."""
    fallos = []
    with app_aislada():
        root = tk.Tk()
        app = T.App(root)
        llamadas = []
        app.motor.iniciar = lambda con_mic=False: llamadas.append(("iniciar", con_mic))
        app.motor.detener = lambda: llamadas.append(("detener",))
        app.motor.activar_mic = lambda activo: llamadas.append(("mic", activo))
        root.deiconify()
        root.focus_force()
        _esperar(root, 0.3)

        def pulsar(secuencia):
            app.texto.focus_force()
            app.texto.event_generate(secuencia)
            _esperar(root, 0.05)

        pulsar("<Control-space>")           # Botón deshabilitado: el modelo no cargó todavía
        if llamadas:
            fallos.append(("inició sin modelo", list(llamadas)))
        app.cola_ui.put(("modelo_listo", None))
        app._procesar_cola()
        pulsar("<Control-space>")
        pulsar("<Control-m>")
        pulsar("<Control-space>")
        if llamadas != [("iniciar", False), ("mic", True), ("detener",)] or not app.mic.get():
            fallos.append(("iniciar / micrófono / detener", list(llamadas), app.mic.get()))
        tamano = app.letra.cget("size")
        pulsar("<Control-equal>")
        pulsar("<Control-plus>")
        pulsar("<Control-minus>")
        if app.letra.cget("size") != tamano + 1:
            fallos.append(("letra", tamano, app.letra.cget("size")))
        app._cancelar_timer_vram()
        root.destroy()
    return fallos


def subtitulos():
    """La franja: no captura clics, queda visible con la ventana minimizada, muestra las últimas
    líneas con sus colores, se ubica abajo o arriba dentro del área útil y vuelve a la vista normal."""
    fallos = []
    with app_aislada():
        root = tk.Tk()
        app = T.App(root)
        base = time.time()
        for i, (fuente, texto) in enumerate([("Otros", "Uno."), ("Yo", "Dos."), ("Otros", "Tres.")]):
            app.cola_ui.put(("final", (fuente, texto, base + i)))
        app.cola_ui.put(("parcial", ("Yo", "cuatro provisional")))
        app._procesar_cola()

        app.alternar_subtitulos()
        _esperar(root, 0.4)
        f = app.franja
        if f is None or not app.franja_sin_clics or not f.winfo_viewable() or root.state() != "iconic":
            fallos.append(("entrar", f is not None, getattr(app, "franja_sin_clics", None),
                           f is not None and f.winfo_viewable(), root.state()))
        lineas = app.texto_franja.get("1.0", "end-1c").splitlines()
        if lineas != ["Otros: Tres.", "Yo: cuatro provisional"]:
            fallos.append(("líneas", lineas))
        if "hablante_Otros" not in app.texto_franja.tag_names("1.2") or "parcial" not in app.texto_franja.tag_names("2.2"):
            fallos.append(("colores", app.texto_franja.tag_names("1.2"), app.texto_franja.tag_names("2.2")))

        x0, y0, x1, y1 = T._area_trabajo()
        abajo = (f.winfo_y(), f.winfo_height())
        app._cambiar_franja("franja_posicion", "arriba")
        app._cambiar_franja("franja_lineas", 3)
        app._cambiar_franja("franja_opacidad", 0.6)
        _esperar(root, 0.2)
        arriba = (f.winfo_y(), f.winfo_height())
        lineas3 = app.texto_franja.get("1.0", "end-1c").splitlines()
        if not (abajo[0] + abajo[1] <= y1 and abajo[0] > (y0 + y1) / 2 and y0 <= arriba[0] < (y0 + y1) / 2
                and arriba[1] > abajo[1] and len(lineas3) == 3 and abs(float(f.attributes("-alpha")) - 0.6) < 0.02):
            fallos.append(("ubicación / líneas / opacidad", abajo, arriba, (y0, y1), lineas3, f.attributes("-alpha")))

        app.alternar_subtitulos()
        _esperar(root, 0.3)
        if app.franja is not None or root.state() != "normal" or app.subtitulos.get():
            fallos.append(("salir", app.franja, root.state(), app.subtitulos.get()))
        root.destroy()
    return fallos


def atajo_global():
    """De punta a punta: una pulsación real (simulada con la API de Windows) del atajo global
    alterna la franja aunque el foco esté en otra ventana. Usa una combinación poco común."""
    import ctypes
    fallos = []
    casos = {"Ctrl+Shift+S": (0x6, ord("S")), "ctrl + alt + f11": (0x3, 0x7A), "Win+F24": (0x8, 0x87),
             "S": None, "Ctrl+Ñ": None, "Hiper+S": None, "Ctrl+F25": None}
    for texto, esperado in casos.items():
        if T._parsear_atajo(texto) != esperado:
            fallos.append(("parsear", texto, T._parsear_atajo(texto), esperado))

    with app_aislada(atajo_global="Ctrl+Alt+Shift+F11"):
        root = tk.Tk()
        app = T.App(root)
        app.atajo_global.registrado.wait(2)
        if not app.atajo_global.ok:
            root.destroy()
            return fallos + [("no se pudo registrar Ctrl+Alt+Shift+F11",)]
        u = ctypes.windll.user32

        def pulsar():
            teclas = (0x11, 0x12, 0x10, 0x7A)      # Ctrl, Alt, Shift, F11
            for vk in teclas:
                u.keybd_event(vk, 0, 0, 0)
            for vk in reversed(teclas):
                u.keybd_event(vk, 0, 2, 0)         # KEYEVENTF_KEYUP
            _esperar(root, 0.5)
            app._procesar_cola()

        pulsar()
        entro = app.franja is not None
        pulsar()
        salio = app.franja is None
        hilo = app.atajo_global
        root.destroy()
        hilo.join(2)
        if not (entro and salio):
            fallos.append(("atajo global", entro, salio))
        if hilo.is_alive():
            fallos.append(("el atajo no se soltó al cerrar",))
    return fallos


PRUEBAS = [orden_cronologico, timer_vram, configuracion, letra_y_colores, medidores_y_estado, atajos,
           subtitulos, atajo_global]


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
