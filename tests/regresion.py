"""Batería de regresión con el modelo real y audio sintético alimentado a ritmo real.

Uso:  venv\\Scripts\\python.exe tests\\regresion.py [--velocidad 1.0]
Tarda unos 4 minutos. Sale con código 1 si algún escenario falla.
Correrla antes de cada commit: ningún cambio entra si empeora el total de errores.
"""

import argparse
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from comun import (T, DISP, PARTES, REF_FRASE, wav, silencio, texto_continuo,
                   frases_con_pausas, errores_palabras, alimentar)

UMBRAL_ACIERTO = 0.95                   # Mínimo de palabras correctas por fuente y escenario


def escenarios():
    rng = np.random.default_rng(1)
    g, ref_g = frases_con_pausas()
    h = np.concatenate([silencio(1), wav("continuo"), silencio(2)])
    ref_h = texto_continuo()
    yo = np.concatenate([silencio(3), g])
    largo = max(len(h), len(yo))
    return {
        "frases con respiraciones": {"Otros": (g, ref_g)},
        "habla continua 25 s": {"Otros": (h, ref_h)},
        "frase + 30 s de silencio": {"Otros": (np.concatenate([silencio(1), wav("frase"), silencio(30)]), REF_FRASE)},
        "15 s de ruido": {"Otros": ((rng.standard_normal(48000 * 15) * 0.02).astype(np.float32), "")},
        "'Gracias.' aislado": {"Otros": (np.concatenate([silencio(2), wav("gracias"), silencio(3)]), "Gracias.")},
        "dos fuentes a la vez": {
            "Otros": (np.concatenate([h, silencio((largo - len(h)) / 48000)]), ref_h),
            "Yo": (np.concatenate([yo, silencio((largo - len(yo)) / 48000)]), ref_g),
        },
    }


def correr_escenario(motor, pistas, velocidad):
    cola_ui = motor.cola_ui
    while not cola_ui.empty():
        cola_ui.get()
    motor.stop = threading.Event()
    motor.cola_audio = queue.Queue()
    fuentes = {n: T.Fuente(n, DISP, motor.cola_audio) for n in pistas}
    motor.fuentes = list(fuentes.values())
    motor.hilo = threading.Thread(target=motor._bucle, args=(motor.stop, motor.cola_audio, motor.fuentes),
                                  daemon=True)
    motor.hilo.start()
    t0 = time.time()
    alimentar({n: a for n, (a, _) in pistas.items()}, fuentes, motor.cola_audio, velocidad)
    time.sleep(1.0)
    motor.stop.set()
    motor.hilo.join(120)
    msgs = list(cola_ui.queue)
    assert msgs and msgs[-1][0] == "detenido", "el hilo no avisó 'detenido'"
    finales = [v for k, v in msgs if k == "final"]
    textos = {n: " ".join(f[1] for f in finales if f[0] == n) for n in pistas}
    # Segundos de audio (desde el inicio de la pista) en que empezó cada frase, por fuente
    inicios = {n: [(f[2] - t0) * velocidad for f in finales if f[0] == n] for n in pistas}
    return textos, inicios


# Dónde empieza la voz en cada pista (segundos de audio), según cómo se arman en escenarios():
# 1 s de silencio + 0.1 s de silencio propio del clip; "Yo" va 3 s detrás de "Otros"
HORAS_ESPERADAS = {
    ("frase + 30 s de silencio", "Otros"): 1.1,
    ("habla continua 25 s", "Otros"): 1.0,
    ("dos fuentes a la vez", "Yo"): 4.0,
}
TOLERANCIA_HORA = 0.4                   # Trozos de 100 ms + reparto de la cola


def pruebas_unitarias(motor):
    """Reglas de descarte, solapes y prompt: no necesitan audio."""
    seg = lambda t, lp=-0.2, ns=0.05, cr=1.2: SimpleNamespace(text=t, avg_logprob=lp, no_speech_prob=ns,
                                                             compression_ratio=cr)
    prompt = T._prompt_vocabulario(T.VOCABULARIO, "es")
    casos = [
        (seg(" Gracias."), 1, 0.6, None, None),
        (seg(" ¡Gracias!", lp=-0.7), 1, 0.6, None, "sospechosa"),
        (seg(" ¡Gracias!", ns=0.5), 1, 0.6, None, "sospechosa"),
        (seg(" Gracias."), 1, 0.26, None, "sospechosa"),
        (seg(" Gracias, Juan, lo reviso mañana.", lp=-0.7), 1, 0.3, None, None),
        (seg(" Thank you."), 1, 0.6, None, None),
        (seg(" algo", lp=-1.2), 1, 2.0, None, "logprob"),
        (seg(" hola", ns=0.9), 1, 2.0, None, "no_voz"),
        (seg(" sí sí sí sí", cr=3.0), 1, 2.0, None, "compresion"),
        (seg(" " + prompt), 1, 3.0, prompt, "eco_prompt"),
        (seg(" el order block y RSI"), 1, 2.0, prompt, None),
    ]
    fallos = [(s.text, esperado, r) for s, n, v, p, esperado in casos
              if (r := motor._motivo_descarte(s, n, v, p)) != esperado]
    otros = [
        (T._quitar_solape("sin afectar a los", "los usuarios que ya"), "usuarios que ya"),
        (T._quitar_solape("hola mundo", "otra cosa"), "otra cosa"),
        (T._prompt_vocabulario(("A", "B", "C"), "es"), "Hoy hablamos de A, B y C."),
        (T._prompt_vocabulario(("A", "B"), "en"), "Today we are talking about A and B."),
        (T._prompt_vocabulario((), "es"), None),
    ]
    fallos += [(obtenido, esperado, "") for obtenido, esperado in otros if obtenido != esperado]
    return len(casos) + len(otros), fallos


def prueba_ventana():
    """Frases que llegan fuera de orden: la ventana, Guardar y el .txt quedan cronológicos,
    y Limpiar no borra nada del .txt. No carga el modelo."""
    import tempfile
    import tkinter as tk
    cargar, carpeta = T.Motor.cargar_modelo, T.CARPETA_SALIDA
    T.Motor.cargar_modelo, T.CARPETA_SALIDA = (lambda self: None), tempfile.mkdtemp(prefix="regresion_")
    try:
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
    finally:
        T.Motor.cargar_modelo, T.CARPETA_SALIDA = cargar, carpeta
    orden = ["Primera.", "Segunda.", "Tercera.", "Cuarta."]
    hora = lambda s: time.strftime("%H:%M:%S", time.localtime(base + s))
    fallos = []
    if [l.split(": ", 1)[-1] for l in ventana[:4]] != orden or ventana[4] != "Otros: provisional":
        fallos.append(("ventana", ventana))
    if [l.split(": ", 1)[-1] for l in archivo] != orden or not archivo[0].startswith(f"[{hora(5)}] Yo:"):
        fallos.append(("archivo", archivo))
    if len(tras_limpiar) != 5:
        fallos.append(("Limpiar borró líneas del .txt", tras_limpiar))
    return fallos


def prueba_timer_vram():
    """La ventana pide liberar la VRAM tras LIBERAR_VRAM_TRAS detenido, y no si se inicia antes."""
    import tempfile
    import tkinter as tk
    cargar, carpeta, espera = T.Motor.cargar_modelo, T.CARPETA_SALIDA, T.LIBERAR_VRAM_TRAS
    T.Motor.cargar_modelo, T.CARPETA_SALIDA = (lambda self: None), tempfile.mkdtemp(prefix="regresion_")
    T.LIBERAR_VRAM_TRAS = 0.2
    try:
        root = tk.Tk()
        app = T.App(root)
        llamadas = []
        app.motor.liberar_vram = lambda: llamadas.append(time.time()) or True
        app.cola_ui.put(("modelo_listo", None))
        app._procesar_cola()

        def esperar(segundos):
            fin = time.time() + segundos
            while time.time() < fin:
                root.update()
                time.sleep(0.02)

        esperar(0.5)
        tras_espera = len(llamadas)
        app._programar_liberar_vram()   # Vuelve a quedar detenido...
        app._cancelar_timer_vram()      # ...pero se inicia antes de que venza
        esperar(0.5)
        tras_cancelar = len(llamadas)
        root.destroy()
    finally:
        T.Motor.cargar_modelo, T.CARPETA_SALIDA, T.LIBERAR_VRAM_TRAS = cargar, carpeta, espera
    return [] if (tras_espera, tras_cancelar) == (1, 1) else [("timer VRAM", tras_espera, tras_cancelar)]


def prueba_recarga(motor, velocidad):
    """Liberar la VRAM baja el uso de la GPU, y al volver a escuchar el modelo se recarga
    sin perder la frase que empezó durante la recarga."""
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    usada = lambda: pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20
    antes = usada()
    if not motor.liberar_vram():
        return [("liberar_vram no liberó", motor.vram_liberada)], 0.0
    liberada = antes - usada()
    pistas = {"Otros": (np.concatenate([silencio(0.3), wav("frase"), silencio(2)]), REF_FRASE)}
    textos, _ = correr_escenario(motor, pistas, velocidad)
    err, _ = errores_palabras(REF_FRASE, textos["Otros"])
    fallos = []
    if liberada < 500:
        fallos.append(("VRAM liberada (MB)", round(liberada)))
    if err or motor.vram_liberada:
        fallos.append(("tras recargar", textos["Otros"], motor.vram_liberada))
    return fallos, liberada


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocidad", type=float, default=1.0, help="1.0 = tiempo real (lo representativo)")
    args = ap.parse_args()

    fallos_unit = prueba_ventana() + prueba_timer_vram()
    motor = T.Motor(queue.Queue())
    motor.cargar_modelo()
    n_unit, f_unit = pruebas_unitarias(motor)
    fallos_unit += f_unit
    n_unit += 2                         # + la ventana y el temporizador de la VRAM
    print(f"\nunitarias: {n_unit - len(fallos_unit)}/{n_unit} OK")
    for f in fallos_unit:
        print("  FALLO:", f)

    total_err = total_pal = fallos = 0
    print(f"\n{'escenario':28s} {'fuente':6s} {'errores':>8s} {'palabras':>9s} {'inicio 1ª frase':>16s}  resultado")
    for nombre, pistas in escenarios().items():
        textos, inicios = correr_escenario(motor, pistas, args.velocidad)
        for fuente, (_, ref) in pistas.items():
            err, pal = errores_palabras(ref, textos[fuente])
            ok = (1 - err / pal >= UMBRAL_ACIERTO) if pal else not textos[fuente]
            esperado = HORAS_ESPERADAS.get((nombre, fuente))
            hora = f"{inicios[fuente][0]:.2f} s" if inicios[fuente] else "-"
            if esperado is not None:
                hora_ok = bool(inicios[fuente]) and abs(inicios[fuente][0] - esperado) <= TOLERANCIA_HORA
                hora += f" (esp. {esperado})"
                ok &= hora_ok
            # Tras un corte en palabra, la frase siguiente empieza donde quedó el corte (~19-20 s)
            if nombre == "habla continua 25 s" and len(inicios[fuente]) > 1:
                ok &= 18.0 <= inicios[fuente][1] <= 21.5
                hora += f", 2ª {inicios[fuente][1]:.1f} s"
            total_err += err
            total_pal += pal
            fallos += not ok
            print(f"{nombre:28s} {fuente:6s} {err:8d} {pal:9d} {hora:>16s}  {'OK' if ok else 'FALLO'}", flush=True)

    f_recarga, liberada = prueba_recarga(motor, args.velocidad)
    fallos += len(f_recarga)
    print(f"{'liberar VRAM y recargar':28s} {'Otros':6s} {'':>8s} {'':>9s} {f'-{liberada:.0f} MB':>16s}  "
          f"{'OK' if not f_recarga else 'FALLO'}")
    for f in f_recarga:
        print("  FALLO:", f)

    print(f"\nTOTAL errores de palabra: {total_err} de {total_pal} "
          f"({100 * total_err / total_pal:.1f} %) | escenarios con fallo: {fallos} | unitarias con fallo: {len(fallos_unit)}")
    sys.exit(1 if fallos or fallos_unit else 0)


if __name__ == "__main__":
    main()
