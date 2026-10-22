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
    alimentar({n: a for n, (a, _) in pistas.items()}, fuentes, motor.cola_audio, velocidad)
    time.sleep(1.0)
    motor.stop.set()
    motor.hilo.join(120)
    msgs = list(cola_ui.queue)
    assert msgs and msgs[-1][0] == "detenido", "el hilo no avisó 'detenido'"
    return {n: " ".join(v[1] for k, v in msgs if k == "final" and v[0] == n) for n in pistas}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocidad", type=float, default=1.0, help="1.0 = tiempo real (lo representativo)")
    args = ap.parse_args()

    motor = T.Motor(queue.Queue())
    motor.cargar_modelo()
    n_unit, fallos_unit = pruebas_unitarias(motor)
    print(f"\nunitarias: {n_unit - len(fallos_unit)}/{n_unit} OK")
    for f in fallos_unit:
        print("  FALLO:", f)

    total_err = total_pal = fallos = 0
    print(f"\n{'escenario':28s} {'fuente':6s} {'errores':>8s} {'palabras':>9s}  resultado")
    for nombre, pistas in escenarios().items():
        textos = correr_escenario(motor, pistas, args.velocidad)
        for fuente, (_, ref) in pistas.items():
            err, pal = errores_palabras(ref, textos[fuente])
            ok = (1 - err / pal >= UMBRAL_ACIERTO) if pal else not textos[fuente]
            total_err += err
            total_pal += pal
            fallos += not ok
            print(f"{nombre:28s} {fuente:6s} {err:8d} {pal:9d}  {'OK' if ok else 'FALLO'}", flush=True)

    print(f"\nTOTAL errores de palabra: {total_err} de {total_pal} "
          f"({100 * total_err / total_pal:.1f} %) | escenarios con fallo: {fallos} | unitarias con fallo: {len(fallos_unit)}")
    sys.exit(1 if fallos or fallos_unit else 0)


if __name__ == "__main__":
    main()
