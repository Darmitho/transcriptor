"""Microbenchmark: ¿un parcial en lote de 2 (Yo + Otros) cuesta menos que dos llamadas?

Uso:  venv\\Scripts\\python.exe tools\\bench_lote.py
Compara, con la misma configuración de los parciales (beam 1, prompt de vocabulario),
dos llamadas separadas contra una llamada de BatchedInferencePipeline con dos clips.
Criterio acordado: la optimización solo entra si el lote ahorra al menos un 30 %.
"""

import os
import statistics
import sys
import threading
import time

import numpy as np
import pynvml
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from comun import T, wav  # noqa: E402

from faster_whisper import BatchedInferencePipeline  # noqa: E402

REPETICIONES = 5
UMBRAL_AHORRO = 0.30


class PicoVram:
    """Máximo de VRAM usada (todo el equipo) mientras dura el bloque with."""

    def __init__(self, gpu):
        self.gpu, self.pico, self._seguir = gpu, 0.0, False

    def _muestrear(self):
        while self._seguir:
            self.pico = max(self.pico, pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 2**20)
            time.sleep(0.005)

    def __enter__(self):
        self._seguir = True
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._seguir = False
        self._hilo.join()


def a16(x):
    return resample_poly(x, 1, 3).astype(np.float32)


def main():
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    motor = T.Motor(None)
    motor.cola_ui = __import__("queue").Queue()
    motor.cargar_modelo()
    modelo = motor.modelo
    lote = BatchedInferencePipeline(model=modelo)
    idioma, prompt = motor.contexto
    opciones = dict(language=idioma, beam_size=1, condition_on_previous_text=False, without_timestamps=True,
                    initial_prompt=prompt)

    cont = wav("continuo")
    pares = {
        "4 s + 4 s": (a16(wav("v1", True)), a16(wav("v5", True))),
        "8 s + 6 s": (a16(cont[:48000 * 8]), a16(wav("frase", True))),
        "15 s + 15 s": (a16(cont[:48000 * 15]), a16(cont[48000 * 8:48000 * 23])),
    }

    def separadas(a, b):
        ta = " ".join(s.text.strip() for s in modelo.transcribe(a, vad_filter=False, **opciones)[0])
        tb = " ".join(s.text.strip() for s in modelo.transcribe(b, vad_filter=False, **opciones)[0])
        return ta, tb

    def en_lote(a, b):
        fin_a = len(a) / 16000
        segs = list(lote.transcribe(np.concatenate([a, b]), batch_size=2,
                                    clip_timestamps=[{"start": 0, "end": fin_a},
                                                     {"start": fin_a, "end": (len(a) + len(b)) / 16000}],
                                    **opciones)[0])
        ta = " ".join(s.text.strip() for s in segs if s.start < fin_a - 0.01)
        tb = " ".join(s.text.strip() for s in segs if s.start >= fin_a - 0.01)
        return ta, tb

    base = pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20
    print(f"VRAM con el modelo cargado y en reposo: {base:.0f} MB\n")
    print(f"{'par':12s} {'2 llamadas':>11s} {'lote de 2':>10s} {'ahorro':>7s} {'VRAM extra 2 llam.':>19s} "
          f"{'VRAM extra lote':>16s}  mismo texto")
    for nombre, (a, b) in pares.items():
        separadas(a, b)
        en_lote(a, b)                   # Calentar ambos caminos
        resultados = {}
        for etiqueta, f in (("sep", separadas), ("lote", en_lote)):
            tiempos, pico = [], 0.0
            for _ in range(REPETICIONES):
                with PicoVram(gpu) as p:
                    t0 = time.perf_counter()
                    textos = f(a, b)
                    tiempos.append(time.perf_counter() - t0)
                pico = max(pico, p.pico)
            resultados[etiqueta] = (statistics.median(tiempos), pico - base, textos)
        t_sep, v_sep, tx_sep = resultados["sep"]
        t_lote, v_lote, tx_lote = resultados["lote"]
        ahorro = 1 - t_lote / t_sep
        igual = [T._normalizar(x) == T._normalizar(y) for x, y in zip(tx_sep, tx_lote)]
        print(f"{nombre:12s} {t_sep * 1000:9.0f} ms {t_lote * 1000:8.0f} ms {ahorro * 100:6.0f} % "
              f"{v_sep:16.0f} MB {v_lote:13.0f} MB  {'sí' if all(igual) else 'NO ' + str(igual)}")
        if not all(igual):
            for x, y in zip(tx_sep, tx_lote):
                if T._normalizar(x) != T._normalizar(y):
                    print(f"     separadas: {x!r}\n     lote:      {y!r}")
    print(f"\nCriterio: el lote entra solo si ahorra al menos {UMBRAL_AHORRO * 100:.0f} % en los tres pares.")


if __name__ == "__main__":
    main()
