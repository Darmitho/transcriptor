"""Utilidades compartidas por la batería de regresión y la medición de recursos."""

import os
import sys
import time
import wave
import difflib
import tempfile
import contextlib

import numpy as np
from scipy.signal import resample_poly

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import transcriptor as T  # noqa: E402  (después de ajustar sys.path)

AUDIO = os.path.join(RAIZ, "tests", "audio")
TASA = 48000                            # Como un dispositivo WASAPI típico
DISP = {"maxInputChannels": 1, "defaultSampleRate": float(TASA), "index": 0, "name": "prueba"}

PARTES = [
    "Primero revisamos el presupuesto del trimestre",
    "luego vimos las fechas de entrega del proyecto",
    "después hablamos de los riesgos principales",
    "y finalmente acordamos los próximos pasos con el cliente",
    "la reunión siguiente será el martes a las diez",
]
REF_FRASE = "Buenos días a todos. Hoy vamos a revisar el presupuesto del proyecto y las fechas de entrega."


@contextlib.contextmanager
def app_aislada(sin_modelo=True):
    """Para crear la ventana (App) en pruebas o mediciones sin tocar las transcripciones
    ni la configuración reales. Con sin_modelo no carga Whisper. Devuelve la carpeta temporal."""
    carpeta = tempfile.mkdtemp(prefix="prueba_")
    antes = (T.CARPETA_SALIDA, T.CONFIG_ARCHIVO, T.Motor.cargar_modelo)
    T.CARPETA_SALIDA, T.CONFIG_ARCHIVO = carpeta, os.path.join(carpeta, "config.json")
    if sin_modelo:
        T.Motor.cargar_modelo = lambda self: None
    try:
        yield carpeta
    finally:
        T.CARPETA_SALIDA, T.CONFIG_ARCHIVO, T.Motor.cargar_modelo = antes


def wav(nombre, recortar=False):
    """Audio de prueba a 48 kHz float32 (se guarda a 16 kHz para ocupar menos)."""
    with wave.open(os.path.join(AUDIO, nombre + ".wav")) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    a = resample_poly(a, TASA // 16000, 1).astype(np.float32)
    if recortar:                        # Sin el silencio inicial y final del TTS
        idx = np.nonzero(np.abs(a) > 0.01)[0]
        a = a[idx[0]:idx[-1] + 1]
    return a


def silencio(segundos):
    return np.zeros(int(TASA * segundos), np.float32)


def texto_continuo():
    with open(os.path.join(AUDIO, "continuo.txt"), encoding="utf-8-sig") as f:
        return f.read().strip()


def frases_con_pausas(pausa=0.25):
    """Las 5 partes seguidas con respiraciones cortas (una frase larga de ~16 s)."""
    trozos = [silencio(1)]
    for i in range(5):
        trozos += [wav(f"parte{i}", recortar=True), silencio(pausa)]
    return np.concatenate(trozos + [silencio(2)]), " ".join(PARTES)


def errores_palabras(referencia, hipotesis):
    """Palabras sustituidas + borradas + insertadas, tras normalizar."""
    r, h = T._normalizar(referencia).split(), T._normalizar(hipotesis).split()
    ops = difflib.SequenceMatcher(None, r, h).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal"), len(r)


def alimentar(pistas, fuentes, cola, velocidad=1.0, fines_voz=None):
    """Envía cada pista a su fuente en trozos de TROZO segundos, a ritmo real / velocidad.
    fines_voz = {nombre: [índice de muestra donde termina la voz, ...]}: devuelve, por
    fuente, el instante (perf_counter) en que se envió cada uno de esos finales."""
    paso = int(TASA * T.TROZO)
    largo = max(len(a) for a in pistas.values())
    pendientes = {n: sorted(v) for n, v in (fines_voz or {}).items()}
    enviados = {n: [] for n in pendientes}
    t = time.perf_counter()
    for i in range(0, largo, paso):
        for n, a in pistas.items():
            if i < len(a):
                cola.put((fuentes[n], a[i:i + paso], time.time()))   # Como el callback: trozo e inicio
            while pendientes.get(n) and pendientes[n][0] <= i + paso:
                pendientes[n].pop(0)
                enviados[n].append(time.perf_counter())
        t += T.TROZO / velocidad
        time.sleep(max(0.0, t - time.perf_counter()))
    return enviados
