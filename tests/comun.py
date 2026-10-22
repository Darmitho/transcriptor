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
def app_aislada(sin_modelo=True, atajo_global=""):
    """Para crear la ventana (App) en pruebas o mediciones sin tocar las transcripciones
    ni la configuración reales. Con sin_modelo no carga Whisper. Sin atajo global salvo que
    se pida uno: registrarlo ocupa esa combinación en todo el sistema. Devuelve la carpeta."""
    carpeta = tempfile.mkdtemp(prefix="prueba_")
    antes = (T.CARPETA_SALIDA, T.CONFIG_ARCHIVO, T.Motor.cargar_modelo,
             T.Configuracion.INICIAL["atajo_subtitulos"])
    T.CARPETA_SALIDA, T.CONFIG_ARCHIVO = carpeta, os.path.join(carpeta, "config.json")
    T.Configuracion.INICIAL["atajo_subtitulos"] = atajo_global
    if sin_modelo:
        T.Motor.cargar_modelo = lambda self: None
    try:
        yield carpeta
    finally:
        (T.CARPETA_SALIDA, T.CONFIG_ARCHIVO, T.Motor.cargar_modelo,
         T.Configuracion.INICIAL["atajo_subtitulos"]) = antes


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


def ruido_rosa(segundos, semilla=0):
    """Ruido de banda ancha, parecido a efectos, multitud o lluvia."""
    n = int(TASA * segundos)
    espectro = np.fft.rfft(np.random.default_rng(semilla).standard_normal(n))
    espectro /= np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / TASA), 20))
    x = np.fft.irfft(espectro, n)
    return (x / np.sqrt(np.mean(x ** 2)) * 0.03).astype(np.float32)


def musica_sintetica(segundos, semilla=0):
    """Acordes con armónicos, bajo, bombo y platillo: un fondo musical simple."""
    rng = np.random.default_rng(semilla)
    t = np.arange(int(TASA * segundos)) / TASA
    x = np.zeros_like(t)
    acordes = [(220, 277, 330), (196, 247, 294), (175, 220, 262), (165, 208, 247)]
    for k in range(int(segundos // 2) + 1):
        ac, m = acordes[k % 4], (t >= k * 2) & (t < (k + 1) * 2)
        for f in ac:
            for h, a in ((1, 1.0), (2, 0.5), (3, 0.25)):
                x[m] += a * np.sin(2 * np.pi * f * h * t[m])
        x[m] += 1.5 * np.sin(2 * np.pi * ac[0] / 2 * t[m])
    x += ((t % 0.5) < 0.05) * np.sin(2 * np.pi * 60 * t) * 3 * np.exp(-(t % 0.5) * 40)
    x += ((t % 0.25) < 0.02) * rng.standard_normal(len(t)) * 0.8
    return (x / np.sqrt(np.mean(x ** 2)) * 0.03).astype(np.float32)


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
                trozo = a[i:i + paso]
                f = fuentes[n]           # Lo mismo que hace el callback de audio
                f.nivel, f.t_nivel = float(np.sqrt(np.mean(trozo * trozo))), time.time()
                cola.put((f, trozo, time.time()))
            while pendientes.get(n) and pendientes[n][0] <= i + paso:
                pendientes[n].pop(0)
                enviados[n].append(time.perf_counter())
        t += T.TROZO / velocidad
        time.sleep(max(0.0, t - time.perf_counter()))
    return enviados
