"""
Transcriptor en vivo del audio del PC (Windows).

Captura lo que suena por la salida de audio predeterminada (WASAPI loopback),
lo transcribe con faster-whisper en la GPU y lo muestra en una ventana:
el texto en gris es provisional (se va corrigiendo mientras la persona habla)
y pasa a blanco cuando la frase termina.
"""

import os
import re
import sys
import glob
import queue
import bisect
import logging
import threading
import time
import json
import datetime
import unicodedata
from collections import Counter, deque
from math import gcd, log10


# ======================= CONFIGURACIÓN =======================
MODELO = "large-v3-turbo"       # Si va lento: "medium" o "small"
DISPOSITIVO = "cuda"
TIPO_COMPUTO = "int8_float16"   # Bajo uso de VRAM, buena calidad en RTX 3050
IDIOMAS = {"Español": "es", "English": "en"}  # Opciones del selector de la ventana
IDIOMA = "es"                   # Idioma al abrir el programa
SR_MODELO = 16000

INTERVALO_PARCIAL = 1.0         # Cada cuántos segundos se refresca el texto provisional
SILENCIO_FIN = 1.0              # Segundos sin voz (según el VAD) que cierran una frase.
                                # Con 0.7 las pausas a mitad de idea partían frases en fragmentos
FRASE_LARGA = 10.0              # Desde aquí, una pausa corta (PAUSA_CORTE) ya cierra la frase
PAUSA_CORTE = 0.2               # Micro-pausa (respiración) suficiente para cortar una frase larga
MAX_FRASE = 20.0                # Sin pausas hasta aquí: se corta antes de la última palabra
MAX_ARRASTRE = 3.0              # Audio máximo que pasa a la frase siguiente tras ese corte
TROZO = 0.1                     # Duración de cada bloque de audio capturado (segundos)
NOMBRE_OTROS = "Otros"          # Etiqueta del audio del PC (lo que dicen los demás)
NOMBRE_YO = "Yo"                # Etiqueta del micrófono (lo que digo yo)

# --- Detección de voz (Silero VAD, incluido en faster-whisper, corre en CPU) ---
VAD_UMBRAL = 0.5                # Probabilidad para considerar que empieza la voz
VAD_UMBRAL_FIN = 0.35           # Una vez en voz, sigue siéndolo mientras supere este valor
VOZ_MIN = 0.25                  # Segundos de voz mínimos para llamar a Whisper
PRE_ROLL = 0.3                  # Audio previo al inicio de la voz que se conserva
RECORTE = 0.3                   # Margen de silencio que se deja al inicio y fin de la frase

# --- Filtros por segmento contra alucinaciones ---
MAX_COMPRESION = 2.4            # compression_ratio mayor = texto repetitivo en bucle
MAX_NO_VOZ = 0.8                # no_speech_prob mayor = el modelo cree que no hay voz
MIN_LOGPROB = -1.0              # avg_logprob menor = confianza muy baja
# Frases típicas de alucinación: solo se descartan si además hay señales de duda
# Créditos que Whisper inventa al final de un video o con música ("Closed Captioning by...").
# Nunca son habla real: se descartan siempre (sobre el texto normalizado, sin tildes)
CREDITOS_INVENTADOS = re.compile(r"^(closed )?caption(s|ing|ed)? by|^subtitles? by|^transcri(bed|ption) by|"
                                 r"^subtitulos? (realizados )?por|^traduccion (realizada )?por")
FRASES_SOSPECHOSAS = (
    "gracias",
    "muchas gracias",
    "gracias por ver el video",
    "suscribete",
    "subtitulos realizados por la comunidad de amara org",
    "musica",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "music",
)
SOSP_LOGPROB = -0.5             # Sospechosa con avg_logprob menor: se descarta
SOSP_NO_VOZ = 0.3               # Sospechosa con no_speech_prob mayor: se descarta
SOSP_VOZ_MIN = 0.3              # Sospechosa, sola en la frase, con menos voz que esto: se descarta

# --- Vocabulario propio: nombres y jerga ---
# No se traducen (ver NO_TRADUCIR). Opcionalmente también se le dan a Whisper como contexto.
VOCABULARIO = ("Claude", "CriptoBuzz", "pullback", "order block", "RSI")
# Darle el vocabulario a Whisper como prompt ("Hoy hablamos de A, B y C."). Desactivado: con
# música o ruido Whisper lo repite como texto inventado (medido sin VAD: 6 de 60 trozos difíciles
# en español, 2 de 60 en inglés; sin prompt, 0) y con voz clara no mejoró la precisión.
USAR_PROMPT_VOCABULARIO = False
# Correcciones de escritura conocidas, sin riesgo de inventar texto: patrón (sin distinguir
# mayúsculas, palabra completa) -> forma correcta. Se aplican a todo lo transcrito.
CORRECCIONES = {
    r"crypto ?bu[sz]+": "CriptoBuzz",
    r"crypto ?boss": "CriptoBuzz",
}

# --- Traducción inglés -> español de las frases finales (opcional) ---
# Opus-MT en-es (Apache-2.0) sobre CTranslate2, en CPU para no quitarle GPU a Whisper.
# Se convierte una vez con tools\convertir_traduccion.md y luego funciona sin red.
MODELO_TRADUCCION = os.path.join(os.path.expanduser("~"), ".cache", "transcriptor", "traduccion", "opus-mt-en-es")
TRADUCCION_HILOS = 1                # 1 hilo: ~0.2 s por frase y ~10 % de un núcleo; con 4, ~0.1 s pero ~50 %
                                    # (los hilos de OpenMP quedan girando en espera entre frases)
# Términos que quedan en inglés al traducir, además de los de VOCABULARIO
NO_TRADUCIR = ("stop loss",)

# Carpeta local (fuera de OneDrive) donde se guardan las transcripciones automáticas
CARPETA_SALIDA = os.path.expandvars(r"%USERPROFILE%\Documents\Transcripciones")
# True = nunca conectarse a Hugging Face (el modelo debe estar ya descargado).
# Poner False solo una vez si hace falta descargar un modelo nuevo.
MODO_OFFLINE = True
# Tras estos segundos detenido se libera la VRAM del modelo (~900 MB) y se recarga al
# iniciar (~5 s, sin perder audio: espera en la cola). None = mantenerlo siempre en la GPU.
LIBERAR_VRAM_TRAS = 300
# Preferencias de la ventana (idioma, tamaño, colores...): local, fuera de OneDrive.
# El micrófono no se recuerda a propósito: siempre arranca apagado.
CONFIG_ARCHIVO = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                              "Transcriptor", "config.json")
# =============================================================

if MODO_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"   # Debe definirse antes de importar faster_whisper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("transcriptor")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


# --- Hacer visibles las DLL de CUDA instaladas por pip (nvidia-cublas / nvidia-cudnn) ---
def _agregar_dlls_cuda():
    if sys.platform != "win32":
        return
    import site
    bases = list(site.getsitepackages()) + [site.getusersitepackages(),
                                            os.path.join(sys.prefix, "Lib", "site-packages")]
    for base in bases:
        for bin_dir in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
            except (FileNotFoundError, OSError):
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_agregar_dlls_cuda()

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
from tkinter import font as tkfont


def _normalizar(texto):
    """Minúsculas, sin tildes ni puntuación: '¡Gracias!' -> 'gracias'."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


# Frase del prompt y conjunción por idioma: un prompt en otro idioma empuja a traducir
PLANTILLAS_PROMPT = {"es": ("Hoy hablamos de {}.", " y "), "en": ("Today we are talking about {}.", " and ")}


def _prompt_vocabulario(vocabulario, idioma):
    """('A', 'B', 'C'), 'es' -> 'Hoy hablamos de A, B y C.'"""
    if not vocabulario:
        return None
    frase, y = PLANTILLAS_PROMPT[idioma]
    lista = ", ".join(vocabulario[:-1]) + y + vocabulario[-1] if len(vocabulario) > 1 else vocabulario[0]
    return frase.format(lista)


_PATRONES_CORRECCION = [(re.compile(r"\b" + p + r"\b", re.IGNORECASE), r) for p, r in CORRECCIONES.items()]


def _corregir(texto):
    """Aplica CORRECCIONES (escritura de nombres propios que Whisper no conoce)."""
    for patron, reemplazo in _PATRONES_CORRECCION:
        texto = patron.sub(reemplazo, texto)
    return texto


def _oraciones(texto):
    """'What? No. Wait!' -> ['What?', 'No.', 'Wait!']. El traductor pierde a veces la primera
    oración corta si recibe varias juntas: se traducen por separado, en una sola llamada."""
    return [o for o in re.split(r"(?<=[.!?¿¡])\s+", texto.strip()) if o]


def _proteger_terminos(texto, terminos):
    """Cambia cada término por un marcador (__0__, __1__...) que el traductor deja tal cual.
    Devuelve el texto marcado y los términos en el orden de sus marcadores."""
    if not terminos:
        return texto, []
    patron = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terminos, key=len, reverse=True)) + r")\b",
                        re.IGNORECASE)
    usados = []
    marcado = patron.sub(lambda m: (usados.append(m.group(0)), f"__{len(usados) - 1}__")[1], texto)
    return marcado, usados


def _restaurar_terminos(traduccion, usados):
    """Vuelve a poner los términos. None si el traductor deformó algún marcador."""
    for i, termino in enumerate(usados):
        if f"__{i}__" not in traduccion:
            return None
        traduccion = traduccion.replace(f"__{i}__", termino)
    return traduccion


def _quitar_solape(previo, nuevo):
    """Quita del inicio de 'nuevo' las palabras (hasta 3) que repiten el final de 'previo'."""
    a = _normalizar(previo).split()
    b = nuevo.split()
    nb = [_normalizar(w) for w in b]
    for k in range(min(3, len(a), len(b)), 0, -1):
        if a[-k:] == nb[:k]:
            return " ".join(b[k:])
    return nuevo


class DetectorVoz:
    """Silero VAD incremental: cuántos segundos de cada bloque de audio son voz."""
    VENTANA = 512                   # Muestras a 16 kHz (32 ms) que evalúa Silero

    def __init__(self, up, down):
        self.modelo = get_vad_model()
        self.up, self.down = up, down
        self.resto = np.zeros(0, np.float32)
        self.previa = np.zeros(self.VENTANA, np.float32)  # Da contexto a la primera ventana
        self.en_voz = False
        self.t_vad = self.t_audio = 0.0

    def segundos_voz(self, x):
        t0 = time.perf_counter()
        y = np.concatenate([self.resto, resample_poly(x, self.up, self.down).astype(np.float32)])
        n = len(y) // self.VENTANA * self.VENTANA
        self.resto = y[n:]
        ventanas_voz = 0
        if n:
            # La primera probabilidad corresponde a la ventana previa (ya evaluada): se ignora
            probs = self.modelo(np.concatenate([self.previa, y[:n]]))[1:]
            self.previa = y[n - self.VENTANA:n]
            for p in probs:   # Histéresis: cuesta más entrar en voz que mantenerse
                self.en_voz = p >= (VAD_UMBRAL_FIN if self.en_voz else VAD_UMBRAL)
                ventanas_voz += self.en_voz
        self.t_vad += time.perf_counter() - t0
        self.t_audio += len(x) / (SR_MODELO * self.down / self.up)
        return ventanas_voz * self.VENTANA / SR_MODELO

    def ms_por_segundo(self):
        return 1000 * self.t_vad / max(self.t_audio, 1e-6)


class Fuente:
    """Un origen de audio (loopback, micrófono...): su captura, su VAD y la frase en curso.
    El callback de audio solo encola; todo lo demás corre en el hilo del Motor."""

    def __init__(self, nombre, disp, cola):
        self.nombre = nombre
        self.disp = disp
        self.cola = cola                    # Compartida por todas las fuentes: (fuente, trozo, inicio)
        self.canales = disp["maxInputChannels"]
        self.tasa = int(disp["defaultSampleRate"])
        g = gcd(SR_MODELO, self.tasa)
        self.up, self.down = SR_MODELO // g, self.tasa // g
        self.stream = None
        self.vad = DetectorVoz(self.up, self.down)
        self.preroll = deque(maxlen=max(1, round(PRE_ROLL / TROZO)))   # (trozo, inicio)
        # Frase actual: audio, segundos de voz y hora (epoch) en que empezó cada trozo
        self.trozos, self.voz, self.inicios = [], [], []
        self.duracion = self.silencio = 0.0  # En tiempo de audio
        self.previo = ""                    # Texto antes de un corte en palabra (para quitar solapes)
        self.ultimo_audio = self.ultimo_parcial = time.time()
        self.stats = Counter()
        self.retirar = False                # El hilo del Motor la cierra y la quita de la lista
        self.nivel, self.t_nivel = 0.0, 0.0  # RMS del último trozo capturado y cuándo, para el medidor

    # ---------- Captura ----------
    def abrir(self, pa):
        self.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.canales,
            rate=self.tasa,
            input=True,
            input_device_index=self.disp["index"],
            frames_per_buffer=int(self.tasa * TROZO),
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        x = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.canales).mean(axis=1)
        # El callback llega cuando el bloque terminó de capturarse: su inicio fue antes
        # El nivel se mide aquí y no en el hilo del Motor: así el medidor no se congela
        # mientras la GPU transcribe
        self.nivel, self.t_nivel = float(np.sqrt(np.mean(x * x))), time.time()
        self.cola.put((self, x.copy(), time.time() - frame_count / self.tasa))
        return (None, pyaudio.paContinue)

    def cerrar_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()   # Espera a que termine el callback en curso
                self.stream.close()
            finally:
                self.stream = None

    # ---------- Armado de frases ----------
    def agregar(self, x, inicio):
        v = self.vad.segundos_voz(x)
        self.ultimo_audio = time.time()
        self.silencio = 0.0 if v > 0 else self.silencio + len(x) / self.tasa
        if not self.trozos:
            if v == 0:                      # Sin voz y sin frase abierta: solo pre-roll
                self.preroll.append((x, inicio))
                return
            for p, t in self.preroll:
                self.trozos.append(p)
                self.voz.append(0.0)
                self.inicios.append(t)
                self.duracion += len(p) / self.tasa
            self.preroll.clear()
        self.trozos.append(x)
        self.voz.append(v)
        self.inicios.append(inicio)
        self.duracion += len(x) / self.tasa

    def inicio_voz(self):
        """Hora (epoch) en que empezó la voz de la frase actual: la del primer trozo con voz."""
        return next(t for t, v in zip(self.inicios, self.voz) if v > 0)

    def motivo_cierre(self, incluir_sin_audio=True):
        if not self.trozos:
            return None
        if self.silencio >= SILENCIO_FIN:
            return "silencio"
        if self.duracion >= FRASE_LARGA and self.silencio >= PAUSA_CORTE:
            return "pausa"
        if self.duracion >= MAX_FRASE:
            return "maximo"
        # WASAPI loopback puede dejar de entregar audio cuando nada suena
        if incluir_sin_audio and time.time() - self.ultimo_audio >= SILENCIO_FIN:
            return "sin_audio"
        return None

    def reiniciar_frase(self, trozos=(), voz=(), inicios=()):
        """Empieza una frase nueva, opcionalmente con el audio arrastrado de un corte."""
        self.trozos, self.voz, self.inicios = list(trozos), list(voz), list(inicios)
        self.duracion = sum(len(x) for x in self.trozos) / self.tasa

    def resumen(self):
        stats = Counter(self.stats)
        cierres = {k[7:]: stats.pop(k) for k in list(stats) if k.startswith("cierre_")}
        log.info("resumen [%s]: frases=%d omitidas_vad=%d cierres:%s descartes:%s | VAD %.2f ms por s de audio",
                 self.nombre, stats.pop("frases", 0), stats.pop("omitida_vad", 0),
                 "".join(f" {k}={v}" for k, v in cierres.items()) or " ninguno",
                 "".join(f" {k}={v}" for k, v in stats.items()) or " ninguno", self.vad.ms_por_segundo())


class FuenteApp(Fuente):
    """Fuente "Otros" que escucha solo una aplicación, o todo menos una, en vez de todo el
    sistema (captura_app.py: application loopback de Windows). Entrega trozos de TROZO
    segundos con su hora y su nivel, igual que el callback de PyAudio."""

    def __init__(self, nombre, app, excluir, cola):
        from captura_app import TASA_CAPTURA
        modo = "todo menos" if excluir else "solo"
        super().__init__(nombre, {"maxInputChannels": 1, "defaultSampleRate": float(TASA_CAPTURA), "index": None,
                                  "name": f"{modo} {_nombre_app(app)}"}, cola)
        self.app, self.excluir = app, excluir
        self.captura = None
        self.pendiente = np.zeros(0, np.float32)

    def abrir(self, pa=None):
        from captura_app import CapturaApp, pid_de_app
        pid = pid_de_app(self.app)
        if pid is None:
            raise RuntimeError(f"{_nombre_app(self.app)} no está abierto.")
        self.captura = CapturaApp(pid, self._recibir, excluir=self.excluir).iniciar()

    def _recibir(self, bloque):
        """Junta los paquetes de Windows (~10 ms) en trozos de TROZO segundos."""
        self.pendiente = np.concatenate([self.pendiente, bloque])
        n = int(self.tasa * TROZO)
        while len(self.pendiente) >= n:
            x, self.pendiente = self.pendiente[:n], self.pendiente[n:]
            fin = time.time() - len(self.pendiente) / self.tasa
            self.nivel, self.t_nivel = float(np.sqrt(np.mean(x * x))), time.time()
            self.cola.put((self, x, fin - n / self.tasa))

    def cerrar_stream(self):
        if self.captura is not None:
            self.captura.detener()
            self.captura = None


def _nombre_app(exe):
    """'Discord.exe' -> 'Discord', para mostrar."""
    return exe[:-4] if exe.lower().endswith(".exe") else exe


class Motor:
    """Un solo modelo Whisper y un solo hilo que atiende a todas las fuentes de audio
    (la GPU de 4 GB no da para dos modelos). Los finales tienen prioridad sobre los parciales."""

    def __init__(self, cola_ui):
        self.cola_ui = cola_ui
        self.cola_audio = queue.Queue()
        self.stop = threading.Event()
        self.modelo = None
        self.pa = None
        self.fuentes = []
        self.hilo = None
        self.vram_liberada = False          # Modelo fuera de la GPU: se recarga al iniciar
        self.transcribiendo = False         # Whisper está procesando ahora mismo
        self.cambiar_idioma(IDIOMA)

    def atraso(self):
        """Segundos de audio por fuente esperando en la cola."""
        return self.cola_audio.qsize() / max(1, len(self.fuentes)) * TROZO

    def cambiar_idioma(self, idioma):
        """Se puede llamar en marcha desde la UI: aplica desde la próxima transcripción.
        Idioma y prompt van en una sola tupla para que el hilo nunca lea uno sin el otro."""
        self.contexto = (idioma, _prompt_vocabulario(VOCABULARIO, idioma) if USAR_PROMPT_VOCABULARIO else None)

    # ---------- Modelo ----------
    def cargar_modelo(self):
        self.cola_ui.put(("estado", f"Cargando modelo {MODELO} en GPU..."))
        try:
            self.modelo = WhisperModel(MODELO, device=DISPOSITIVO, compute_type=TIPO_COMPUTO,
                                       local_files_only=MODO_OFFLINE)
            # Prueba rápida para detectar si faltan librerías de CUDA
            list(self.modelo.transcribe(np.zeros(SR_MODELO, dtype=np.float32), language=IDIOMA)[0])
            self.cola_ui.put(("estado", f"Listo ({MODELO}, GPU)"))
        except Exception as e:
            self.cola_ui.put(("estado", "GPU no disponible, usando CPU con modelo small..."))
            log.error("Error al cargar %s en %s: %s", MODELO, DISPOSITIVO, e)
            try:
                self.modelo = WhisperModel("small", device="cpu", compute_type="int8",
                                           local_files_only=MODO_OFFLINE)
            except Exception as e2:
                log.error("Error al cargar small en CPU: %s", e2)
                self.cola_ui.put(("estado", "No se pudo cargar ningún modelo (ver consola; "
                                            "¿falta descargarlo con MODO_OFFLINE = False?)"))
                return
            self.cola_ui.put(("estado", "Listo (small, CPU)"))
        self.cola_ui.put(("modelo_listo", None))

    # ---------- Captura ----------
    def _dispositivo_loopback(self):
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        salida = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if salida.get("isLoopbackDevice"):
            return salida
        for lb in self.pa.get_loopback_device_info_generator():
            if salida["name"] in lb["name"]:
                return lb
        raise RuntimeError("No se encontró un dispositivo loopback para la salida predeterminada.")

    def _dispositivo_mic(self):
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        if wasapi["defaultInputDevice"] < 0:
            raise RuntimeError("No hay un micrófono predeterminado en Windows.")
        return self.pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    def hilo_activo(self):
        return self.hilo is not None and self.hilo.is_alive()

    def liberar_vram(self):
        """Saca el modelo de la GPU si la captura está detenida. Devuelve si lo hizo.
        Solo aplica en GPU: el modelo de respaldo en CPU no ocupa VRAM."""
        if self.modelo is None or self.vram_liberada or self.hilo_activo() or self.modelo.model.device != "cuda":
            return False
        self.modelo.model.unload_model(to_cpu=False)
        self.vram_liberada = True
        log.info("modelo descargado de la GPU (captura detenida)")
        return True

    def _recargar_si_hace_falta(self):
        """En el hilo del Motor al empezar: el audio que llega mientras tanto espera en la cola."""
        if self.vram_liberada:
            self.cola_ui.put(("estado", "Recargando el modelo en la GPU..."))
            t0 = time.perf_counter()
            self.modelo.model.load_model()
            self.vram_liberada = False
            log.info("modelo recargado en la GPU en %.1f s", time.perf_counter() - t0)
            self.cola_ui.put(("estado", "Escuchando"))

    def activar_mic(self, activo):
        """Abre o cierra el micrófono con la captura en marcha (se llama desde la UI)."""
        if self.pa is None:
            return
        mic = next((f for f in self.fuentes if f.nombre == NOMBRE_YO and not f.retirar), None)
        if activo and mic is None:
            mic = Fuente(NOMBRE_YO, self._dispositivo_mic(), self.cola_audio)
            mic.abrir(self.pa)
            self.fuentes.append(mic)
            self.cola_ui.put(("estado", f"Micrófono activo: {mic.disp['name']}"))
        elif not activo and mic is not None:
            mic.cerrar_stream()
            mic.retirar = True              # El hilo cierra su última frase y la quita
            self.cola_ui.put(("estado", "Micrófono apagado"))

    def _abrir_otros(self, escuchar):
        """Fuente "Otros" abierta: todo el sistema, solo una app o todo menos una.
        escuchar = None (todo el sistema) o ("solo" | "menos", "Discord.exe")."""
        if escuchar is not None:
            modo, app = escuchar
            try:
                otros = FuenteApp(NOMBRE_OTROS, app, modo == "menos", self.cola_audio)
                otros.abrir()
                return otros
            except RuntimeError:
                if modo == "solo":
                    raise
                # "Todo menos" una app cerrada es todo el sistema
                self.cola_ui.put(("estado", f"{_nombre_app(app)} no está abierto: se escucha todo el sistema"))
        otros = Fuente(NOMBRE_OTROS, self._dispositivo_loopback(), self.cola_audio)
        otros.abrir(self.pa)
        return otros

    def iniciar(self, con_mic=False, escuchar=None):
        if self.hilo_activo():
            raise RuntimeError("La captura anterior aún se está cerrando, intenta de nuevo.")
        # Cola y evento nuevos por sesión: nada de una sesión anterior se mezcla con esta
        self.stop = threading.Event()
        self.cola_audio = queue.Queue()
        self.pa = pyaudio.PyAudio()
        try:
            otros = self._abrir_otros(escuchar)
        except Exception:
            self.pa.terminate()
            self.pa = None
            raise
        self.fuentes = [otros]
        self.hilo = threading.Thread(target=self._bucle, args=(self.stop, self.cola_audio, self.fuentes),
                                     daemon=True)
        self.hilo.start()
        self.cola_ui.put(("estado", f"Escuchando: {otros.disp['name']}"))
        if con_mic:
            try:
                self.activar_mic(True)
            except Exception as e:          # Sin micrófono se sigue con el audio del PC
                self.cola_ui.put(("error_mic", str(e)))

    def detener(self):
        """Corta la captura. El hilo cierra las frases pendientes y avisa con 'detenido'."""
        try:
            for f in self.fuentes:
                f.cerrar_stream()
        finally:
            if self.pa:
                self.pa.terminate()
            self.pa = None
            self.stop.set()
        self.cola_ui.put(("estado", "Deteniendo..."))

    def reconectar(self, con_mic, escuchar=None):
        """Vuelve a abrir la captura con los dispositivos predeterminados actuales de Windows
        (p. ej. tras conectar audífonos) o con otra app a escuchar, sin detener el hilo ni
        recargar el modelo. Se llama desde la UI."""
        if self.pa is None:
            return
        # Las fuentes actuales se retiran: el hilo transcribe lo que alcanzó a llegar
        for f in self.fuentes:
            if not f.retirar:
                f.cerrar_stream()
                f.retirar = True
        # PortAudio solo enumera los dispositivos al iniciarse: hay que reiniciarlo
        self.pa.terminate()
        self.pa = pyaudio.PyAudio()
        otros = self._abrir_otros(escuchar)
        self.fuentes.append(otros)
        self.cola_ui.put(("estado", f"Reconectado: {otros.disp['name']}"))
        if con_mic:
            try:
                self.activar_mic(True)
            except Exception as e:          # Sin micrófono se sigue con el audio del PC
                self.cola_ui.put(("error_mic", str(e)))

    # ---------- Transcripción ----------
    def _motivo_descarte(self, s, n_segs, voz_seg, prompt=None):
        """Devuelve por qué se descarta un segmento de Whisper, o None si se conserva."""
        if s.compression_ratio > MAX_COMPRESION:
            return "compresion"
        if s.no_speech_prob > MAX_NO_VOZ:
            return "no_voz"
        if s.avg_logprob < MIN_LOGPROB:
            return "logprob"
        texto = _normalizar(s.text)
        if prompt:
            prompt = _normalizar(prompt)
            # Whisper repitió buena parte del prompt en vez de transcribir. Exigir al menos
            # la mitad evita descartar habla real corta como "el order block"
            if len(texto.split()) >= max(4, len(prompt.split()) // 2) and texto in prompt:
                return "eco_prompt"
        if CREDITOS_INVENTADOS.match(texto):
            return "credito"
        if texto in FRASES_SOSPECHOSAS and (
                s.avg_logprob < SOSP_LOGPROB or s.no_speech_prob > SOSP_NO_VOZ
                or (n_segs == 1 and voz_seg < SOSP_VOZ_MIN)):
            return "sospechosa"
        return None

    def _transcribir(self, fuente, trozos, final, voz_seg, palabras=False):
        """Devuelve (texto, segmentos conservados). Con palabras=True incluye sus tiempos."""
        t0 = time.perf_counter()
        idioma, prompt = self.contexto
        audio = np.concatenate(trozos)
        audio = resample_poly(audio, fuente.up, fuente.down).astype(np.float32)
        self.transcribiendo = True          # Para el indicador de estado de la ventana
        try:
            segmentos, _ = self.modelo.transcribe(
                audio,
                language=idioma,
                beam_size=5 if final else 1,
                vad_filter=False,           # El VAD propio ya decidió que hay voz
                condition_on_previous_text=False,
                # word_timestamps junto con without_timestamps hace que Whisper siga
                # transcribiendo la cola de silencio (alucina "Gracias..."): no combinarlos
                without_timestamps=not palabras,
                word_timestamps=palabras,
                initial_prompt=prompt,
            )
            segmentos = list(segmentos)     # La transcripción ocurre al recorrer el generador
        finally:
            self.transcribiendo = False
        conservados, descartes = [], Counter()
        for s in segmentos:
            motivo = self._motivo_descarte(s, len(segmentos), voz_seg, prompt)
            if motivo:
                descartes[motivo] += 1
            else:
                conservados.append(s)
        if final:
            fuente.stats.update(descartes)
        # Solo métricas, nunca el texto (puede ser información de clientes)
        seg_audio = len(audio) / SR_MODELO
        dt = time.perf_counter() - t0
        log.info("%-7s [%s] %s audio %5.1fs voz %4.1fs -> %.2fs (RTF %.2f) cola=%d segs %d/%d%s",
                 "final" if final else "parcial", fuente.nombre, idioma, seg_audio, voz_seg, dt,
                 dt / max(seg_audio, 1e-6), self.cola_audio.qsize(), len(conservados), len(segmentos),
                 "".join(f" {k}={v}" for k, v in descartes.items()))
        return " ".join(s.text.strip() for s in conservados).strip(), conservados

    def _recortar(self, voz):
        """Índices (ini, fin) de los trozos sin el silencio inicial y final, con RECORTE de margen."""
        con_voz = [i for i, v in enumerate(voz) if v > 0]
        margen = max(1, round(RECORTE / TROZO))
        return max(0, con_voz[0] - margen), con_voz[-1] + 1 + margen

    def _partir_en_palabra(self, fuente, texto, segmentos, trozos, voz, inicios):
        """Fija el texto hasta la penúltima palabra y devuelve el audio desde la última
        (que pudo quedar cortada) para la frase siguiente: (texto, (trozos, voz, inicios) del resto)."""
        sin_resto = (texto, ([], [], []))
        palabras = [w for s in segmentos for w in (s.words or [])]
        if len(palabras) < 2:
            return sin_resto
        muestra = int(max(0.0, palabras[-1].start - 0.1) * fuente.tasa)
        acum = 0
        for j, x in enumerate(trozos):
            if acum + len(x) > muestra:
                break
            acum += len(x)
        else:
            return sin_resto
        desde = muestra - acum
        resto_t = [trozos[j][desde:]] + trozos[j + 1:]
        resto_v = [voz[j] * (len(trozos[j]) - desde) / len(trozos[j])] + voz[j + 1:]
        resto_i = [inicios[j] + desde / fuente.tasa] + inicios[j + 1:]
        arrastre = sum(len(x) for x in resto_t) / fuente.tasa
        if arrastre > MAX_ARRASTRE:         # Tiempos poco fiables: mejor no arrastrar
            return sin_resto
        log.info("corte en palabra [%s]: %.2fs de audio pasan a la frase siguiente", fuente.nombre, arrastre)
        return "".join(w.word for w in palabras[:-1]).strip(), (resto_t, resto_v, resto_i)

    def _cerrar_frase(self, fuente, motivo):
        """Transcribe y publica la frase en curso de la fuente. Si el motivo es 'maximo',
        el audio desde la última palabra pasa a la frase siguiente."""
        trozos, voz, inicios = fuente.trozos, fuente.voz, fuente.inicios
        fuente.stats["cierre_" + motivo] += 1
        resto = ([], [], [])
        try:
            voz_seg = sum(voz)
            if voz_seg < VOZ_MIN:
                fuente.stats["omitida_vad"] += 1
                log.info("omitida [%s] audio %5.1fs voz %4.2fs (< VOZ_MIN)",
                         fuente.nombre, len(trozos) * TROZO, voz_seg)
                self.cola_ui.put(("parcial", (fuente.nombre, "")))
                return
            cortar = motivo == "maximo"
            ini, fin = self._recortar(voz)
            if cortar:
                fin = len(trozos)           # La voz sigue: no recortar el final
            texto, segmentos = self._transcribir(fuente, trozos[ini:fin], True, voz_seg, palabras=cortar)
            if cortar:
                texto, resto = self._partir_en_palabra(
                    fuente, texto, segmentos, trozos[ini:], voz[ini:], inicios[ini:])
            if fuente.previo:
                texto = _quitar_solape(fuente.previo, texto)
            fuente.previo = texto if resto[0] else ""
            self.cola_ui.put(("parcial", (fuente.nombre, "")))
            fuente.stats["frases"] += 1
            if texto:
                self.cola_ui.put(("final", (fuente.nombre, _corregir(texto), fuente.inicio_voz())))
        finally:
            fuente.reiniciar_frase(*resto)

    def _parcial(self, fuente):
        fuente.ultimo_parcial = time.time()
        voz_seg = sum(fuente.voz)
        if voz_seg >= VOZ_MIN:
            ini, fin = self._recortar(fuente.voz)
            texto = self._transcribir(fuente, fuente.trozos[ini:fin], False, voz_seg)[0]
            self.cola_ui.put(("parcial", (fuente.nombre, _corregir(texto))))
        fuente.ultimo_parcial = time.time()

    def _bucle(self, stop, cola_audio, fuentes):
        try:
            self._recargar_si_hace_falta()
        except Exception as e:
            log.error("Error al recargar el modelo: %s", e)
            stop.set()                      # Sin modelo no hay nada que hacer: avisa "detenido"
        while not stop.is_set():
            # Recoger todo el audio disponible, repartido a su fuente
            vaciada = False
            try:
                fuente, x, inicio = cola_audio.get(timeout=0.1)
                while True:
                    fuente.agregar(x, inicio)
                    if fuente.motivo_cierre(incluir_sin_audio=False):
                        break               # Cerrar antes de mezclar con la frase siguiente
                    fuente, x, inicio = cola_audio.get_nowait()
            except queue.Empty:
                vaciada = True

            try:
                # Fuentes apagadas (micrófono): con la cola vacía ya no queda audio suyo
                retiradas = [f for f in fuentes if f.retirar] if vaciada else []
                for f in retiradas:
                    try:
                        if f.trozos:
                            self._cerrar_frase(f, "detener")
                    finally:
                        fuentes.remove(f)
                        f.resumen()
                        self.cola_ui.put(("parcial", (f.nombre, "")))

                # Primero los finales (lo que se guarda), luego un parcial a la vez:
                # el de la fuente que lleva más tiempo sin refrescarse
                pendientes = [(f, f.motivo_cierre()) for f in fuentes]
                pendientes = [(f, m) for f, m in pendientes if m]
                for f, motivo in pendientes:
                    self._cerrar_frase(f, motivo)
                if pendientes:
                    continue
                ahora = time.time()
                candidatas = [f for f in fuentes if f.trozos and ahora - f.ultimo_parcial >= INTERVALO_PARCIAL]
                if candidatas:
                    self._parcial(min(candidatas, key=lambda f: f.ultimo_parcial))
            except Exception as e:
                self.cola_ui.put(("estado", f"Error: {e}"))

        # Audio que quedó en la cola al detener (los streams ya están cerrados)
        try:
            while True:
                try:
                    fuente, x, inicio = cola_audio.get_nowait()
                except queue.Empty:
                    break
                fuente.agregar(x, inicio)
            for f in fuentes:
                if f.trozos:
                    self._cerrar_frase(f, "detener")
        except Exception as e:
            self.cola_ui.put(("estado", f"Error: {e}"))
        finally:
            for f in fuentes:
                f.resumen()
            self.cola_ui.put(("detenido", None))


class Traductor(threading.Thread):
    """Traduce frases finales inglés -> español en su propio hilo, en CPU, sin frenar la
    transcripción. pedir(clave, texto) encola; el resultado llega a la ventana como
    ("traduccion", (clave, texto_es)). Si el modelo no está: ("error_traduccion", mensaje)."""

    def __init__(self, cola_ui, ruta=None):
        super().__init__(daemon=True)
        self.cola_ui = cola_ui
        self.ruta = ruta or MODELO_TRADUCCION
        self.pendientes = queue.Queue()
        self.listo = threading.Event()      # Se marca al terminar de cargar, haya funcionado o no
        self.error = None
        self.terminos = tuple(VOCABULARIO) + tuple(NO_TRADUCIR)

    def run(self):
        try:
            import ctranslate2
            import sentencepiece as spm
            t0 = time.perf_counter()
            self.modelo = ctranslate2.Translator(self.ruta, device="cpu", compute_type="int8",
                                                 inter_threads=1, intra_threads=TRADUCCION_HILOS)
            self.sp_origen = spm.SentencePieceProcessor(model_file=os.path.join(self.ruta, "source.spm"))
            self.sp_destino = spm.SentencePieceProcessor(model_file=os.path.join(self.ruta, "target.spm"))
            log.info("traductor cargado en %.1f s", time.perf_counter() - t0)
        except Exception as e:
            self.error = (f"No se pudo cargar el modelo de traducción ({e}).\n\n"
                          f"Se convierte una vez siguiendo tools\\convertir_traduccion.md y queda en:\n{self.ruta}")
            log.error("Traductor: %s", e)
            self.cola_ui.put(("error_traduccion", self.error))
            self.listo.set()
            return
        self.listo.set()
        while True:
            clave, texto = self.pendientes.get()
            if clave is None:
                break
            try:
                t0 = time.perf_counter()
                traduccion = self.traducir(texto)
                # Solo métricas, nunca el texto (puede ser información de clientes)
                log.info("traducción %4.0f ms, %d palabras", (time.perf_counter() - t0) * 1000, len(texto.split()))
                self.cola_ui.put(("traduccion", (clave, traduccion)))
            except Exception as e:
                log.error("Error al traducir: %s", e)

    def _traducir_lote(self, textos):
        piezas = [self.sp_origen.encode(t, out_type=str) + ["</s>"] for t in textos]
        return [self.sp_destino.decode(r.hypotheses[0]) for r in self.modelo.translate_batch(piezas, beam_size=4)]

    def traducir(self, texto):
        """Oración por oración (en un solo lote) y con los términos protegidos; si el modelo
        deformó algún marcador, esa oración se traduce sin protegerlos."""
        oraciones = _oraciones(texto) or [texto]
        marcadas = [_proteger_terminos(o, self.terminos) for o in oraciones]
        traducidas = self._traducir_lote([m for m, _ in marcadas])
        resultado = []
        for original, (_, usados), traducida in zip(oraciones, marcadas, traducidas):
            restaurada = _restaurar_terminos(traducida, usados) if usados else traducida
            resultado.append(restaurada if restaurada is not None else self._traducir_lote([original])[0])
        return " ".join(resultado)

    def pedir(self, clave, texto):
        self.pendientes.put((clave, texto))

    def detener(self):
        self.pendientes.put((None, None))


class Frase:
    """Una frase final, tal como la muestran la ventana y la franja y la guarda el .txt.
    La traducción llega después (otro hilo), por eso es mutable."""

    __slots__ = ("inicio", "hora", "fuente", "texto", "traduccion", "marca")

    def __init__(self, inicio, fuente, texto):
        self.inicio, self.fuente, self.texto = inicio, fuente, texto
        # Hora en que empezó a hablarse, no en que terminó de transcribirse
        self.hora = datetime.datetime.fromtimestamp(inicio).strftime("%H:%M:%S")
        self.traduccion = None
        self.marca = None                   # Marca de Tk donde empieza en la ventana

    def lineas_txt(self):
        lineas = [f"[{self.hora}] {self.fuente}: {self.texto}"]
        if self.traduccion:
            lineas.append(f"[{self.hora}] {self.fuente} (es): {self.traduccion}")
        return lineas


# Vistas de la traducción en el selector de la ventana -> valor guardado en la configuración
VISTAS_TRADUCCION = {"Sin traducir": "no", "Original + traducción": "ambos", "Solo traducción": "solo"}


class Configuracion(dict):
    """Preferencias de la ventana guardadas entre sesiones en CONFIG_ARCHIVO.
    Un archivo dañado o con valores de otro tipo no impide abrir: se usan los valores iniciales."""

    INICIAL = {"idioma": IDIOMA, "siempre_encima": True, "geometria": "760x440",
               "tamano_letra": 14, "color_yo": "#7fd18b", "color_otros": "#e8e8e8",
               # Franja de subtítulos
               "franja_tamano": 22, "franja_lineas": 2, "franja_opacidad": 0.9, "franja_posicion": "abajo",
               # Atajo de todo el sistema para entrar y salir de los subtítulos ("" = sin atajo global)
               "atajo_subtitulos": "Ctrl+Shift+S",
               # Traducción inglés -> español: "no", "ambos" (original + traducción) o "solo"
               "traduccion": "no",
               # Qué audio del PC escuchar: "todo", "solo" una app o "menos" una app (escuchar_app)
               "escuchar_modo": "todo", "escuchar_app": ""}
    LETRA_MIN, LETRA_MAX = 9, 36
    FRANJA_LINEAS_MAX = 4

    def __init__(self, ruta=None):
        super().__init__(self.INICIAL)
        self.ruta = ruta or CONFIG_ARCHIVO
        try:
            with open(self.ruta, encoding="utf-8") as f:
                guardada = json.load(f)
            for clave, valor in guardada.items():
                inicial = self.INICIAL.get(clave)
                # Solo claves conocidas y del mismo tipo (un int sirve donde se espera float)
                if clave in self.INICIAL and (type(valor) is type(inicial)
                                              or (type(inicial) is float and type(valor) is int)):
                    self[clave] = valor
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError) as e:
            log.warning("Configuración ilegible (%s): se usan los valores iniciales", e)

    def guardar(self):
        try:
            os.makedirs(os.path.dirname(self.ruta), exist_ok=True)
            tmp = self.ruta + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.ruta)
        except OSError as e:
            log.error("No se pudo guardar la configuración: %s", e)


def _geometria_visible(geometria):
    """La geometría guardada solo si su esquina queda dentro de las pantallas actuales
    (un monitor desconectado dejaría la ventana fuera de la vista)."""
    m = re.fullmatch(r"(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)", geometria)
    if not m:
        return geometria if re.fullmatch(r"\d+x\d+", geometria) else Configuracion.INICIAL["geometria"]
    x, y = int(m.group(3)), int(m.group(4))
    try:
        import ctypes
        u = ctypes.windll.user32       # Escritorio virtual: todos los monitores juntos
        x0, y0, ancho, alto = (u.GetSystemMetrics(i) for i in (76, 77, 78, 79))
    except (AttributeError, OSError):
        return geometria
    dentro = x0 <= x < x0 + ancho - 100 and y0 <= y < y0 + alto - 50
    return geometria if dentro else f"{m.group(1)}x{m.group(2)}"


def _parsear_atajo(texto):
    """'Ctrl+Shift+S' -> (modificadores, tecla virtual) para RegisterHotKey, o None si no se entiende."""
    MODIFICADORES = {"ctrl": 0x2, "control": 0x2, "alt": 0x1, "shift": 0x4, "mayus": 0x4, "win": 0x8}
    partes = [p.strip().lower() for p in texto.split("+") if p.strip()]
    if len(partes) < 2 or any(p not in MODIFICADORES for p in partes[:-1]):
        return None
    mods = 0
    for p in partes[:-1]:
        mods |= MODIFICADORES[p]
    tecla = partes[-1]
    if len(tecla) == 1 and tecla.isalnum() and tecla.isascii():
        return mods, ord(tecla.upper())
    if re.fullmatch(r"f([1-9]|1\d|2[0-4])", tecla):
        return mods, 0x70 + int(tecla[1:]) - 1
    return None


class AtajoGlobal(threading.Thread):
    """Atajo de todo el sistema: funciona aunque la ventana no tenga el foco (p. ej. desde Teams).
    Windows entrega WM_HOTKEY a la cola de mensajes del hilo que registró el atajo, por eso
    vive en su propio hilo y avisa a la ventana por cola_ui con ("atajo", nombre)."""

    WM_HOTKEY, WM_QUIT, MOD_NOREPEAT = 0x0312, 0x0012, 0x4000

    def __init__(self, combinacion, nombre, cola_ui):
        super().__init__(daemon=True)
        self.combinacion, self.nombre, self.cola_ui = combinacion, nombre, cola_ui
        self.id_hilo = None
        self.registrado = threading.Event()  # Se marca también si falló, para no quedar esperando
        self.ok = False

    def run(self):
        import ctypes
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self.id_hilo = kernel32.GetCurrentThreadId()
        tecla = _parsear_atajo(self.combinacion)
        self.ok = bool(tecla) and bool(user32.RegisterHotKey(None, 1, tecla[0] | self.MOD_NOREPEAT, tecla[1]))
        self.registrado.set()
        if not self.ok:
            log.warning("No se pudo registrar el atajo global %s (¿lo usa otra app?)", self.combinacion)
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == self.WM_HOTKEY:
                self.cola_ui.put(("atajo", self.nombre))
        user32.UnregisterHotKey(None, 1)

    def detener(self):
        if self.id_hilo is not None:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self.id_hilo, self.WM_QUIT, 0, 0)


def _area_trabajo():
    """(x0, y0, x1, y1) de la pantalla principal sin la barra de tareas."""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0)   # SPI_GETWORKAREA
        return r.left, r.top, r.right, r.bottom
    except (AttributeError, OSError):
        return None


def _no_capturar_clics(hwnd):
    """Estilos de Windows para la franja: los clics pasan a lo que está debajo, no aparece
    en la barra de tareas ni en Alt+Tab y no roba el foco. Devuelve si quedaron aplicados."""
    import ctypes
    GWL_EXSTYLE = -20
    WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x20, 0x80, 0x8000000
    u = ctypes.windll.user32
    estilos = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    u.SetWindowLongW(hwnd, GWL_EXSTYLE, u.GetWindowLongW(hwnd, GWL_EXSTYLE) | estilos)
    return u.GetWindowLongW(hwnd, GWL_EXSTYLE) & estilos == estilos


class App:
    FONDO, TEXTO, GRIS, ACENTO = "#1e1f22", "#e8e8e8", "#8a8d93", "#6ea8fe"
    VERDE, AMARILLO, ROJO = "#5cc98a", "#e5c07b", "#e06c75"
    ANCHO_MEDIDOR = 160                 # Píxeles de cada barra de nivel
    ATRASO_AVISO = 1.0                  # Segundos de atraso de la cola que ponen el estado en amarillo
    NIVEL_VIGENTE = 0.3                 # Un nivel más viejo que esto cuenta como silencio

    def __init__(self, root):
        self.root = root
        self.config = Configuracion()
        if self.config["idioma"] not in IDIOMAS.values():
            self.config["idioma"] = IDIOMA
        if self.config["franja_posicion"] not in ("abajo", "arriba"):
            self.config["franja_posicion"] = "abajo"
        self.config["franja_lineas"] = max(1, min(Configuracion.FRANJA_LINEAS_MAX, self.config["franja_lineas"]))
        self.config["franja_opacidad"] = max(0.3, min(1.0, self.config["franja_opacidad"]))
        self.config["franja_tamano"] = max(12, min(48, self.config["franja_tamano"]))
        if self.config["traduccion"] not in VISTAS_TRADUCCION.values():
            self.config["traduccion"] = "no"
        if self.config["escuchar_modo"] not in ("todo", "solo", "menos") or \
                (self.config["escuchar_modo"] != "todo" and not self.config["escuchar_app"]):
            self.config["escuchar_modo"] = "todo"
        self.cola_ui = queue.Queue()
        self.motor = Motor(self.cola_ui)
        self.motor.cambiar_idioma(self.config["idioma"])
        # Frases finales (inicio, hora, fuente, texto) ordenadas por inicio del habla:
        # las de la ventana (Limpiar las borra) y las de la sesión (van al .txt automático)
        self.finales, self.registro = [], []    # Listas de Frase
        self.traductor = None               # Se carga al activar la traducción
        self.por_traducir = {}              # Clave del pedido -> Frase que espera su traducción
        self.n_marcas = self.n_pedidos = 0
        self.parciales = {}                 # Texto provisional por fuente
        self.activo = False
        self.cerrando = False
        self.destruida = False
        os.makedirs(CARPETA_SALIDA, exist_ok=True)
        onedrive = os.environ.get("OneDrive")
        if onedrive and os.path.abspath(CARPETA_SALIDA).lower().startswith(os.path.abspath(onedrive).lower()):
            log.warning("CARPETA_SALIDA está dentro de OneDrive: las transcripciones se sincronizarán a la nube")
        nombre = datetime.datetime.now().strftime("transcripcion_%Y%m%d_%H%M.txt")
        self.archivo_auto = os.path.join(CARPETA_SALIDA, nombre)

        root.title("Transcriptor en vivo")
        root.geometry(_geometria_visible(self.config["geometria"]))
        root.configure(bg=self.FONDO)
        root.attributes("-topmost", self.config["siempre_encima"])
        self.estilo = dict(bg="#2b2d31", fg=self.TEXTO, activebackground="#3a3d43",
                           activeforeground=self.TEXTO, relief="flat", padx=10, pady=4)
        estilo = self.estilo

        # Arriba: lo que controla la captura
        barra = tk.Frame(root, bg=self.FONDO)
        barra.pack(fill="x", padx=8, pady=(6, 2))
        self.btn = tk.Button(barra, text="▶ Iniciar", command=self.alternar, state="disabled", **estilo)
        self.btn.pack(side="left")
        self.mic = tk.BooleanVar(value=False)   # Apagado al iniciar: evita transcribirme por accidente
        tk.Checkbutton(barra, text="🎤 Micrófono", variable=self.mic, command=self._alternar_mic,
                       indicatoron=False, selectcolor="#2f6f3e", offrelief="flat",
                       **estilo).pack(side="left", padx=4)
        # Tras conectar audífonos u otro dispositivo: toma los predeterminados actuales de Windows
        self.btn_reconectar = tk.Button(barra, text="🔄 Reconectar", command=self.reconectar,
                                        state="disabled", **estilo)
        self.btn_reconectar.pack(side="left")
        # Qué audio del PC escuchar: todo, solo una app (p. ej. Discord) o todo menos una (p. ej. Chrome)
        self.btn_escuchar = tk.Menubutton(barra, text=self._texto_escucha(), **estilo)
        self.menu_escuchar = tk.Menu(self.btn_escuchar, tearoff=False, bg="#2b2d31", fg=self.TEXTO,
                                     activebackground="#3a3d43", postcommand=self._llenar_menu_escuchar)
        self.btn_escuchar["menu"] = self.menu_escuchar
        self.btn_escuchar.pack(side="left", padx=4)
        # Idioma que se habla en el audio (no traduce): aplica desde la próxima frase
        self.idioma = tk.StringVar(value=next(n for n, c in IDIOMAS.items() if c == self.config["idioma"]))
        menu = tk.OptionMenu(barra, self.idioma, *IDIOMAS, command=self._cambiar_idioma)
        menu.configure(highlightthickness=0, **estilo)
        menu["menu"].configure(bg="#2b2d31", fg=self.TEXTO, activebackground="#3a3d43")
        menu.pack(side="left", padx=4)
        # Traducción al español de las frases finales (cuando el idioma del audio es English)
        self.vista = tk.StringVar(value=next(n for n, v in VISTAS_TRADUCCION.items()
                                             if v == self.config["traduccion"]))
        menu = tk.OptionMenu(barra, self.vista, *VISTAS_TRADUCCION, command=self.cambiar_vista_traduccion)
        menu.configure(highlightthickness=0, **estilo)
        menu["menu"].configure(bg="#2b2d31", fg=self.TEXTO, activebackground="#3a3d43")
        menu.pack(side="left")
        self.subtitulos = tk.BooleanVar(value=False)
        tk.Checkbutton(barra, text="CC Subtítulos", variable=self.subtitulos,
                       command=lambda: self.alternar_subtitulos(self.subtitulos.get()),
                       indicatoron=False, selectcolor="#2f4f7f", offrelief="flat", **estilo).pack(side="left")
        tk.Button(barra, text="⚙", command=self.abrir_ajustes, **estilo).pack(side="left", padx=4)

        # Medidores de nivel por fuente (confirman que entra audio) e indicador de estado
        fila = tk.Frame(root, bg=self.FONDO)
        fila.pack(fill="x", padx=12, pady=(2, 0))
        self.medidores = tk.Canvas(fila, height=22, width=2 * (60 + self.ANCHO_MEDIDOR) + 20,
                                   bg=self.FONDO, highlightthickness=0)
        self.medidores.pack(side="left")
        self.barras = {}                    # fuente -> [id de la barra, fracción mostrada]
        for i, nombre in enumerate((NOMBRE_OTROS, NOMBRE_YO)):
            x = i * (60 + self.ANCHO_MEDIDOR + 20)
            self.medidores.create_text(x, 11, text=nombre, anchor="w", fill=self.GRIS)
            self.medidores.create_rectangle(x + 52, 5, x + 52 + self.ANCHO_MEDIDOR, 17, outline="#3a3d43")
            self.barras[nombre] = [self.medidores.create_rectangle(x + 53, 6, x + 53, 17, width=0), 0.0]
        self.texto_estado = tk.Label(fila, text="Detenido", bg=self.FONDO, fg=self.GRIS)
        self.texto_estado.pack(side="right")
        self.punto_estado = tk.Label(fila, text="●", bg=self.FONDO, fg=self.GRIS, font=("Segoe UI", 13))
        self.punto_estado.pack(side="right")
        self.hay_error = False

        # Abajo: lo que afecta al texto (se empaqueta antes que el texto para que no lo tape)
        pie = tk.Frame(root, bg=self.FONDO)
        pie.pack(side="bottom", fill="x", padx=8, pady=(2, 6))
        tk.Button(pie, text="A−", command=lambda: self.cambiar_letra(-1), **estilo).pack(side="left")
        tk.Button(pie, text="A+", command=lambda: self.cambiar_letra(+1), **estilo).pack(side="left", padx=(4, 12))
        tk.Button(pie, text="Limpiar", command=self.limpiar, **estilo).pack(side="left")
        tk.Button(pie, text="Guardar...", command=self.guardar, **estilo).pack(side="left", padx=4)
        self.encima = tk.BooleanVar(value=self.config["siempre_encima"])
        tk.Checkbutton(pie, text="Siempre encima", variable=self.encima, command=self._encima,
                       bg=self.FONDO, fg=self.TEXTO, selectcolor="#2b2d31",
                       activebackground=self.FONDO, activeforeground=self.TEXTO).pack(side="right")
        # Mensajes (qué se escucha, errores...): en el pie, donde hay espacio (los atajos, en Ajustes)
        self.estado = tk.Label(pie, text="", bg=self.FONDO, fg=self.GRIS, anchor="e")
        self.estado.pack(side="right", fill="x", expand=True, padx=12)

        # Fuentes con nombre: cambiar su tamaño actualiza todo el texto que las usa
        self.letra = tkfont.Font(family="Segoe UI", size=self.config["tamano_letra"])
        self.letra_hora = tkfont.Font(family="Segoe UI", size=max(8, self.config["tamano_letra"] - 4))
        self.texto = tk.Text(root, wrap="word", font=self.letra, bg=self.FONDO, fg=self.TEXTO,
                             insertbackground=self.TEXTO, relief="flat", padx=12, pady=8)
        self.texto.pack(fill="both", expand=True)
        self.texto.tag_config("parcial", foreground=self.GRIS)
        self.texto.tag_config("hora", foreground=self.ACENTO, font=self.letra_hora)
        self.letra_traduccion = tkfont.Font(family="Segoe UI", size=self.config["tamano_letra"], slant="italic")
        self.texto.tag_config("traduccion", font=self.letra_traduccion)
        self._sangria_traduccion()
        self._aplicar_colores()
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.mark_gravity("parcial_ini", "left")
        self.texto.configure(state="disabled")
        self.ajustes = None                 # Ventana de ajustes, si está abierta
        self.franja = None                  # Franja de subtítulos, si está visible
        self._enlazar_atajos()
        self.atajo_global = None
        if self.config["atajo_subtitulos"]:
            self.atajo_global = AtajoGlobal(self.config["atajo_subtitulos"], "subtitulos", self.cola_ui)
            self.atajo_global.start()
            # Soltar el atajo al destruirse la ventana, se cierre como se cierre
            root.bind("<Destroy>", lambda e: e.widget is root and self.atajo_global.detener(), add="+")

        root.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.timer_vram = None              # Liberar la VRAM tras LIBERAR_VRAM_TRAS detenido
        threading.Thread(target=self.motor.cargar_modelo, daemon=True).start()
        root.after(100, self.revisar_cola)

    # ---------- Controles ----------
    def alternar(self):
        if self.activo:
            # El botón se reactiva cuando el hilo termina la última frase ("detenido")
            self.btn.configure(state="disabled")
            self.btn_reconectar.configure(state="disabled")
            self.activo = False
            self.motor.detener()
        else:
            self._cancelar_timer_vram()
            self.hay_error = False
            try:
                self.motor.iniciar(con_mic=self.mic.get(), escuchar=self._escuchar())
            except Exception as e:
                messagebox.showerror("Error de audio", str(e))
                self._programar_liberar_vram()
                return
            self.btn.configure(text="■ Detener")
            self.btn_reconectar.configure(state="normal")
            self.activo = True

    def _programar_liberar_vram(self):
        self._cancelar_timer_vram()
        if LIBERAR_VRAM_TRAS is not None:
            self.timer_vram = self.root.after(int(LIBERAR_VRAM_TRAS * 1000), self._liberar_vram)

    def _cancelar_timer_vram(self):
        if self.timer_vram is not None:
            self.root.after_cancel(self.timer_vram)
            self.timer_vram = None

    def _liberar_vram(self):
        self.timer_vram = None
        if not self.activo and self.motor.liberar_vram():
            self.estado.configure(text="Modelo fuera de la GPU: se recarga al iniciar (~5 s)")

    def reconectar(self):
        if not self.activo:
            return
        try:
            self.motor.reconectar(con_mic=self.mic.get(), escuchar=self._escuchar())
        except Exception as e:
            messagebox.showerror("Error de audio",
                                 f"{e}\n\nRevisa el dispositivo de salida en Windows y pulsa Reconectar otra vez.")

    # ---------- Qué audio escuchar ----------
    def _escuchar(self):
        """None = todo el sistema; si no, ("solo" | "menos", "Discord.exe")."""
        modo = self.config["escuchar_modo"]
        return None if modo == "todo" else (modo, self.config["escuchar_app"])

    def _texto_escucha(self):
        modo, app = self.config["escuchar_modo"], _nombre_app(self.config["escuchar_app"])
        return {"todo": "🔊 Todo", "solo": f"🔊 Solo {app}", "menos": f"🔊 Todo menos {app}"}[modo] + " ▾"

    def _llenar_menu_escuchar(self, apps=None):
        """Se rellena al abrir el menú: las apps que están sonando en ese momento."""
        m = self.menu_escuchar
        m.delete(0, "end")
        actual = self.config["escuchar_modo"], self.config["escuchar_app"]
        marca = lambda modo, app="": ("✓ " if actual == (modo, app) or (modo == "todo" == actual[0]) else "   ")
        m.add_command(label=marca("todo") + "Todo el sistema", command=lambda: self._cambiar_escucha("todo", ""))
        if apps is None:
            try:
                from captura_app import apps_con_audio
                apps = apps_con_audio()
            except Exception as e:
                log.error("No se pudo listar las apps con audio: %s", e)
                apps = []
        if actual[1] and actual[1] not in apps:     # La elegida aunque ahora no suene
            apps = sorted(apps + [actual[1]], key=str.lower)
        m.add_separator()
        if not apps:
            m.add_command(label="   (ninguna app está sonando)", state="disabled")
        for app in apps:
            m.add_command(label=marca("solo", app) + f"Solo {_nombre_app(app)}",
                          command=lambda a=app: self._cambiar_escucha("solo", a))
        if apps:
            m.add_separator()
        for app in apps:
            m.add_command(label=marca("menos", app) + f"Todo menos {_nombre_app(app)}",
                          command=lambda a=app: self._cambiar_escucha("menos", a))

    def _cambiar_escucha(self, modo, app):
        """Se guarda; con la captura en marcha se aplica al tiro (como Reconectar)."""
        anterior = self.config["escuchar_modo"], self.config["escuchar_app"]
        self.config["escuchar_modo"], self.config["escuchar_app"] = modo, app
        if self.activo:
            try:
                self.motor.reconectar(con_mic=self.mic.get(), escuchar=self._escuchar())
            except Exception as e:
                self.config["escuchar_modo"], self.config["escuchar_app"] = anterior
                messagebox.showerror("Escuchar", f"{e}\n\nSe sigue escuchando como antes.")
                try:
                    self.motor.reconectar(con_mic=self.mic.get(), escuchar=self._escuchar())
                except Exception as e2:
                    log.error("No se pudo volver a la captura anterior: %s", e2)
        self.config.guardar()
        self.btn_escuchar.configure(text=self._texto_escucha())

    def _cambiar_idioma(self, nombre):
        self.motor.cambiar_idioma(IDIOMAS[nombre])
        self.config["idioma"] = IDIOMAS[nombre]
        self.config.guardar()
        self.estado.configure(text=f"Idioma: {nombre}")

    def _alternar_mic(self):
        """Con la captura detenida solo se recuerda la elección; en marcha se aplica al tiro."""
        if not self.activo:
            return
        try:
            self.motor.activar_mic(self.mic.get())
        except Exception as e:
            self.mic.set(False)
            messagebox.showerror("Error de micrófono", str(e))

    def _encima(self):
        self.root.attributes("-topmost", self.encima.get())
        self.config["siempre_encima"] = self.encima.get()
        self.config.guardar()

    # ---------- Atajos de teclado (con la ventana enfocada) ----------
    ATAJOS = {
        "iniciar_detener": ("<Control-space>",),
        "microfono": ("<Control-m>", "<Control-M>"),
        "letra_mas": ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"),
        "letra_menos": ("<Control-minus>", "<Control-KP_Subtract>"),
        "subtitulos": ("<Control-S>",),     # Respaldo local; el global (Ctrl+Shift+S) lo atrapa antes
    }

    def _enlazar_atajos(self):
        acciones = {"iniciar_detener": self.atajo_iniciar_detener, "microfono": self.atajo_microfono,
                    "letra_mas": lambda: self.cambiar_letra(+1), "letra_menos": lambda: self.cambiar_letra(-1),
                    "subtitulos": self.alternar_subtitulos}
        for nombre, secuencias in self.ATAJOS.items():
            for secuencia in secuencias:
                # "break": que el atajo no llegue además al widget enfocado
                self.root.bind_all(secuencia, lambda e, a=acciones[nombre]: (a(), "break")[1])

    def atajo_iniciar_detener(self):
        """Igual que el botón: no hace nada mientras está deshabilitado (cargando o deteniendo)."""
        if str(self.btn["state"]) == "normal":
            self.alternar()

    def atajo_microfono(self):
        self.mic.set(not self.mic.get())
        self._alternar_mic()

    # ---------- Modo subtítulos ----------
    def alternar_subtitulos(self, activar=None):
        """Alterna entre la vista normal y la franja de subtítulos (la ventana se minimiza)."""
        activar = (self.franja is None) if activar is None else activar
        self.subtitulos.set(activar)
        if activar and self.franja is None:
            self._crear_franja()
            self.root.iconify()
        elif not activar and self.franja is not None:
            self.franja.destroy()
            self.franja = None
            self.root.deiconify()
            self.root.lift()

    def _crear_franja(self):
        f = self.franja = tk.Toplevel(self.root, bg="#000000")
        f.overrideredirect(True)            # Sin bordes ni barra de título
        f.attributes("-topmost", True)
        f.attributes("-alpha", self.config["franja_opacidad"])
        self.letra_franja = tkfont.Font(family="Segoe UI", size=self.config["franja_tamano"])
        self.texto_franja = tk.Text(f, bg="#000000", fg=self.TEXTO, font=self.letra_franja, wrap="word",
                                    relief="flat", borderwidth=0, highlightthickness=0, padx=16, pady=8)
        self.texto_franja.pack(fill="both", expand=True)
        self.texto_franja.tag_config("parcial", foreground=self.GRIS)
        self.letra_franja_traduccion = tkfont.Font(family="Segoe UI", size=self.config["franja_tamano"],
                                                   slant="italic")
        self.texto_franja.tag_config("traduccion", font=self.letra_franja_traduccion)
        for fuente, clave in ((NOMBRE_YO, "color_yo"), (NOMBRE_OTROS, "color_otros")):
            self.texto_franja.tag_config(f"hablante_{fuente}", foreground=self.config[clave])
        self._ubicar_franja()
        f.update_idletasks()
        try:
            self.franja_sin_clics = _no_capturar_clics(int(f.wm_frame(), 16))
        except (AttributeError, OSError, ValueError):
            self.franja_sin_clics = False
        self._actualizar_franja()

    def _ubicar_franja(self):
        """80 % del ancho de la pantalla principal, centrada, sobre la barra de tareas o arriba."""
        area = _area_trabajo() or (0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        x0, y0, x1, y1 = area
        ancho = int((x1 - x0) * 0.8)
        alto = self.config["franja_lineas"] * self.letra_franja.metrics("linespace") + 16
        x = x0 + (x1 - x0 - ancho) // 2
        y = y1 - alto - 24 if self.config["franja_posicion"] == "abajo" else y0 + 24
        self.franja.geometry(f"{ancho}x{alto}+{x}+{y}")
        self.texto_franja.configure(height=self.config["franja_lineas"])

    def _actualizar_franja(self):
        """Las últimas frases (finales y provisionales) que caben en la franja."""
        if self.franja is None:
            return
        vista, lineas = self.config["traduccion"], []
        for frase in self.finales:
            hablante = f"hablante_{frase.fuente}"
            if vista == "solo" and frase.traduccion:
                lineas.append((f"{frase.fuente}: {frase.traduccion}", hablante))
                continue
            lineas.append((f"{frase.fuente}: {frase.texto}", hablante))
            if vista == "ambos" and frase.traduccion:
                lineas.append((f"→ {frase.traduccion}", (hablante, "traduccion")))
        lineas += [(f"{fuente}: {t}", "parcial") for fuente, t in self.parciales.items() if t]
        t = self.texto_franja
        t.configure(state="normal")
        t.delete("1.0", "end")
        for i, (texto, etiqueta) in enumerate(lineas[-self.config["franja_lineas"]:]):
            t.insert("end", ("\n" if i else "") + texto, etiqueta)
        t.configure(state="disabled")
        t.see("end")                        # Si una frase larga ocupa varias líneas, se ve su final

    def _cambiar_franja(self, clave, valor):
        """Desde Ajustes: guarda y, si la franja está visible, la actualiza al instante."""
        self.config[clave] = valor
        self.config.guardar()
        if self.franja is not None:
            self.franja.attributes("-alpha", self.config["franja_opacidad"])
            self.letra_franja.configure(size=self.config["franja_tamano"])
            self.letra_franja_traduccion.configure(size=self.config["franja_tamano"])
            self._ubicar_franja()
            self._actualizar_franja()

    def _sangria_traduccion(self):
        """Si una traducción ocupa varias líneas, las siguientes quedan alineadas tras la flecha."""
        self.texto.tag_config("traduccion", lmargin2=self.letra_traduccion.measure("    → "))

    # ---------- Letra y colores ----------
    def cambiar_letra(self, delta=0, tamano=None):
        tamano = tamano if tamano is not None else self.config["tamano_letra"] + delta
        tamano = max(Configuracion.LETRA_MIN, min(Configuracion.LETRA_MAX, int(tamano)))
        self.letra.configure(size=tamano)
        self.letra_traduccion.configure(size=tamano)
        self._sangria_traduccion()
        self.letra_hora.configure(size=max(8, tamano - 4))
        self.config["tamano_letra"] = tamano
        self.config.guardar()
        self.texto.see("end")

    def _aplicar_colores(self):
        """Etiqueta de color por hablante; un color inválido en la configuración vuelve al inicial."""
        for fuente, clave in ((NOMBRE_YO, "color_yo"), (NOMBRE_OTROS, "color_otros")):
            try:
                self.root.winfo_rgb(self.config[clave])
            except tk.TclError:
                self.config[clave] = Configuracion.INICIAL[clave]
            self.texto.tag_config(f"hablante_{fuente}", foreground=self.config[clave])
            self.medidores.itemconfigure(self.barras[fuente][0], fill=self.config[clave])
            if getattr(self, "franja", None) is not None:
                self.texto_franja.tag_config(f"hablante_{fuente}", foreground=self.config[clave])

    def elegir_color(self, clave, boton=None):
        _, color = colorchooser.askcolor(self.config[clave], parent=self.ajustes or self.root,
                                         title="Color de " + (NOMBRE_YO if clave == "color_yo" else NOMBRE_OTROS))
        if color:
            self.config[clave] = color
            self.config.guardar()
            self._aplicar_colores()
            if boton is not None:
                boton.configure(bg=color, activebackground=color)

    def abrir_ajustes(self):
        """Ventana de ajustes: los cambios se aplican y se guardan al instante."""
        if self.ajustes is not None and self.ajustes.winfo_exists():
            self.ajustes.lift()
            return
        v = self.ajustes = tk.Toplevel(self.root, bg=self.FONDO, padx=16, pady=12)
        v.title("Ajustes")
        v.resizable(False, False)
        v.transient(self.root)
        etiqueta = dict(bg=self.FONDO, fg=self.TEXTO, anchor="w")
        tk.Label(v, text="Tamaño de letra", **etiqueta).grid(row=0, column=0, sticky="w", pady=4)
        tamano = tk.IntVar(value=self.config["tamano_letra"])
        tk.Spinbox(v, from_=Configuracion.LETRA_MIN, to=Configuracion.LETRA_MAX, width=5, textvariable=tamano,
                   command=lambda: self.cambiar_letra(tamano=tamano.get())).grid(row=0, column=1, sticky="w")
        for fila, (clave, nombre) in enumerate((("color_yo", NOMBRE_YO), ("color_otros", NOMBRE_OTROS)), start=1):
            tk.Label(v, text=f"Color de «{nombre}»", **etiqueta).grid(row=fila, column=0, sticky="w", pady=4)
            b = tk.Button(v, width=4, relief="flat", bg=self.config[clave], activebackground=self.config[clave])
            b.configure(command=lambda c=clave, b=b: self.elegir_color(c, b))
            b.grid(row=fila, column=1, sticky="w")
        # Franja de subtítulos
        tk.Label(v, text="Subtítulos", bg=self.FONDO, fg=self.ACENTO).grid(row=3, column=0, sticky="w", pady=(12, 2))
        franja_tamano = tk.IntVar(value=self.config["franja_tamano"])
        franja_lineas = tk.IntVar(value=self.config["franja_lineas"])
        tk.Label(v, text="Tamaño de letra", **etiqueta).grid(row=4, column=0, sticky="w", pady=4)
        tk.Spinbox(v, from_=12, to=48, width=5, textvariable=franja_tamano,
                   command=lambda: self._cambiar_franja("franja_tamano", franja_tamano.get())
                   ).grid(row=4, column=1, sticky="w")
        tk.Label(v, text="Líneas", **etiqueta).grid(row=5, column=0, sticky="w", pady=4)
        tk.Spinbox(v, from_=1, to=Configuracion.FRANJA_LINEAS_MAX, width=5, textvariable=franja_lineas,
                   command=lambda: self._cambiar_franja("franja_lineas", franja_lineas.get())
                   ).grid(row=5, column=1, sticky="w")
        tk.Label(v, text="Opacidad", **etiqueta).grid(row=6, column=0, sticky="w", pady=4)
        escala = tk.Scale(v, from_=30, to=100, orient="horizontal", length=120, bg=self.FONDO, fg=self.TEXTO,
                          highlightthickness=0, troughcolor="#2b2d31")
        escala.set(round(self.config["franja_opacidad"] * 100))
        # El comando va después de set(): si no, abrir Ajustes ya contaría como un cambio
        escala.configure(command=lambda x: self._cambiar_franja("franja_opacidad", int(float(x)) / 100))
        escala.grid(row=6, column=1, sticky="w")
        tk.Label(v, text="Posición", **etiqueta).grid(row=7, column=0, sticky="w", pady=4)
        posicion = tk.StringVar(value=self.config["franja_posicion"])
        m = tk.OptionMenu(v, posicion, "abajo", "arriba", command=lambda p: self._cambiar_franja("franja_posicion", p))
        m.configure(highlightthickness=0, **self.estilo)
        m.grid(row=7, column=1, sticky="w")
        atajo = self.config["atajo_subtitulos"] or "ninguno"
        estado_atajo = "" if self.atajo_global is None or self.atajo_global.ok else " (no disponible)"
        tk.Label(v, text=f"Atajos: Ctrl+Espacio iniciar/detener · Ctrl+M micrófono · Ctrl +/− letra\n"
                         f"Subtítulos, en todo el sistema: {atajo}{estado_atajo}\nSe cambia en {CONFIG_ARCHIVO}",
                 bg=self.FONDO, fg=self.GRIS, justify="left", wraplength=320).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tk.Button(v, text="Cerrar", command=v.destroy, **self.estilo).grid(row=9, column=0, columnspan=2, pady=(12, 0))

    def limpiar(self):
        for frase in self.finales:
            self.texto.mark_unset(frase.marca)
        self.finales.clear()
        self.parciales.clear()
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.configure(state="disabled")
        self._actualizar_franja()

    def guardar(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".txt", initialdir=CARPETA_SALIDA,
                                            filetypes=[("Texto", "*.txt")])
        if ruta:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write("\n".join(l for frase in self.finales for l in frase.lineas_txt()))

    def cerrar(self):
        if self.cerrando:
            return
        self.cerrando = True
        if self.root.state() == "normal":   # Minimizada o maximizada: se guarda la última normal
            self.config["geometria"] = self.root.geometry()
        self.config.guardar()
        self.btn.configure(state="disabled")
        self.btn_reconectar.configure(state="disabled")
        if self.activo:
            self.activo = False
            self.motor.detener()
        if self.motor.hilo_activo():
            # Se cierra al llegar "detenido"; respaldo por si el hilo no responde
            self.estado.configure(text="Guardando la última frase...")
            self.root.after(15000, self._destruir)
        else:
            self._procesar_cola()   # Mensajes pendientes (p. ej. la última frase)
            self._destruir()

    def _destruir(self):
        if not self.destruida:
            self.destruida = True
            if self.atajo_global is not None:
                self.atajo_global.detener()
            self._detener_traductor()
            self.root.destroy()

    # ---------- Actualización de la ventana ----------
    def revisar_cola(self):
        self._procesar_cola()
        if not self.destruida:
            self._actualizar_indicadores()  # Mismo sondeo de 100 ms: sin temporizador aparte
            self.root.after(100, self.revisar_cola)

    def _actualizar_indicadores(self):
        """Medidores de nivel y estado. Solo toca la ventana cuando algo cambió."""
        ahora = time.time()
        fuentes = {f.nombre: f for f in self.motor.fuentes if not f.retirar} if self.activo else {}
        for nombre, item in self.barras.items():
            barra, mostrado = item
            f = fuentes.get(nombre)
            objetivo = 0.0
            if f is not None and ahora - f.t_nivel < self.NIVEL_VIGENTE:
                # De -60 dBFS (vacío) a 0 dBFS (lleno): la voz normal queda a media barra
                objetivo = min(1.0, max(0.0, (20 * log10(max(f.nivel, 1e-6)) + 60) / 60))
            nuevo = max(objetivo, mostrado - 0.1)   # Caída suave, como un vúmetro
            if abs(nuevo - mostrado) > 0.01:
                x0, y0, _, y1 = self.medidores.coords(barra)
                self.medidores.coords(barra, x0, y0, x0 + nuevo * (self.ANCHO_MEDIDOR - 2), y1)
                item[1] = nuevo

        if not self.activo:
            estado, color = ("Error", self.ROJO) if self.hay_error else ("Detenido", self.GRIS)
        else:
            atraso = self.motor.atraso()
            if self.hay_error:
                estado, color = "Error", self.ROJO
            elif atraso > self.ATRASO_AVISO:
                estado, color = "Atrasado", self.AMARILLO
            elif self.motor.transcribiendo:
                estado, color = "Transcribiendo", self.ACENTO
            else:
                estado, color = "Escuchando", self.VERDE
            estado += f" · atraso {atraso:.1f} s"
        if self.texto_estado.cget("text") != estado:
            self.texto_estado.configure(text=estado)
        if self.punto_estado.cget("fg") != color:
            self.punto_estado.configure(fg=color)

    def _procesar_cola(self):
        try:
            while not self.destruida:
                tipo, valor = self.cola_ui.get_nowait()
                if tipo == "estado":
                    self.estado.configure(text=valor)
                    self.hay_error = valor.startswith("Error")
                elif tipo == "modelo_listo":
                    self.btn.configure(state="normal")
                    self._programar_liberar_vram()
                elif tipo == "parcial":
                    self._poner_parcial(valor)
                elif tipo == "final":
                    self._agregar_final(valor)
                elif tipo == "atajo" and valor == "subtitulos":
                    self.alternar_subtitulos()
                elif tipo == "traduccion":
                    self._recibir_traduccion(valor)
                elif tipo == "error_traduccion":
                    self.traductor = None
                    self.cambiar_vista_traduccion("Sin traducir")
                    messagebox.showerror("Traducción", valor)
                elif tipo == "error_mic":
                    self.mic.set(False)
                    messagebox.showerror("Error de micrófono",
                                         f"{valor}\n\nSe sigue transcribiendo solo el audio del PC.")
                elif tipo == "detenido":
                    if self.cerrando:
                        self._destruir()
                    else:
                        error = self.activo     # El hilo terminó solo: no pudo recargar el modelo
                        self.hay_error = error
                        if error:
                            self.activo = False
                            self.motor.detener()
                            self.btn_reconectar.configure(state="disabled")
                        self.estado.configure(text="Error al recargar el modelo (ver consola)" if error
                                              else "Detenido")
                        self.btn.configure(text="▶ Iniciar", state="normal")
                        self._programar_liberar_vram()
        except queue.Empty:
            pass

    def _mostrar_parciales(self):
        self.texto.configure(state="normal")
        self.texto.delete("parcial_ini", "end-1c")   # "end" borraría también el último salto de línea
        s = "\n".join(f"{fuente}: {t}" for fuente, t in self.parciales.items() if t)
        if s:
            self.texto.insert("end", s, "parcial")
        self.texto.configure(state="disabled")
        self.texto.see("end")

    def _poner_parcial(self, valor):
        fuente, s = valor
        self.parciales[fuente] = s
        self._mostrar_parciales()
        self._actualizar_franja()

    def _agregar_final(self, valor):
        fuente, s, inicio = valor
        self.parciales[fuente] = ""
        # Con dos fuentes solapadas una frase puede llegar después de otra que empezó más
        # tarde: se inserta en su lugar cronológico
        frase = Frase(inicio, fuente, s)
        bisect.insort_right(self.registro, frase, key=lambda f: f.inicio)
        self._guardar_registro()
        pos = bisect.bisect_right(self.finales, inicio, key=lambda f: f.inicio)
        self.finales.insert(pos, frase)
        self._dibujar_frase(pos, nueva=True)
        self._actualizar_franja()
        self._pedir_traduccion(frase)

    # ---------- Frases finales en la ventana ----------
    def _segmentos(self, frase):
        """Texto y etiquetas de una frase según la vista de traducción, para Text.insert."""
        hablante = f"hablante_{frase.fuente}"
        vista = self.config["traduccion"]
        texto = frase.traduccion if vista == "solo" and frase.traduccion else frase.texto
        segmentos = [f"{frase.hora}  ", "hora", f"{frase.fuente}: {texto}\n", hablante]
        if vista == "ambos" and frase.traduccion:
            segmentos += [f"    → {frase.traduccion}\n", (hablante, "traduccion")]
        return segmentos

    def _dibujar_frase(self, pos, nueva):
        """Inserta (nueva) o vuelve a dibujar la frase pos de self.finales. Cada frase
        empieza en una marca propia: así ocupa una o dos líneas sin descuadrar las demás."""
        t = self.texto
        frase = self.finales[pos]
        fin = self.finales[pos + 1].marca if pos + 1 < len(self.finales) else "parcial_ini"
        t.configure(state="normal")
        t.delete("parcial_ini", "end-1c")   # Lo provisional se vuelve a poner al final
        if nueva:
            frase.marca = f"frase_{self.n_marcas}"
            self.n_marcas += 1
            inicio = t.index(fin)
        else:
            inicio = t.index(frase.marca)
            t.delete(inicio, fin)
        t.insert(inicio, *self._segmentos(frase))
        # Gravedad derecha: si luego se inserta otra frase justo antes, la marca se corre con su texto
        t.mark_set(frase.marca, inicio)
        t.mark_gravity(frase.marca, "right")
        t.mark_set("parcial_ini", "end-1c")
        t.configure(state="disabled")
        self._mostrar_parciales()           # Parciales debajo de la última frase

    def _redibujar(self):
        """Todo de nuevo (al cambiar la vista de traducción)."""
        t = self.texto
        t.configure(state="normal")
        t.delete("1.0", "end")
        for frase in self.finales:
            t.mark_unset(frase.marca)
            frase.marca = f"frase_{self.n_marcas}"
            self.n_marcas += 1
            inicio = t.index("end-1c")
            t.insert(inicio, *self._segmentos(frase))
            t.mark_set(frase.marca, inicio)
            t.mark_gravity(frase.marca, "right")
        t.mark_set("parcial_ini", "end-1c")
        t.configure(state="disabled")
        self._mostrar_parciales()

    # ---------- Traducción ----------
    def cambiar_vista_traduccion(self, nombre):
        vista = VISTAS_TRADUCCION[nombre]
        self.vista.set(nombre)
        self.config["traduccion"] = vista
        self.config.guardar()
        if vista == "no":
            self._detener_traductor()       # Libera su RAM (~100 MB); volver a cargarlo tarda 0.2 s
        else:
            self._asegurar_traductor()
            if self.motor.contexto[0] != "en":
                self.estado.configure(text="La traducción se aplica cuando el idioma es English")
        self._redibujar()
        self._actualizar_franja()

    def _asegurar_traductor(self):
        if self.traductor is None:
            self.traductor = Traductor(self.cola_ui)
            self.traductor.start()

    def _detener_traductor(self):
        if self.traductor is not None:
            self.traductor.detener()
            self.traductor = None

    def _pedir_traduccion(self, frase):
        """Solo frases finales y solo si el audio es inglés: los parciales no se traducen."""
        if self.config["traduccion"] == "no" or self.motor.contexto[0] != "en":
            return
        self._asegurar_traductor()
        self.n_pedidos += 1
        self.por_traducir[self.n_pedidos] = frase
        self.traductor.pedir(self.n_pedidos, frase.texto)

    def _recibir_traduccion(self, valor):
        clave, traduccion = valor
        frase = self.por_traducir.pop(clave, None)
        if frase is None:
            return
        frase.traduccion = traduccion
        self._guardar_registro()
        for pos, f in enumerate(self.finales):  # Puede no estar: Limpiar la quitó de la ventana
            if f is frase:
                if self.config["traduccion"] != "no":
                    self._dibujar_frase(pos, nueva=False)
                break
        self._actualizar_franja()

    def _guardar_registro(self):
        """Reescribe el .txt automático ya ordenado. Se escribe a un temporal y se
        reemplaza, para que un corte a mitad de escritura no deje el archivo a medias."""
        tmp = self.archivo_auto + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(l + "\n" for frase in self.registro for l in frase.lineas_txt())
            os.replace(tmp, self.archivo_auto)
        except OSError as e:
            log.error("No se pudo guardar %s: %s", self.archivo_auto, e)
            self.estado.configure(text="Error al guardar el .txt (ver consola)")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
