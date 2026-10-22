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
from math import gcd


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

# --- Contexto fijo para Whisper (nunca se alimenta del propio texto transcrito) ---
# Nombres propios y jerga que Whisper no conoce. Se le pasan como una frase natural y
# puntuada en el idioma elegido ("Hoy hablamos de A, B y C."), que en pruebas dio mejor
# resultado que hotwords o un prompt descriptivo. Mantener la lista corta. () = sin contexto.
VOCABULARIO = ("Claude", "CriptoBuzz", "pullback", "order block", "RSI")

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
        self.cambiar_idioma(IDIOMA)

    def cambiar_idioma(self, idioma):
        """Se puede llamar en marcha desde la UI: aplica desde la próxima transcripción.
        Idioma y prompt van en una sola tupla para que el hilo nunca lea uno sin el otro."""
        self.contexto = (idioma, _prompt_vocabulario(VOCABULARIO, idioma))

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

    def iniciar(self, con_mic=False):
        if self.hilo_activo():
            raise RuntimeError("La captura anterior aún se está cerrando, intenta de nuevo.")
        # Cola y evento nuevos por sesión: nada de una sesión anterior se mezcla con esta
        self.stop = threading.Event()
        self.cola_audio = queue.Queue()
        self.pa = pyaudio.PyAudio()
        try:
            otros = Fuente(NOMBRE_OTROS, self._dispositivo_loopback(), self.cola_audio)
            otros.abrir(self.pa)
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

    def reconectar(self, con_mic):
        """Vuelve a abrir la captura con los dispositivos predeterminados actuales de Windows
        (p. ej. tras conectar audífonos), sin detener el hilo ni recargar el modelo.
        Se llama desde la UI."""
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
        otros = Fuente(NOMBRE_OTROS, self._dispositivo_loopback(), self.cola_audio)
        otros.abrir(self.pa)
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
        segmentos, _ = self.modelo.transcribe(
            audio,
            language=idioma,
            beam_size=5 if final else 1,
            vad_filter=False,               # El VAD propio ya decidió que hay voz
            condition_on_previous_text=False,
            # word_timestamps junto con without_timestamps hace que Whisper siga
            # transcribiendo la cola de silencio (alucina "Gracias..."): no combinarlos
            without_timestamps=not palabras,
            word_timestamps=palabras,
            initial_prompt=prompt,
        )
        segmentos = list(segmentos)
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
                self.cola_ui.put(("final", (fuente.nombre, texto, fuente.inicio_voz())))
        finally:
            fuente.reiniciar_frase(*resto)

    def _parcial(self, fuente):
        fuente.ultimo_parcial = time.time()
        voz_seg = sum(fuente.voz)
        if voz_seg >= VOZ_MIN:
            ini, fin = self._recortar(fuente.voz)
            texto = self._transcribir(fuente, fuente.trozos[ini:fin], False, voz_seg)[0]
            self.cola_ui.put(("parcial", (fuente.nombre, texto)))
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


class Configuracion(dict):
    """Preferencias de la ventana guardadas entre sesiones en CONFIG_ARCHIVO.
    Un archivo dañado o con valores de otro tipo no impide abrir: se usan los valores iniciales."""

    INICIAL = {"idioma": IDIOMA, "siempre_encima": True, "geometria": "760x440",
               "tamano_letra": 14, "color_yo": "#7fd18b", "color_otros": "#e8e8e8"}
    LETRA_MIN, LETRA_MAX = 9, 36

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


class App:
    FONDO, TEXTO, GRIS, ACENTO = "#1e1f22", "#e8e8e8", "#8a8d93", "#6ea8fe"

    def __init__(self, root):
        self.root = root
        self.config = Configuracion()
        if self.config["idioma"] not in IDIOMAS.values():
            self.config["idioma"] = IDIOMA
        self.cola_ui = queue.Queue()
        self.motor = Motor(self.cola_ui)
        self.motor.cambiar_idioma(self.config["idioma"])
        # Frases finales (inicio, hora, fuente, texto) ordenadas por inicio del habla:
        # las de la ventana (Limpiar las borra) y las de la sesión (van al .txt automático)
        self.finales, self.registro = [], []
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
        # Idioma que se habla en el audio (no traduce): aplica desde la próxima frase
        self.idioma = tk.StringVar(value=next(n for n, c in IDIOMAS.items() if c == self.config["idioma"]))
        menu = tk.OptionMenu(barra, self.idioma, *IDIOMAS, command=self._cambiar_idioma)
        menu.configure(highlightthickness=0, **estilo)
        menu["menu"].configure(bg="#2b2d31", fg=self.TEXTO, activebackground="#3a3d43")
        menu.pack(side="left", padx=4)
        tk.Button(barra, text="⚙", command=self.abrir_ajustes, **estilo).pack(side="left")
        self.estado = tk.Label(barra, text="", bg=self.FONDO, fg=self.GRIS, anchor="e")
        self.estado.pack(side="right", fill="x", expand=True)

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

        # Fuentes con nombre: cambiar su tamaño actualiza todo el texto que las usa
        self.letra = tkfont.Font(family="Segoe UI", size=self.config["tamano_letra"])
        self.letra_hora = tkfont.Font(family="Segoe UI", size=max(8, self.config["tamano_letra"] - 4))
        self.texto = tk.Text(root, wrap="word", font=self.letra, bg=self.FONDO, fg=self.TEXTO,
                             insertbackground=self.TEXTO, relief="flat", padx=12, pady=8)
        self.texto.pack(fill="both", expand=True)
        self.texto.tag_config("parcial", foreground=self.GRIS)
        self.texto.tag_config("hora", foreground=self.ACENTO, font=self.letra_hora)
        self._aplicar_colores()
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.mark_gravity("parcial_ini", "left")
        self.texto.configure(state="disabled")
        self.ajustes = None                 # Ventana de ajustes, si está abierta

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
            try:
                self.motor.iniciar(con_mic=self.mic.get())
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
            self.motor.reconectar(con_mic=self.mic.get())
        except Exception as e:
            messagebox.showerror("Error de audio",
                                 f"{e}\n\nRevisa el dispositivo de salida en Windows y pulsa Reconectar otra vez.")

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

    # ---------- Letra y colores ----------
    def cambiar_letra(self, delta=0, tamano=None):
        tamano = tamano if tamano is not None else self.config["tamano_letra"] + delta
        tamano = max(Configuracion.LETRA_MIN, min(Configuracion.LETRA_MAX, int(tamano)))
        self.letra.configure(size=tamano)
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
        tk.Button(v, text="Cerrar", command=v.destroy, **self.estilo).grid(row=9, column=0, columnspan=2, pady=(12, 0))

    def limpiar(self):
        self.finales.clear()
        self.parciales.clear()
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.configure(state="disabled")

    def guardar(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".txt", initialdir=CARPETA_SALIDA,
                                            filetypes=[("Texto", "*.txt")])
        if ruta:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write("\n".join(f"[{h}] {fuente}: {t}" for _, h, fuente, t in self.finales))

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
            self.root.destroy()

    # ---------- Actualización de la ventana ----------
    def revisar_cola(self):
        self._procesar_cola()
        if not self.destruida:
            self.root.after(100, self.revisar_cola)

    def _procesar_cola(self):
        try:
            while not self.destruida:
                tipo, valor = self.cola_ui.get_nowait()
                if tipo == "estado":
                    self.estado.configure(text=valor)
                elif tipo == "modelo_listo":
                    self.btn.configure(state="normal")
                    self._programar_liberar_vram()
                elif tipo == "parcial":
                    self._poner_parcial(valor)
                elif tipo == "final":
                    self._agregar_final(valor)
                elif tipo == "error_mic":
                    self.mic.set(False)
                    messagebox.showerror("Error de micrófono",
                                         f"{valor}\n\nSe sigue transcribiendo solo el audio del PC.")
                elif tipo == "detenido":
                    if self.cerrando:
                        self._destruir()
                    else:
                        error = self.activo     # El hilo terminó solo: no pudo recargar el modelo
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

    def _agregar_final(self, valor):
        fuente, s, inicio = valor
        self.parciales[fuente] = ""
        # Hora en que empezó a hablarse, no en que terminó de transcribirse. Con dos
        # fuentes solapadas una frase puede llegar después de otra que empezó más tarde:
        # se inserta en su lugar cronológico
        hora = datetime.datetime.fromtimestamp(inicio).strftime("%H:%M:%S")
        frase = (inicio, hora, fuente, s)
        bisect.insort_right(self.registro, frase, key=lambda f: f[0])
        self._guardar_registro()
        pos = bisect.bisect_right(self.finales, inicio, key=lambda f: f[0])
        self.finales.insert(pos, frase)
        self.texto.configure(state="normal")
        self.texto.delete("parcial_ini", "end-1c")   # "end" borraría también el último salto de línea
        linea = pos + 1                     # Cada frase final ocupa una línea del texto
        self.texto.insert(f"{linea}.0", f"{hora}  ", "hora")
        self.texto.insert(f"{linea}.{len(hora) + 2}", f"{fuente}: {s}\n", f"hablante_{fuente}")
        self.texto.mark_set("parcial_ini", "end-1c")
        self.texto.configure(state="disabled")
        self._mostrar_parciales()           # Parciales de otras fuentes, debajo del final

    def _guardar_registro(self):
        """Reescribe el .txt automático ya ordenado. Se escribe a un temporal y se
        reemplaza, para que un corte a mitad de escritura no deje el archivo a medias."""
        tmp = self.archivo_auto + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(f"[{h}] {fuente}: {t}\n" for _, h, fuente, t in self.registro)
            os.replace(tmp, self.archivo_auto)
        except OSError as e:
            log.error("No se pudo guardar %s: %s", self.archivo_auto, e)
            self.estado.configure(text="Error al guardar el .txt (ver consola)")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
