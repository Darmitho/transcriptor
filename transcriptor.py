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
import logging
import threading
import time
import datetime
import unicodedata
from collections import Counter, deque
from math import gcd


# ======================= CONFIGURACIÓN =======================
MODELO = "large-v3-turbo"       # Si va lento: "medium" o "small"
DISPOSITIVO = "cuda"
TIPO_COMPUTO = "int8_float16"   # Bajo uso de VRAM, buena calidad en RTX 3050
IDIOMA = "es"                   # None = detección automática
SR_MODELO = 16000

INTERVALO_PARCIAL = 1.0         # Cada cuántos segundos se refresca el texto provisional
SILENCIO_FIN = 0.7              # Segundos sin voz (según el VAD) que cierran una frase
MAX_FRASE = 12.0                # Corte forzado de frases largas (segundos)
TROZO = 0.1                     # Duración de cada bloque de audio capturado (segundos)

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
)
SOSP_LOGPROB = -0.5             # Sospechosa con avg_logprob menor: se descarta
SOSP_NO_VOZ = 0.3               # Sospechosa con no_speech_prob mayor: se descarta
SOSP_VOZ_MIN = 0.3              # Sospechosa, sola en la frase, con menos voz que esto: se descarta

PROMPT_FIJO = None              # Ej.: "Hola. Esta es una conversación, con puntuación." (None = sin prompt)

# Carpeta local (fuera de OneDrive) donde se guardan las transcripciones automáticas
CARPETA_SALIDA = os.path.expandvars(r"%USERPROFILE%\Documents\Transcripciones")
# True = nunca conectarse a Hugging Face (el modelo debe estar ya descargado).
# Poner False solo una vez si hace falta descargar un modelo nuevo.
MODO_OFFLINE = True
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
from tkinter import filedialog, messagebox


def _normalizar(texto):
    """Minúsculas, sin tildes ni puntuación: '¡Gracias!' -> 'gracias'."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


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


class Motor:
    """Captura de audio + transcripción en un hilo aparte."""

    def __init__(self, cola_ui):
        self.cola_ui = cola_ui
        self.cola_audio = queue.Queue()
        self.stop = threading.Event()
        self.modelo = None
        self.pa = None
        self.stream = None
        self.hilo = None
        self.tasa = 48000

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

    def _callback(self, in_data, frame_count, time_info, status):
        x = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.canales).mean(axis=1)
        self.cola_audio.put(x.copy())
        return (None, pyaudio.paContinue)

    def hilo_activo(self):
        return self.hilo is not None and self.hilo.is_alive()

    def iniciar(self):
        if self.hilo_activo():
            raise RuntimeError("La captura anterior aún se está cerrando, intenta de nuevo.")
        # Cola y evento nuevos por sesión: nada de una sesión anterior se mezcla con esta
        self.stop = threading.Event()
        self.cola_audio = queue.Queue()
        self.pa = pyaudio.PyAudio()
        try:
            disp = self._dispositivo_loopback()
            self.canales = disp["maxInputChannels"]
            self.tasa = int(disp["defaultSampleRate"])
            g = gcd(SR_MODELO, self.tasa)
            self.up, self.down = SR_MODELO // g, self.tasa // g
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.canales,
                rate=self.tasa,
                input=True,
                input_device_index=disp["index"],
                frames_per_buffer=int(self.tasa * TROZO),
                stream_callback=self._callback,
            )
            self.stream.start_stream()
        except Exception:
            if self.stream:
                self.stream.close()
            self.pa.terminate()
            self.stream = self.pa = None
            raise
        self.hilo = threading.Thread(target=self._bucle, args=(self.stop, self.cola_audio), daemon=True)
        self.hilo.start()
        self.cola_ui.put(("estado", f"Escuchando: {disp['name']}"))

    def detener(self):
        """Corta la captura. El hilo cierra la frase pendiente y avisa con 'detenido'."""
        try:
            if self.stream:
                self.stream.stop_stream()   # Espera a que termine el callback en curso
                self.stream.close()
        finally:
            if self.pa:
                self.pa.terminate()
            self.stream = self.pa = None
            self.stop.set()
        self.cola_ui.put(("estado", "Deteniendo..."))

    # ---------- Transcripción ----------
    def _motivo_descarte(self, s, n_segs, voz_seg):
        """Devuelve por qué se descarta un segmento de Whisper, o None si se conserva."""
        if s.compression_ratio > MAX_COMPRESION:
            return "compresion"
        if s.no_speech_prob > MAX_NO_VOZ:
            return "no_voz"
        if s.avg_logprob < MIN_LOGPROB:
            return "logprob"
        if _normalizar(s.text) in FRASES_SOSPECHOSAS and (
                s.avg_logprob < SOSP_LOGPROB or s.no_speech_prob > SOSP_NO_VOZ
                or (n_segs == 1 and voz_seg < SOSP_VOZ_MIN)):
            return "sospechosa"
        return None

    def _transcribir(self, trozos, final, voz_seg, stats=None):
        t0 = time.perf_counter()
        audio = np.concatenate(trozos)
        audio = resample_poly(audio, self.up, self.down).astype(np.float32)
        segmentos, _ = self.modelo.transcribe(
            audio,
            language=IDIOMA,
            beam_size=5 if final else 1,
            vad_filter=False,               # El VAD propio ya decidió que hay voz
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=PROMPT_FIJO,
        )
        segmentos = list(segmentos)
        partes, descartes = [], Counter()
        for s in segmentos:
            motivo = self._motivo_descarte(s, len(segmentos), voz_seg)
            if motivo:
                descartes[motivo] += 1
            else:
                partes.append(s.text.strip())
        if stats is not None:
            stats.update(descartes)
        # Solo métricas, nunca el texto (puede ser información de clientes)
        seg_audio = len(audio) / SR_MODELO
        dt = time.perf_counter() - t0
        log.info("%-7s audio %5.1fs voz %4.1fs -> %.2fs (RTF %.2f) cola=%d segs %d/%d%s",
                 "final" if final else "parcial", seg_audio, voz_seg, dt, dt / max(seg_audio, 1e-6),
                 self.cola_audio.qsize(), len(partes), len(segmentos),
                 "".join(f" {k}={v}" for k, v in descartes.items()))
        return " ".join(partes).strip()

    def _recortar(self, trozos, voz):
        """Quita el silencio inicial y final, dejando RECORTE segundos de margen."""
        con_voz = [i for i, v in enumerate(voz) if v > 0]
        margen = max(1, round(RECORTE / TROZO))
        return trozos[max(0, con_voz[0] - margen):con_voz[-1] + 1 + margen]

    def _cerrar_frase(self, trozos, voz, stats):
        voz_seg = sum(voz)
        if voz_seg < VOZ_MIN:
            stats["omitida_vad"] += 1
            log.info("omitida audio %5.1fs voz %4.2fs (< VOZ_MIN)", len(trozos) * TROZO, voz_seg)
            self.cola_ui.put(("parcial", ""))
            return
        texto = self._transcribir(self._recortar(trozos, voz), True, voz_seg, stats)
        self.cola_ui.put(("parcial", ""))
        stats["frases"] += 1
        if texto:
            self.cola_ui.put(("final", texto))

    def _bucle(self, stop, cola_audio):
        vad = DetectorVoz(self.up, self.down)
        preroll = deque(maxlen=max(1, round(PRE_ROLL / TROZO)))
        trozos, voz = [], []                # Audio de la frase actual y segundos de voz de cada trozo
        duracion = silencio = 0.0           # En tiempo de audio
        ultimo_audio = ultimo_parcial = time.time()
        stats = Counter()

        def agregar(x):
            nonlocal duracion, silencio
            v = vad.segundos_voz(x)
            silencio = 0.0 if v > 0 else silencio + len(x) / self.tasa
            if not trozos:
                if v == 0:                  # Sin voz y sin frase abierta: solo pre-roll
                    preroll.append(x)
                    return
                for p in preroll:
                    trozos.append(p)
                    voz.append(0.0)
                    duracion += len(p) / self.tasa
                preroll.clear()
            trozos.append(x)
            voz.append(v)
            duracion += len(x) / self.tasa

        def cerrar():
            nonlocal trozos, voz, duracion
            try:
                self._cerrar_frase(trozos, voz, stats)
            finally:
                trozos, voz, duracion = [], [], 0.0

        while not stop.is_set():
            # Recoger todo el audio disponible
            try:
                x = cola_audio.get(timeout=0.1)
                while True:
                    agregar(x)
                    ultimo_audio = time.time()
                    if trozos and (silencio >= SILENCIO_FIN or duracion >= MAX_FRASE):
                        break               # Cerrar antes de mezclar con la frase siguiente
                    x = cola_audio.get_nowait()
            except queue.Empty:
                pass

            if not trozos:
                continue
            # WASAPI loopback puede dejar de entregar audio cuando nada suena
            sin_audio = time.time() - ultimo_audio >= SILENCIO_FIN
            try:
                if silencio >= SILENCIO_FIN or sin_audio or duracion >= MAX_FRASE:
                    cerrar()
                elif time.time() - ultimo_parcial >= INTERVALO_PARCIAL:
                    voz_seg = sum(voz)
                    if voz_seg >= VOZ_MIN:
                        self.cola_ui.put(("parcial", self._transcribir(
                            self._recortar(trozos, voz), False, voz_seg)))
                    ultimo_parcial = time.time()
            except Exception as e:
                self.cola_ui.put(("estado", f"Error: {e}"))

        # Audio que quedó en la cola al detener (el stream ya está cerrado)
        try:
            while True:
                try:
                    agregar(cola_audio.get_nowait())
                except queue.Empty:
                    break
            if trozos:
                cerrar()
        except Exception as e:
            self.cola_ui.put(("estado", f"Error: {e}"))
        finally:
            log.info("resumen: frases=%d omitidas_vad=%d descartes:%s | VAD %.2f ms por s de audio",
                     stats.pop("frases", 0), stats.pop("omitida_vad", 0),
                     "".join(f" {k}={v}" for k, v in stats.items()) or " ninguno", vad.ms_por_segundo())
            self.cola_ui.put(("detenido", None))


class App:
    FONDO, TEXTO, GRIS, ACENTO = "#1e1f22", "#e8e8e8", "#8a8d93", "#6ea8fe"

    def __init__(self, root):
        self.root = root
        self.cola_ui = queue.Queue()
        self.motor = Motor(self.cola_ui)
        self.finales = []
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
        root.geometry("760x440")
        root.configure(bg=self.FONDO)
        root.attributes("-topmost", True)

        barra = tk.Frame(root, bg=self.FONDO)
        barra.pack(fill="x", padx=8, pady=6)
        estilo = dict(bg="#2b2d31", fg=self.TEXTO, activebackground="#3a3d43",
                      activeforeground=self.TEXTO, relief="flat", padx=10, pady=4)
        self.btn = tk.Button(barra, text="▶ Iniciar", command=self.alternar, state="disabled", **estilo)
        self.btn.pack(side="left")
        tk.Button(barra, text="Limpiar", command=self.limpiar, **estilo).pack(side="left", padx=4)
        tk.Button(barra, text="Guardar...", command=self.guardar, **estilo).pack(side="left")
        self.encima = tk.BooleanVar(value=True)
        tk.Checkbutton(barra, text="Siempre encima", variable=self.encima, command=self._encima,
                       bg=self.FONDO, fg=self.TEXTO, selectcolor="#2b2d31",
                       activebackground=self.FONDO, activeforeground=self.TEXTO).pack(side="left", padx=8)
        self.estado = tk.Label(barra, text="", bg=self.FONDO, fg=self.GRIS, anchor="e")
        self.estado.pack(side="right", fill="x", expand=True)

        self.texto = tk.Text(root, wrap="word", font=("Segoe UI", 14), bg=self.FONDO, fg=self.TEXTO,
                             insertbackground=self.TEXTO, relief="flat", padx=12, pady=8)
        self.texto.pack(fill="both", expand=True)
        self.texto.tag_config("parcial", foreground=self.GRIS)
        self.texto.tag_config("hora", foreground=self.ACENTO, font=("Segoe UI", 10))
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.mark_gravity("parcial_ini", "left")
        self.texto.configure(state="disabled")

        root.protocol("WM_DELETE_WINDOW", self.cerrar)
        threading.Thread(target=self.motor.cargar_modelo, daemon=True).start()
        root.after(100, self.revisar_cola)

    # ---------- Controles ----------
    def alternar(self):
        if self.activo:
            # El botón se reactiva cuando el hilo termina la última frase ("detenido")
            self.btn.configure(state="disabled")
            self.activo = False
            self.motor.detener()
        else:
            try:
                self.motor.iniciar()
            except Exception as e:
                messagebox.showerror("Error de audio", str(e))
                return
            self.btn.configure(text="■ Detener")
            self.activo = True

    def _encima(self):
        self.root.attributes("-topmost", self.encima.get())

    def limpiar(self):
        self.finales.clear()
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.mark_set("parcial_ini", "1.0")
        self.texto.configure(state="disabled")

    def guardar(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".txt", initialdir=CARPETA_SALIDA,
                                            filetypes=[("Texto", "*.txt")])
        if ruta:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write("\n".join(f"[{h}] {t}" for h, t in self.finales))

    def cerrar(self):
        if self.cerrando:
            return
        self.cerrando = True
        self.btn.configure(state="disabled")
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
                elif tipo == "parcial":
                    self._poner_parcial(valor)
                elif tipo == "final":
                    self._agregar_final(valor)
                elif tipo == "detenido":
                    if self.cerrando:
                        self._destruir()
                    else:
                        self.estado.configure(text="Detenido")
                        self.btn.configure(text="▶ Iniciar", state="normal")
        except queue.Empty:
            pass

    def _poner_parcial(self, s):
        self.texto.configure(state="normal")
        self.texto.delete("parcial_ini", "end")
        if s:
            self.texto.insert("end", s, "parcial")
        self.texto.configure(state="disabled")
        self.texto.see("end")

    def _agregar_final(self, s):
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        self.finales.append((hora, s))
        with open(self.archivo_auto, "a", encoding="utf-8") as f:
            f.write(f"[{hora}] {s}\n")
        self.texto.configure(state="normal")
        self.texto.delete("parcial_ini", "end")
        self.texto.insert("end", f"{hora}  ", "hora")
        self.texto.insert("end", s + "\n")
        self.texto.mark_set("parcial_ini", "end-1c")
        self.texto.configure(state="disabled")
        self.texto.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
