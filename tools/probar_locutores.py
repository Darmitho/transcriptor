"""Transcribe una grabación con el flujo de la app y etiqueta cada frase con "Persona N".

Uso:  venv\\Scripts\\python.exe tools\\probar_locutores.py "ruta\\a\\reunion.mp3" [--idioma es] [--umbral 0.5]
Sirve para validar la detección de personas con audio real antes de integrarla (es opcional y
no está en la app). Todo local: Whisper en la GPU y la huella de voz (3D-Speaker CAM++) en CPU.
Escribe el resultado en CARPETA_SALIDA como <nombre>_personas.txt.

Agrupación en línea (igual que en tools\\poc_locutores.py): cada frase se compara con las voces
ya vistas en la sesión; si se parece lo suficiente se le asigna esa persona, si no es una nueva.
Las frases con poca voz (< VOZ_MIN_NUEVA) solo se asignan si el parecido es alto; si no, "?".
"""

import argparse
import datetime
import os
import queue
import sys
import threading
import time

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from comun import T, alimentar  # noqa: E402

MODELO_HUELLA = os.path.join(os.path.expanduser("~"), ".cache", "transcriptor", "locutores",
                             "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
UMBRAL = 0.50                           # Ajustado en la reunión AMI ES2004a (poc_locutores.py)
VOZ_MIN_NUEVA = 1.5
MARGEN_CORTAS = 0.10
TASA = 48000                            # El audio se lleva a la tasa de un dispositivo WASAPI típico


def leer_audio(ruta):
    """Cualquier formato que lea PyAV (mp3, wav, m4a...) -> float32 mono a TASA."""
    import av
    partes = []
    with av.open(ruta) as c:
        r = av.AudioResampler(format="flt", layout="mono", rate=TASA)
        for frame in c.decode(audio=0):
            partes += [f.to_ndarray().reshape(-1) for f in r.resample(frame)]
        partes += [f.to_ndarray().reshape(-1) for f in r.resample(None)]
    return np.concatenate(partes).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--idioma", default="es", choices=["es", "en"])
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    args = ap.parse_args()
    import logging
    import sherpa_onnx
    logging.getLogger("transcriptor").setLevel(logging.WARNING)

    audio = leer_audio(args.archivo)
    print(f"{os.path.basename(args.archivo)}: {len(audio) / TASA / 60:.1f} min. Transcribiendo (~{len(audio) / TASA / 4 / 60:.0f} min)...")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODELO_HUELLA, num_threads=1))
    m = T.Motor(queue.Queue())
    m.cargar_modelo()
    m.cambiar_idioma(args.idioma)

    # Al cerrar cada frase se guarda su huella de voz, con la hora de inicio como clave
    huellas = {}
    cerrar = m._cerrar_frase

    def _cerrar_frase(fuente, motivo):
        voz = sum(fuente.voz)
        if voz >= T.VOZ_MIN:
            ini, fin = m._recortar(fuente.voz)
            a16 = resample_poly(np.concatenate(fuente.trozos[ini:fin]), fuente.up, fuente.down).astype(np.float32)
            s = extractor.create_stream()
            s.accept_waveform(T.SR_MODELO, a16)
            s.input_finished()
            e = np.array(extractor.compute(s), dtype=np.float32)
            huellas[fuente.inicio_voz()] = (e / np.linalg.norm(e), voz)
        return cerrar(fuente, motivo)

    m._cerrar_frase = _cerrar_frase
    m.stop, m.cola_audio = threading.Event(), queue.Queue()
    f = T.Fuente("Otros", {"maxInputChannels": 1, "defaultSampleRate": float(TASA), "index": 0}, m.cola_audio)
    m.fuentes = [f]
    m.hilo = threading.Thread(target=m._bucle, args=(m.stop, m.cola_audio, m.fuentes), daemon=True)
    t0 = time.time()
    m.hilo.start()
    alimentar({"Otros": audio}, {"Otros": f}, m.cola_audio, velocidad=4.0)
    time.sleep(2)
    m.stop.set()
    m.hilo.join(300)
    finales = [v for k, v in m.cola_ui.queue if k == "final"]

    # Agrupación en línea, en el orden de la conversación
    sumas, lineas, por_persona = [], [], {}
    for _, texto, inicio in sorted(finales, key=lambda v: v[2]):
        huella, voz = huellas.get(inicio, (None, 0))
        etiqueta = "?"
        if huella is not None:
            sims = [float(huella @ (s / np.linalg.norm(s))) for s in sumas]
            mejor = int(np.argmax(sims)) if sims else -1
            parecido = sims[mejor] if sims else -1.0
            if voz >= VOZ_MIN_NUEVA:
                if parecido >= args.umbral:
                    sumas[mejor] += huella
                else:
                    sumas.append(huella.copy())
                    mejor = len(sumas) - 1
                etiqueta = f"Persona {mejor + 1}"
            elif parecido >= args.umbral + MARGEN_CORTAS:
                etiqueta = f"Persona {mejor + 1}"
        segundo = (inicio - t0) * 4
        lineas.append(f"[{int(segundo // 60):02d}:{int(segundo % 60):02d}] {etiqueta}: {texto}")
        por_persona[etiqueta] = por_persona.get(etiqueta, 0) + 1

    nombre = os.path.splitext(os.path.basename(args.archivo))[0] + "_personas.txt"
    salida = os.path.join(T.CARPETA_SALIDA, nombre)
    os.makedirs(T.CARPETA_SALIDA, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write(f"# {os.path.basename(args.archivo)} | idioma {args.idioma} | umbral {args.umbral} | "
                 f"{datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        fh.write("\n".join(lineas) + "\n")
    print(f"\n{len(lineas)} frases | " + " | ".join(f"{k}: {v}" for k, v in sorted(por_persona.items())))
    print(f"Resultado en {salida}")


if __name__ == "__main__":
    main()
