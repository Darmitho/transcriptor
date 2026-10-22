"""Compara modelos de traducción inglés -> español sobre CTranslate2: calidad, velocidad y memoria.

Uso:  venv\\Scripts\\python.exe tools\\bench_traduccion.py
Necesita los modelos ya convertidos en CARPETA_MODELOS (ver tools\\convertir_traduccion.md) y
sacrebleu (requirements-dev.txt). Cada combinación modelo/dispositivo corre en un proceso
aparte para medir su RAM y VRAM sin mezclarlas.

Mide además el efecto sobre Whisper: cuánto se alarga una transcripción mientras el
traductor trabaja en paralelo en la GPU o en la CPU.
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_MODELOS = os.path.join(os.path.expanduser("~"), ".cache", "transcriptor", "traduccion")
FRASES = os.path.join(RAIZ, "tests", "traduccion_en_es.tsv")
MODELOS = {
    "opus-mt-en-es": "Apache-2.0",
    "opus-mt-tc-big-en-es": "CC-BY-4.0",
    "nllb-200-distilled-600M": "CC-BY-NC-4.0 (solo referencia)",
}
HILOS_CPU = 4                           # Deja núcleos libres para el VAD, la ventana y el sistema


def cargar_frases():
    with open(FRASES, encoding="utf-8") as f:
        pares = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    return [p[0] for p in pares], [p[1] for p in pares]


class Traductor:
    """Mismo código que usaría la app: SentencePiece + ctranslate2.Translator."""

    def __init__(self, nombre, dispositivo):
        import ctranslate2
        import sentencepiece as spm
        ruta = os.path.join(CARPETA_MODELOS, nombre)
        self.nllb = nombre.startswith("nllb")
        tipo = "int8_float16" if dispositivo == "cuda" else "int8"
        self.modelo = ctranslate2.Translator(ruta, device=dispositivo, compute_type=tipo,
                                             inter_threads=1, intra_threads=HILOS_CPU)
        if self.nllb:
            self.sp_origen = self.sp_destino = spm.SentencePieceProcessor(
                model_file=os.path.join(ruta, "sentencepiece.bpe.model"))
        else:
            self.sp_origen = spm.SentencePieceProcessor(model_file=os.path.join(ruta, "source.spm"))
            self.sp_destino = spm.SentencePieceProcessor(model_file=os.path.join(ruta, "target.spm"))

    def traducir(self, texto, beam=4):
        piezas = self.sp_origen.encode(texto, out_type=str)
        if self.nllb:
            piezas = ["eng_Latn"] + piezas + ["</s>"]
            r = self.modelo.translate_batch([piezas], beam_size=beam, target_prefix=[["spa_Latn"]])
            salida = r[0].hypotheses[0][1:]
        else:
            r = self.modelo.translate_batch([piezas + ["</s>"]], beam_size=beam)
            salida = r[0].hypotheses[0]
        return self.sp_destino.decode(salida)


def hijo(nombre, dispositivo):
    """Proceso medido: carga un modelo en un dispositivo, traduce las frases y reporta."""
    sys.path.insert(0, RAIZ)
    import transcriptor  # noqa: F401  (hace visibles las DLL de CUDA instaladas por pip)
    import psutil
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    proc = psutil.Process()
    import ctranslate2  # noqa: F401  (para que la RAM base incluya la librería)
    ram0, vram0 = proc.memory_info().rss, pynvml.nvmlDeviceGetMemoryInfo(gpu).used
    t0 = time.perf_counter()
    tr = Traductor(nombre, dispositivo)
    carga = time.perf_counter() - t0
    origen, _ = cargar_frases()
    tr.traducir(origen[0])                  # Calentar
    tiempos, salidas = [], []
    proc.cpu_percent(None)
    t_total = time.perf_counter()
    for frase in origen:
        t = time.perf_counter()
        salidas.append(tr.traducir(frase))
        tiempos.append(time.perf_counter() - t)
    cpu = proc.cpu_percent(None)
    print(json.dumps({
        "carga_s": carga, "ram_mb": (proc.memory_info().rss - ram0) / 2**20,
        "vram_mb": (pynvml.nvmlDeviceGetMemoryInfo(gpu).used - vram0) / 2**20,
        "ms_mediana": statistics.median(tiempos) * 1000,
        "ms_max": max(tiempos) * 1000, "cpu_pct": cpu, "seg_total": time.perf_counter() - t_total,
        "salidas": salidas}, ensure_ascii=False))


def efecto_en_whisper():
    """Duración de un parcial de Whisper de 4 s: solo, y con el traductor trabajando a la vez."""
    sys.path.insert(0, os.path.join(RAIZ, "tests"))
    import queue
    import numpy as np
    from scipy.signal import resample_poly
    from comun import T, wav
    m = T.Motor(queue.Queue())
    m.cargar_modelo()
    audio = resample_poly(wav("v2", recortar=True), 1, 3).astype(np.float32)

    def parcial():
        t = time.perf_counter()
        list(m.modelo.transcribe(audio, language="es", beam_size=1, without_timestamps=True,
                                 condition_on_previous_text=False)[0])
        return time.perf_counter() - t

    origen, _ = cargar_frases()
    resultados = {}
    for etiqueta, dispositivo in (("solo Whisper", None), ("+ traductor en GPU", "cuda"),
                                  ("+ traductor en CPU", "cpu")):
        seguir = threading.Event()
        if dispositivo:
            tr = Traductor("opus-mt-en-es", dispositivo)
            seguir.set()

            def traducir_sin_parar():
                i = 0
                while seguir.is_set():
                    tr.traducir(origen[i % len(origen)])
                    i += 1

            hilo = threading.Thread(target=traducir_sin_parar, daemon=True)
            hilo.start()
        parcial()
        tiempos = [parcial() for _ in range(10)]
        seguir.clear()
        if dispositivo:
            hilo.join()
            del tr
        resultados[etiqueta] = statistics.median(tiempos) * 1000
    return resultados


def main():
    import sacrebleu
    origen, referencias = cargar_frases()
    print(f"{len(origen)} frases | CPU: {HILOS_CPU} hilos | GPU: int8_float16\n")
    print(f"{'modelo':26s} {'disp.':5s} {'chrF':>5s} {'BLEU':>5s} {'ms/frase':>9s} {'máx ms':>7s} "
          f"{'CPU %':>6s} {'RAM MB':>7s} {'VRAM MB':>8s} {'carga s':>8s}  licencia")
    muestras = {}
    for nombre, licencia in MODELOS.items():
        for dispositivo in ("cpu", "cuda"):
            p = subprocess.run([sys.executable, __file__, "--hijo", nombre, dispositivo],
                               capture_output=True, text=True, encoding="utf-8")
            lineas = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
            if not lineas:
                print(f"{nombre:26s} {dispositivo:5s} FALLÓ: {p.stderr.strip().splitlines()[-1:]}")
                continue
            r = json.loads(lineas[-1])
            chrf = sacrebleu.corpus_chrf(r["salidas"], [referencias]).score
            bleu = sacrebleu.corpus_bleu(r["salidas"], [referencias]).score
            muestras[nombre] = r["salidas"]
            print(f"{nombre:26s} {dispositivo:5s} {chrf:5.1f} {bleu:5.1f} {r['ms_mediana']:9.0f} {r['ms_max']:7.0f} "
                  f"{r['cpu_pct']:6.0f} {r['ram_mb']:7.0f} {r['vram_mb']:8.0f} {r['carga_s']:8.1f}  {licencia}",
                  flush=True)
    print("\nEfecto sobre Whisper (parcial de 4 s, mediana de 10):")
    for etiqueta, ms in efecto_en_whisper().items():
        print(f"  {etiqueta:22s} {ms:6.0f} ms")
    print("\nMuestras (referencia / opus-mt-en-es / opus-mt-tc-big-en-es / NLLB):")
    for i in (4, 10, 14, 15, 17, 25, 29):
        print(f"\n  EN:   {origen[i]}\n  REF:  {referencias[i]}")
        for nombre in MODELOS:
            print(f"  {nombre[:12]:12s}: {muestras[nombre][i]}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--hijo":
        hijo(sys.argv[2], sys.argv[3])
    else:
        main()
