"""Mide el consumo del transcriptor en escenarios repetibles: ventana real + audio sintético.

Uso:  venv\\Scripts\\python.exe tools\\medir_recursos.py --etiqueta linea-base
Tarda unos 5 minutos. Guarda los resultados en mediciones\\<fecha>_<etiqueta>.json.
Conviene correrlo con la GPU libre (avisa si otro proceso la usa) y sin tocar la ventana:
arrastrarla bloquea sus eventos y la corrida se marca como no válida.
Para comparar dos versiones: --escenarios dos_fuentes, varias corridas de cada una
(entre corridas hay ±0.15 s de latencia y ±3 puntos de CPU de ruido).

Escenarios: detenido (modelo cargado) · escuchando en silencio · una fuente hablando ·
dos fuentes hablando a la vez (Yo y Otros, muy solapadas: el peor caso).
"""

import argparse
import datetime
import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import psutil
import pynvml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from comun import T, DISP, RAIZ, wav, silencio, alimentar  # noqa: E402

import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402

INTERVALO_MUESTRA = 0.5                 # Segundos entre muestras de CPU / RAM / GPU
PAUSA_ENTRE_FRASES = 1.5                # Mayor que SILENCIO_FIN: cada clip cierra su frase


class Muestreador:
    """Toma muestras de CPU, RAM y GPU del proceso en un hilo aparte."""

    def __init__(self, motor):
        self.proc = psutil.Process()
        self.motor = motor
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.muestras = []
        self.activo = threading.Event()
        threading.Thread(target=self._bucle, daemon=True).start()

    def _bucle(self):
        self.proc.cpu_percent(None)
        while True:
            time.sleep(INTERVALO_MUESTRA)
            cpu = self.proc.cpu_percent(None)
            if not self.activo.is_set():
                continue
            self.muestras.append({
                "cpu": cpu,
                "ram": self.proc.memory_info().rss / 2**20,
                "vram": pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 2**20,
                "gpu": pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu,
                # Atraso por fuente: la cola mezcla los trozos de todas
                "cola": self.motor.cola_audio.qsize() / max(1, len(self.motor.fuentes)),
            })

    def medir(self):
        self.muestras = []
        self.activo.set()

    def resultado(self):
        self.activo.clear()
        m = self.muestras
        prom = lambda k: statistics.fmean(x[k] for x in m) if m else 0.0
        maxi = lambda k: max((x[k] for x in m), default=0.0)
        return {"cpu_prom": prom("cpu"), "cpu_max": maxi("cpu"), "ram_prom": prom("ram"), "ram_max": maxi("ram"),
                "vram_max": maxi("vram"), "gpu_prom": prom("gpu"), "gpu_max": maxi("gpu"),
                "cola_max_s": maxi("cola") * T.TROZO}


def p95(valores):
    v = sorted(valores)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else 0.0


def secuencia(nombres):
    """Clips recortados separados por pausas: devuelve el audio y dónde termina cada voz."""
    trozos, fines, pos = [], [], 0
    for n in nombres:
        a = wav(n, recortar=True)
        trozos += [a, silencio(PAUSA_ENTRE_FRASES)]
        pos += len(a)
        fines.append(pos)
        pos += len(trozos[-1])
    return np.concatenate(trozos), fines


class Medicion:
    def __init__(self, app, muestreador):
        self.app, self.motor, self.mu = app, app.motor, muestreador
        self.transcripciones = []       # (fuente, final, audio_s, segundos)
        self.finales = []               # (perf_counter, fuente) cuando el texto llega a la ventana
        transcribir, agregar = self.motor._transcribir, app._agregar_final

        def _transcribir(fuente, trozos, final, voz_seg, palabras=False):
            t0 = time.perf_counter()
            r = transcribir(fuente, trozos, final, voz_seg, palabras)
            self.transcripciones.append((fuente.nombre, final, sum(len(x) for x in trozos) / fuente.tasa,
                                         time.perf_counter() - t0))
            return r

        def _agregar_final(valor):
            self.finales.append((time.perf_counter(), valor[0]))
            return agregar(valor)

        self.motor._transcribir = _transcribir
        app._agregar_final = _agregar_final

    def escenario(self, nombre, pistas=None, fines=None, segundos=None):
        print(f"  midiendo: {nombre}...", flush=True)
        self.transcripciones, self.finales = [], []
        self.mu.medir()
        enviados = {}
        t_inicio = time.perf_counter()
        esperado = segundos or 0
        if pistas is None:
            time.sleep(segundos)
        else:
            m = self.motor
            m.stop, m.cola_audio = threading.Event(), queue.Queue()
            fuentes = {n: T.Fuente(n, DISP, m.cola_audio) for n in pistas}
            m.fuentes = list(fuentes.values())
            m.hilo = threading.Thread(target=m._bucle, args=(m.stop, m.cola_audio, m.fuentes), daemon=True)
            # Medidores e indicador funcionando, como en uso real. Igual que el botón Iniciar:
            # solo después de que la ventana procesó el "detenido" del escenario anterior
            while not self.app.cola_ui.empty():
                time.sleep(0.05)
            self.app.activo = True
            m.hilo.start()
            esperado = max(len(a) for a in pistas.values()) / 48000 + 3.0
            enviados = alimentar(pistas, fuentes, m.cola_audio, fines_voz=fines)
            time.sleep(3.0)             # Margen para que lleguen los últimos finales
            self.app.activo = False         # Como el botón Detener: primero la ventana, luego el motor
            m.stop.set()
            m.hilo.join(120)
        r = self.mu.resultado()
        for tipo, es_final in (("parcial", False), ("final", True)):
            t = [x for x in self.transcripciones if x[1] == es_final]
            rtf = [s / a for _, _, a, s in t if a > 0]
            r[f"{tipo}_n"] = len(t)
            r[f"{tipo}_rtf_prom"] = statistics.fmean(rtf) if rtf else 0.0
            r[f"{tipo}_seg_p95"] = p95([s for *_, s in t])
        latencias = []
        for fuente, tiempos in enviados.items():
            llegadas = sorted(t for t, f in self.finales if f == fuente)
            for t_fin in tiempos:
                siguiente = next((t for t in llegadas if t >= t_fin), None)
                if siguiente is not None:
                    latencias.append(siguiente - t_fin)
        r["latencia_n"] = len(latencias)
        r["latencia_prom"] = statistics.fmean(latencias) if latencias else 0.0
        r["latencia_p95"] = p95(latencias)
        r["latencia_max"] = max(latencias, default=0.0)
        # Si la ventana deja de procesar eventos (p. ej. mientras se arrastra), los finales
        # llegan tarde o nunca y la corrida no sirve: se marca para descartarla
        r["duracion_s"] = time.perf_counter() - t_inicio
        r["valida"] = r["duracion_s"] <= esperado * 1.3 + 2 and (not fines or r["latencia_n"] > 0)
        if not r["valida"]:
            print(f"  AVISO: corrida no válida ({r['duracion_s']:.0f} s, esperado ~{esperado:.0f} s, "
                  f"{r['latencia_n']} latencias): no tocar la ventana mientras se mide")
        return r


def otros_procesos_gpu():
    try:
        salida = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10).stdout
        return [int(p) for p in salida.split() if p.strip().isdigit() and int(p) != os.getpid()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", default="medicion")
    ap.add_argument("--escenarios", default="detenido,silencio,una_fuente,dos_fuentes",
                    help="Lista separada por comas: para repetir solo los que interesan")
    args = ap.parse_args()

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    otros = otros_procesos_gpu()
    if otros:
        print(f"AVISO: otros procesos usan la GPU {otros}: los tiempos y la VRAM no serán representativos")
    base = {"vram": pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20, "ram": psutil.Process().memory_info().rss / 2**20}

    carpeta = tempfile.mkdtemp(prefix="medicion_")         # No tocar transcripciones ni configuración reales
    T.CARPETA_SALIDA, T.CONFIG_ARCHIVO = carpeta, os.path.join(carpeta, "config.json")
    listo = threading.Event()
    cargar = T.Motor.cargar_modelo

    def cargar_y_avisar(self):
        cargar(self)
        listo.set()

    T.Motor.cargar_modelo = cargar_y_avisar
    root = tk.Tk()
    app = T.App(root)
    mu = Muestreador(app.motor)
    resultados, terminado = {}, threading.Event()

    def guion():
        try:
            listo.wait()
            time.sleep(2)
            med = Medicion(app, mu)
            otros_a, fines_a = secuencia(["frase", "v1", "v2", "gracias", "continuo", "v3",
                                          "parte0", "parte1", "parte2", "v4", "v5"])
            yo_b, fines_b = secuencia(["v5", "parte3", "v4", "frase", "parte4", "v1", "v3", "continuo", "v2"])
            yo_b = np.concatenate([silencio(1.5), yo_b])
            fines_b = [f + int(1.5 * 48000) for f in fines_b]
            pedidos = args.escenarios.split(",")
            guiones = {
                "detenido": lambda: med.escenario("detenido (modelo cargado)", segundos=15),
                "silencio": lambda: med.escenario("escuchando en silencio", pistas={"Otros": silencio(20)}),
                "una_fuente": lambda: med.escenario("una fuente hablando", pistas={"Otros": otros_a},
                                                    fines={"Otros": fines_a}),
                "dos_fuentes": lambda: med.escenario("dos fuentes hablando", pistas={"Otros": otros_a, "Yo": yo_b},
                                                     fines={"Otros": fines_a, "Yo": fines_b}),
            }
            for nombre in pedidos:
                resultados[nombre] = guiones[nombre]()
        finally:
            terminado.set()

    def revisar_fin():
        if terminado.is_set():
            app._destruir()
        else:
            root.after(200, revisar_fin)

    threading.Thread(target=guion, daemon=True).start()
    root.after(200, revisar_fin)
    root.mainloop()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True,
                            text=True).stdout.strip()
    sucio = bool(subprocess.run(["git", "status", "--porcelain", "transcriptor.py"], cwd=RAIZ,
                                capture_output=True, text=True).stdout.strip())
    datos = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"), "etiqueta": args.etiqueta,
             "commit": commit + ("+cambios" if sucio else ""), "otros_procesos_gpu": otros,
             "nucleos": psutil.cpu_count(), "base_antes_de_cargar": base, "escenarios": resultados}
    os.makedirs(os.path.join(RAIZ, "mediciones"), exist_ok=True)
    ruta = os.path.join(RAIZ, "mediciones", f"{datetime.datetime.now():%Y%m%d_%H%M}_{args.etiqueta}.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    filas = [
        ("CPU % de un núcleo (prom / máx)", lambda r: f"{r['cpu_prom']:.0f} / {r['cpu_max']:.0f}"),
        ("RAM MB (prom / máx)", lambda r: f"{r['ram_prom']:.0f} / {r['ram_max']:.0f}"),
        ("VRAM sobre la base, MB (máx)", lambda r: f"{r['vram_max'] - base['vram']:.0f}"),
        ("Uso de GPU % (prom / máx)", lambda r: f"{r['gpu_prom']:.0f} / {r['gpu_max']:.0f}"),
        ("Parciales: n, RTF prom", lambda r: f"{r['parcial_n']}, {r['parcial_rtf_prom']:.2f}"),
        ("Finales: n, RTF prom", lambda r: f"{r['final_n']}, {r['final_rtf_prom']:.2f}"),
        ("Latencia fin de habla -> texto (prom / p95 / máx s)",
         lambda r: f"{r['latencia_prom']:.2f} / {r['latencia_p95']:.2f} / {r['latencia_max']:.2f}" if r["latencia_n"] else "-"),
        ("Atraso máx de la cola por fuente (s)", lambda r: f"{r['cola_max_s']:.1f}"),
    ]
    cols = [c for c in ("detenido", "silencio", "una_fuente", "dos_fuentes") if c in resultados]
    print(f"\n{'':52s}" + "".join(f"{c:>20s}" for c in cols))
    for etiqueta, f in filas:
        print(f"{etiqueta:52s}" + "".join(f"{f(resultados[c]):>20s}" for c in cols))
    print(f"\nbase antes de cargar el modelo: RAM {base['ram']:.0f} MB, VRAM usada por todo el equipo {base['vram']:.0f} MB")
    print("guardado en", ruta)


if __name__ == "__main__":
    main()
