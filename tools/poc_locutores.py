"""Prueba de concepto: distinguir locutores dentro de "Otros" con huellas de voz (sin integrar).

Uso:  venv\\Scripts\\python.exe tools\\poc_locutores.py
Necesita sherpa-onnx y, en CARPETA (ver más abajo), los modelos de huella de voz y reuniones del
corpus AMI (CC-BY-4.0) con su anotación RTTM. Todo corre en CPU, sin PyTorch.

Método, igual a como funcionaría en la app:
1. Frases: el mismo corte que la app (Fuente: VAD de Silero, SILENCIO_FIN, PAUSA_CORTE...).
2. Verdad de cada frase: el locutor que más habla en ella según la anotación.
3. Huella de voz (embedding) por frase y agrupación en línea por similitud coseno con los
   centroides de la sesión: se asigna al más parecido si supera el umbral; si no, "Persona N".
   Frases con poca voz (< VOZ_MIN_NUEVA): solo se asignan si el parecido es alto y nunca crean
   una persona nueva; si no, quedan como "Otros".
4. El umbral se ajusta en una reunión y se evalúa en otra.
"""

import itertools
import os
import sys
import time
import wave

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from comun import T  # noqa: E402

CARPETA = os.path.join(os.path.expanduser("~"), ".cache", "transcriptor", "locutores")
MODELOS = {
    "WeSpeaker ResNet34": "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "3D-Speaker CAM++ zh/en": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "3D-Speaker ERes2Net en": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "NeMo TitaNet small": "nemo_en_titanet_small.onnx",
}
AJUSTE, EVALUACION = "ES2004a", "IS1009a"
VOZ_MIN_NUEVA = 1.5                     # Segundos de voz para crear una persona nueva
MARGEN_CORTAS = 0.10                    # Las frases cortas necesitan umbral + este margen
SR = 16000


def leer_wav(ruta):
    with wave.open(ruta) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return a.reshape(-1, w.getnchannels()).mean(axis=1)


def leer_rttm(ruta):
    return [(float(c[3]), float(c[3]) + float(c[4]), c[7]) for c in (l.split() for l in open(ruta)) if c]


def frases_como_la_app(audio):
    """Corta el audio en frases con la misma lógica que la app (Fuente)."""
    f = T.Fuente("Otros", {"maxInputChannels": 1, "defaultSampleRate": float(SR), "index": 0}, None)
    paso, frases = int(SR * T.TROZO), []
    for i in range(0, len(audio), paso):
        f.agregar(audio[i:i + paso], i / SR)
        if f.motivo_cierre(incluir_sin_audio=False):
            voz = sum(f.voz)
            if voz >= T.VOZ_MIN:
                ini, fin = T.Motor._recortar(None, f.voz)
                frases.append({"inicio": f.inicios[ini], "audio": np.concatenate(f.trozos[ini:fin]), "voz": voz})
            f.reiniciar_frase()
    for fr in frases:
        fr["fin"] = fr["inicio"] + len(fr["audio"]) / SR
    return frases


def etiquetar(frases, rttm):
    """Locutor real de cada frase (el que más habla) y pureza (qué fracción del habla es suya)."""
    for fr in frases:
        tiempo = fr["tiempos"] = {}
        for a, b, quien in rttm:
            solape = min(b, fr["fin"]) - max(a, fr["inicio"])
            if solape > 0:
                tiempo[quien] = tiempo.get(quien, 0) + solape
        total = sum(tiempo.values())
        fr["real"] = max(tiempo, key=tiempo.get) if tiempo else None
        fr["pureza"] = tiempo[fr["real"]] / total if total else 0
    return [fr for fr in frases if fr["real"]]


def huellas(frases, modelo):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=os.path.join(CARPETA, MODELOS[modelo]), num_threads=1)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    t0 = time.process_time()
    for fr in frases:
        s = ext.create_stream()
        s.accept_waveform(SR, fr["audio"])
        s.input_finished()
        e = np.array(ext.compute(s), dtype=np.float32)
        fr["huella"] = e / np.linalg.norm(e)
    cpu = time.process_time() - t0
    audio_s = sum(len(fr["audio"]) for fr in frases) / SR
    return 1000 * cpu / len(frases), 1000 * cpu / audio_s


def agrupar(frases, umbral):
    """Agrupación en línea por similitud coseno con los centroides. -1 = sin asignar ("Otros")."""
    sumas, asignado = [], []
    for fr in frases:
        sims = [float(fr["huella"] @ (s / np.linalg.norm(s))) for s in sumas]
        mejor = int(np.argmax(sims)) if sims else -1
        parecido = sims[mejor] if sims else -1.0
        if fr["voz"] >= VOZ_MIN_NUEVA:
            if parecido >= umbral:
                sumas[mejor] += fr["huella"]
                asignado.append(mejor)
            else:
                sumas.append(fr["huella"].copy())
                asignado.append(len(sumas) - 1)
        else:
            asignado.append(mejor if parecido >= umbral + MARGEN_CORTAS else -1)
    return asignado


def evaluar(frases, asignado):
    """Acierto ponderado por voz con el mejor emparejamiento uno a uno grupo <-> persona real
    (dividir a una persona en dos grupos o juntar a dos cuenta como error)."""
    reales = sorted({fr["real"] for fr in frases})
    grupos = sorted({a for a in asignado if a >= 0})
    voz_total = sum(fr["voz"] for fr in frases)
    voz_asignada = sum(fr["voz"] for fr, a in zip(frases, asignado) if a >= 0)
    if not grupos:
        return 0.0, 0.0, 0
    m = np.zeros((len(grupos), len(reales)))
    for fr, a in zip(frases, asignado):
        if a >= 0:
            m[grupos.index(a), reales.index(fr["real"])] += fr["voz"]
    filas, cols = linear_sum_assignment(-m)
    acierto = m[filas, cols].sum() / voz_asignada
    return acierto, voz_asignada / voz_total, len(grupos)


def tiempo_bien_atribuido(frases, asignado):
    """Fracción del tiempo de habla real (según la anotación) que queda con la etiqueta de la
    persona correcta. Una frase con dos voces recibe una sola etiqueta: la otra voz cuenta mal,
    y lo que queda como "Otros" también."""
    reales = sorted({fr["real"] for fr in frases})
    grupos = sorted({a for a in asignado if a >= 0})
    m = np.zeros((len(grupos), len(reales)))
    for fr, a in zip(frases, asignado):
        if a >= 0:
            for quien, t in fr["tiempos"].items():
                if quien in reales:
                    m[grupos.index(a), reales.index(quien)] += t
    filas, cols = linear_sum_assignment(-m)
    total = sum(t for fr in frases for t in fr["tiempos"].values())
    return m[filas, cols].sum() / total


def main():
    import logging
    logging.getLogger("transcriptor").setLevel(logging.WARNING)
    datos = {}
    for reunion in (AJUSTE, EVALUACION):
        audio = leer_wav(os.path.join(CARPETA, "ami", reunion + ".wav"))
        frases = etiquetar(frases_como_la_app(audio), leer_rttm(os.path.join(CARPETA, "ami", reunion + ".rttm")))
        datos[reunion] = frases
        puras = sum(fr["pureza"] >= 0.9 for fr in frases)
        print(f"{reunion}: {len(audio) / SR / 60:.1f} min, {len(frases)} frases, {len({f['real'] for f in frases})} "
              f"personas, {puras} frases de una sola voz (>= 90 %), voz por frase mediana "
              f"{np.median([f['voz'] for f in frases]):.1f} s")

    print(f"\nUmbral ajustado en {AJUSTE}, evaluado en {EVALUACION} (4 personas):")
    print(f"{'modelo':24s} {'ms/frase':>8s} {'ms CPU/s':>9s} {'umbral':>7s} {'acierto':>8s} {'asignado':>9s} {'grupos':>7s} {'tiempo bien':>12s}")
    mejores = {}
    for modelo in MODELOS:
        costo = None
        for reunion in datos:
            c = huellas(datos[reunion], modelo)
            costo = costo or c
        umbral = max(np.arange(0.20, 0.86, 0.025),
                     key=lambda u: (lambda r: r[0] * r[1])(evaluar(datos[AJUSTE], agrupar(datos[AJUSTE], u))))
        asig = agrupar(datos[EVALUACION], umbral)
        acierto, cobertura, grupos = evaluar(datos[EVALUACION], asig)
        bien = tiempo_bien_atribuido(datos[EVALUACION], asig)
        mejores[modelo] = (umbral, acierto)
        print(f"{modelo:24s} {costo[0]:8.0f} {costo[1]:9.1f} {umbral:7.3f} {100 * acierto:7.0f}% "
              f"{100 * cobertura:8.0f}% {grupos:4d}/4 {100 * bien:11.0f}%", flush=True)
        for fr in itertools.chain(*datos.values()):
            fr[f"huella_{modelo}"] = fr["huella"]

    modelo = max(mejores, key=lambda k: mejores[k][1])
    umbral = mejores[modelo][0]
    print(f"\nCon {modelo} (umbral {umbral:.3f}), reuniones de 2 y 3 personas (subconjuntos reales):")
    for reunion, frases in datos.items():
        for fr in frases:
            fr["huella"] = fr[f"huella_{modelo}"]
        personas = sorted({fr["real"] for fr in frases})
        for k in (2, 3, 4):
            res = []
            for grupo in itertools.combinations(personas, k):
                sub = [fr for fr in frases if fr["real"] in grupo]
                a = agrupar(sub, umbral)
                res.append(evaluar(sub, a) + (tiempo_bien_atribuido(sub, a),))
            print(f"  {reunion} {k} personas ({len(res)} combinaciones): acierto {100 * np.mean([r[0] for r in res]):.0f} % "
                  f"(peor {100 * min(r[0] for r in res):.0f} %), asignado {100 * np.mean([r[1] for r in res]):.0f} %, "
                  f"grupos {np.mean([r[2] for r in res]):.1f} para {k} | tiempo bien atribuido {100 * np.mean([r[3] for r in res]):.0f} %")

    # Dónde falla: por duración de la frase y por mezcla de voces
    frases = datos[EVALUACION]
    asignado = agrupar(frases, umbral)
    reales = sorted({f["real"] for f in frases})
    grupos = sorted({a for a in asignado if a >= 0})
    m = np.zeros((len(grupos), len(reales)))
    for fr, a in zip(frases, asignado):
        if a >= 0:
            m[grupos.index(a), reales.index(fr["real"])] += fr["voz"]
    filas, cols = linear_sum_assignment(-m)
    mapa = {grupos[f]: reales[c] for f, c in zip(filas, cols)}
    print(f"\nDónde falla ({EVALUACION}):")
    for nombre, cond in (("voz < 1.5 s", lambda f: f["voz"] < 1.5), ("voz 1.5-4 s", lambda f: 1.5 <= f["voz"] < 4),
                         ("voz >= 4 s", lambda f: f["voz"] >= 4), ("una sola voz (>= 90 %)", lambda f: f["pureza"] >= 0.9),
                         ("voces mezcladas (< 90 %)", lambda f: f["pureza"] < 0.9)):
        sel = [(f, a) for f, a in zip(frases, asignado) if cond(f)]
        asig = [(f, a) for f, a in sel if a >= 0]
        bien = sum(f["voz"] for f, a in asig if mapa.get(a) == f["real"])
        print(f"  {nombre:26s} {len(sel):4d} frases | asignadas {len(asig):4d} | acierto "
              f"{100 * bien / max(1e-9, sum(f['voz'] for f, _ in asig)):5.0f} %")


if __name__ == "__main__":
    main()
