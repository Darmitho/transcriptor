"""Prueba de viabilidad: capturar solo el audio de una aplicación (o todo menos una).

Uso:  venv\\Scripts\\python.exe tools\\poc_captura_app.py
Lanza dos procesos que reproducen a la vez tonos distintos (A: 731 Hz, B: 1373 Hz) a volumen
bajo durante unos segundos (se oyen por la salida predeterminada) y captura de cuatro formas:
solo A, solo B, todo menos A y todo menos B. Un tono "está" si sobresale >= 12 dB de su
entorno en el espectro y "no está" si queda bajo 10 dB. No son notas musicales, para que otro
audio que suene en el equipo (un video, música) no se confunda con los tonos.
"""

import os
import subprocess
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

DURACION_TONOS = 10.0
AMPLITUD = 0.05                         # Bajo: la prueba no tiene que molestar
TONO_A, TONO_B = 731.0, 1373.0
PRESENTE_DB = 12.0                      # Con otro audio sonando fuerte, el tono incluido queda en ~15-20 dB
AUSENTE_DB = 10.0                       # El excluido queda en el ruido del espectro (~4-8 dB)


def reproducir_tono(frecuencia, segundos):
    """Proceso hijo: un tono continuo por la salida predeterminada."""
    import pyaudiowpatch as pyaudio
    pa = pyaudio.PyAudio()
    tasa = 48000
    s = pa.open(format=pyaudio.paFloat32, channels=1, rate=tasa, output=True)
    t = np.arange(int(tasa * segundos)) / tasa
    s.write((AMPLITUD * np.sin(2 * np.pi * frecuencia * t)).astype(np.float32).tobytes())
    s.close()
    pa.terminate()


def prominencia_db(audio, frecuencia, tasa):
    """Cuánto sobresale el pico en la frecuencia respecto de su entorno (±80 Hz), en dB."""
    if not len(audio):
        return 0.0
    espectro = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) + 1e-12
    f = np.fft.rfftfreq(len(audio), 1 / tasa)
    pico = espectro[(f > frecuencia - 3) & (f < frecuencia + 3)].max()
    entorno = np.median(espectro[(np.abs(f - frecuencia) > 10) & (np.abs(f - frecuencia) < 80)])
    return float(20 * np.log10(pico / entorno))


def capturar(pid, excluir, segundos):
    from captura_app import CapturaApp, TASA_CAPTURA
    bloques = []
    c = CapturaApp(pid, bloques.append, excluir=excluir).iniciar()
    time.sleep(segundos)
    c.detener()
    audio = np.concatenate(bloques) if bloques else np.zeros(0, np.float32)
    return audio, TASA_CAPTURA


def main():
    a = subprocess.Popen([sys.executable, __file__, "--tono", str(TONO_A)])
    b = subprocess.Popen([sys.executable, __file__, "--tono", str(TONO_B)])
    time.sleep(1.5)                     # Que ambos estén sonando
    print(f"Tonos sonando: A (PID {a.pid}) {TONO_A:.0f} Hz, B (PID {b.pid}) {TONO_B:.0f} Hz")
    print()
    print(f"{'captura':14s} {'segundos':>8s} {'nivel RMS':>10s} {'A (dB)':>8s} {'B (dB)':>8s}  resultado")
    casos = [("solo A", a.pid, False, (True, False)), ("solo B", b.pid, False, (False, True)),
             ("todo menos A", a.pid, True, (False, True)), ("todo menos B", b.pid, True, (True, False))]
    ok = True
    for nombre, pid, excluir, esperado in casos:
        audio, tasa = capturar(pid, excluir, 1.5)
        pa, pb = prominencia_db(audio, TONO_A, tasa), prominencia_db(audio, TONO_B, tasa)
        cumple = lambda p, esta: p >= PRESENTE_DB if esta else p < AUSENTE_DB
        bien = cumple(pa, esperado[0]) and cumple(pb, esperado[1]) and len(audio) > 0
        ok &= bien
        nivel = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        print(f"{nombre:14s} {len(audio) / tasa:8.2f} {nivel:10.4f} {pa:8.1f} {pb:8.1f}  {'OK' if bien else 'FALLO'}")
    a.wait()
    b.wait()
    print()
    print("VIABLE" if ok else "NO FUNCIONÓ COMO SE ESPERABA")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--tono":
        reproducir_tono(float(sys.argv[2]), DURACION_TONOS)
    else:
        main()
