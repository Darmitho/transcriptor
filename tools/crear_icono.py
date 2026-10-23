"""Genera assets\\transcriptor.ico (un micrófono) sin dependencias extra: numpy + zlib.

Uso:  venv\\Scripts\\python.exe tools\\crear_icono.py
Dibuja con funciones de distancia y supermuestreo (bordes suaves) y guarda cada tamaño como
PNG dentro del .ico, formato que Windows admite desde Vista.
"""

import os
import struct
import zlib

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "assets", "transcriptor.ico")
TAMANOS = (16, 24, 32, 48, 64, 128, 256)
FONDO, ACENTO = (0x1E, 0x1F, 0x22), (0x6E, 0xA8, 0xFE)   # Los mismos colores de la ventana


def _segmento(x, y, x0, y0, x1, y1):
    """Distancia de cada punto al segmento (x0, y0)-(x1, y1)."""
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))


def _caja(x, y, cx, cy, mx, my, radio):
    """Distancia con signo a un rectángulo de esquinas redondeadas (negativa adentro)."""
    qx, qy = np.abs(x - cx) - mx + radio, np.abs(y - cy) - my + radio
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radio


def dibujar(tamano, sobremuestreo=4):
    n = tamano * sobremuestreo
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    fondo = _caja(x, y, 0.5, 0.5, 0.46, 0.46, 0.2) <= 0
    capsula = _segmento(x, y, 0.5, 0.25, 0.5, 0.48) <= 0.13
    anillo = (np.abs(np.hypot(x - 0.5, y - 0.48) - 0.23) <= 0.035) & (y >= 0.48)
    pie = _caja(x, y, 0.5, 0.77, 0.03, 0.06, 0.01) <= 0
    base = _caja(x, y, 0.5, 0.84, 0.15, 0.03, 0.03) <= 0
    microfono = capsula | anillo | pie | base
    rgba = np.zeros((n, n, 4), np.float32)
    rgba[fondo] = (*FONDO, 255)
    rgba[microfono & fondo] = (*ACENTO, 255)
    # Promedio de cada bloque de sobremuestreo: bordes suaves al reducir
    rgba = rgba.reshape(tamano, sobremuestreo, tamano, sobremuestreo, 4).mean(axis=(1, 3))
    return np.round(rgba).astype(np.uint8)


def png(rgba):
    alto, ancho = rgba.shape[:2]
    crudo = b"".join(b"\x00" + rgba[f].tobytes() for f in range(alto))   # Filtro 0 por fila

    def bloque(tipo, datos):
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", zlib.crc32(tipo + datos))

    return (b"\x89PNG\r\n\x1a\n" + bloque(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 6, 0, 0, 0))
            + bloque(b"IDAT", zlib.compress(crudo, 9)) + bloque(b"IEND", b""))


def ico(imagenes):
    cabecera = struct.pack("<HHH", 0, 1, len(imagenes))
    desplazamiento = 6 + 16 * len(imagenes)
    entradas, datos = b"", b""
    for tamano, contenido in imagenes:
        lado = tamano if tamano < 256 else 0       # 0 significa 256 en el formato .ico
        entradas += struct.pack("<BBBBHHII", lado, lado, 0, 0, 1, 32, len(contenido), desplazamiento + len(datos))
        datos += contenido
    return cabecera + entradas + datos


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as f:
        f.write(ico([(t, png(dibujar(t))) for t in TAMANOS]))
    with open(os.path.join(os.path.dirname(SALIDA), "transcriptor_256.png"), "wb") as f:
        f.write(png(dibujar(256)))              # Para verlo sin abrir el .ico
    print("guardado:", SALIDA, os.path.getsize(SALIDA), "bytes")
