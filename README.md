# Transcriptor en vivo

Subtítulos en vivo para Windows: transcribe lo que suena en el PC (videos, transmisiones,
reuniones de Teams, Zoom o Meet) y, si quieres, también tu micrófono, separando **Yo** de
**Otros**. Funciona 100 % local con [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
en la GPU: el audio y el texto nunca salen del equipo.

- Texto provisional en gris que se corrige mientras se habla y se fija al cerrar cada frase.
- Cada frase lleva la hora en que empezó a hablarse y queda en un `.txt` en orden cronológico.
- Modo subtítulos: una franja semitransparente siempre encima que deja pasar los clics.
- Detección de voz con Silero VAD y filtros contra las alucinaciones típicas de Whisper en silencio.
- Español e inglés, vocabulario propio (nombres, jerga) y configuración que se recuerda entre sesiones.
- Traducción en vivo del inglés al español (opcional, en CPU), con tu vocabulario sin traducir.

## Requisitos

| | Probado con |
|---|---|
| Sistema | Windows 10 u 11 (WSL no sirve: no tiene acceso al audio de salida) |
| GPU | NVIDIA con al menos 2 GB de VRAM libres (probado en una RTX 3050 Laptop de 4 GB) |
| Python | 3.13 |
| Disco | ~1.6 GB para el modelo `large-v3-turbo` + ~1 GB de librerías de CUDA |

Sin GPU NVIDIA el programa intenta usar el modelo `small` en CPU, mucho más lento (ver
[Solución de problemas](#solución-de-problemas)).

## Instalación

```powershell
git clone https://github.com/Darmitho/transcriptor.git
cd transcriptor
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` incluye las DLL de CUDA (cuBLAS y cuDNN) como paquetes de pip: no hace falta
instalar el CUDA Toolkit, solo tener el driver de NVIDIA al día.

### Primera vez: descargar el modelo

El programa trabaja sin conexión (`MODO_OFFLINE = True`) para que nada salga del equipo. La
primera vez hay que permitir la descarga del modelo, una sola vez:

1. En `transcriptor.py`, cambia `MODO_OFFLINE = True` por `MODO_OFFLINE = False`.
2. Ejecuta el programa y espera a que la barra diga **Listo (large-v3-turbo, GPU)**.
   El modelo (~1.6 GB) queda en `%USERPROFILE%\.cache\huggingface`.
3. Vuelve a poner `MODO_OFFLINE = True`.

**Para la traducción (opcional):** el modelo Opus-MT en-es (Apache-2.0, 79 MB) se convierte una vez
a CTranslate2 siguiendo [tools/convertir_traduccion.md](tools/convertir_traduccion.md). Queda en
`%USERPROFILE%\.cache\transcriptor\traduccion` y luego funciona sin red.

## Uso

```powershell
venv\Scripts\python.exe transcriptor.py
```

Conviene ejecutarlo desde una terminal: ahí aparecen las métricas de cada transcripción
(tiempos, atraso, descartes), nunca el texto transcrito.

1. Espera a que el botón **▶ Iniciar** se habilite (el modelo tarda unos segundos en cargar).
2. Pulsa **▶ Iniciar**: empieza a transcribir lo que suena por la salida predeterminada de Windows.
3. Para reuniones, activa **🎤 Micrófono**: tus frases salen como **Yo** y las de los demás como
   **Otros**. Usa audífonos: con parlantes el micrófono capta también a los demás.
4. Pulsa **CC Subtítulos** (o Ctrl+Shift+S desde cualquier app) para pasar a la franja; la ventana
   se minimiza. Vuelve con el mismo atajo.

### La ventana

| Control | Qué hace |
|---|---|
| ▶ Iniciar / ■ Detener | Empieza o detiene la captura; al detener se transcribe la frase en curso |
| 🎤 Micrófono | Agrega o quita tu micrófono, también con la captura en marcha. Siempre arranca apagado |
| 🔄 Reconectar | Toma los dispositivos predeterminados actuales de Windows (tras conectar audífonos) |
| Español / English | Idioma que se habla en el audio; aplica desde la frase siguiente |
| Sin traducir / Original + traducción / Solo traducción | Traduce al español las frases finales cuando el idioma es English. La traducción aparece ~0.3 s después del texto, en cursiva debajo del original o en su lugar; la franja de subtítulos sigue la misma vista |
| CC Subtítulos | Alterna entre la ventana y la franja de subtítulos |
| ⚙ | Ajustes: letra y colores de la ventana; tamaño, líneas, opacidad y posición de la franja |
| Medidores Otros / Yo | Nivel de audio de cada fuente, para confirmar que entra sonido |
| ● Estado | Detenido, Escuchando, Transcribiendo, Atrasado (cola de más de 1 s) o Error, con el atraso |
| A− / A+ | Tamaño de la letra (9 a 36 pt) |
| Limpiar / Guardar... | Vacía la ventana (el `.txt` automático no se toca) / guarda lo que está en la ventana |
| Siempre encima | Mantiene la ventana sobre las demás |

### Atajos de teclado

| Atajo | Acción | Dónde funciona |
|---|---|---|
| **Ctrl+Shift+S** | Entrar o salir del modo subtítulos | **En todo el sistema** (también desde Teams) |
| Ctrl+Espacio | Iniciar o detener | Con la ventana enfocada |
| Ctrl+M | Activar o desactivar el micrófono | Con la ventana enfocada |
| Ctrl + `+` (o `=`) / Ctrl + `−` | Agrandar o achicar la letra | Con la ventana enfocada |

Mientras el programa está abierto, Ctrl+Shift+S le pertenece. En algunas apps (Word, VS Code)
es "Guardar como": si te molesta, cambia `atajo_subtitulos` en la configuración (por ejemplo
`"Ctrl+Alt+F9"`, o `""` para no usar atajo global). **Ajustes** muestra el atajo activo y avisa
si otra app ya lo ocupa.

## Dónde se guarda todo

| Qué | Dónde |
|---|---|
| Transcripciones (un `.txt` por sesión) | `%USERPROFILE%\Documents\Transcripciones` |
| Preferencias de la ventana | `%LOCALAPPDATA%\Transcriptor\config.json` |
| Modelo | `%USERPROFILE%\.cache\huggingface` |

Ninguna de estas carpetas está dentro de OneDrive. Si `CARPETA_SALIDA` apunta a una carpeta
de OneDrive, la consola lo advierte: las transcripciones se sincronizarían a la nube.

Cada línea del `.txt` tiene la forma `[10:32:05] Otros: texto`, con la hora en que empezó a
hablarse la frase. Con la traducción activa, cada frase lleva debajo su versión en español:
`[10:32:05] Otros (es): texto traducido`.

## Configuración

**En la ventana (⚙)** se ajusta lo de uso diario; se guarda solo en `config.json`.

**Al inicio de `transcriptor.py`** están los parámetros de fondo. Los más útiles:

| Parámetro | Valor | Para qué |
|---|---|---|
| `VOCABULARIO` | `("Claude", "CriptoBuzz", ...)` | Nombres y jerga propios: no se traducen |
| `CORRECCIONES` | `{"crypto ?bu[sz]+": "CriptoBuzz", ...}` | Arregla la escritura de nombres que Whisper no conoce ("CryptoBus" → "CriptoBuzz"), sin riesgo de inventar texto |
| `USAR_PROMPT_VOCABULARIO` | `False` | Darle el vocabulario a Whisper como contexto. Desactivado: con música o ruido lo repetía como texto inventado y no mejoró la precisión |
| `IDIOMA` / `IDIOMAS` | `"es"` / es, en | Idioma al abrir y opciones del selector |
| `CARPETA_SALIDA` | `Documents\Transcripciones` | Dónde van los `.txt` |
| `LIBERAR_VRAM_TRAS` | `300` | Segundos detenido tras los que se libera la VRAM (~900 MB); al iniciar se recarga en ~5 s sin perder audio. `None` = nunca |
| `SILENCIO_FIN` | `1.0` | Silencio que cierra una frase. Más bajo = texto final antes, pero frases más cortadas |
| `MODELO` | `"large-v3-turbo"` | `"medium"` o `"small"` si la GPU no da abasto |
| `MODO_OFFLINE` | `True` | Sin conexión a internet (ver [Primera vez](#primera-vez-descargar-el-modelo)) |
| `NO_TRADUCIR` | `("stop loss",)` | Términos que quedan en inglés al traducir, además de los de `VOCABULARIO` |
| `TRADUCCION_HILOS` | `1` | Núcleos para traducir. Con 4 la traducción llega ~0.1 s antes pero gasta ~5 veces más CPU |

## Consumo de referencia

Medido en una RTX 3050 Laptop (4 GB) con `tools\medir_recursos.py`:

| | Detenido | Una fuente hablando | Dos fuentes a la vez |
|---|---|---|---|
| CPU (% de un núcleo) | 1 | ~40 | ~75 |
| RAM | ~780 MB | ~800 MB | ~800 MB |
| VRAM | ~1 GB (~85 MB tras 5 min detenido) | ~1.1 GB | ~1.1 GB |
| Latencia: fin del habla a texto final | | ~1.7 s | ~1.9 s |

La traducción suma ~100 MB de RAM, nada de VRAM y casi nada de CPU (44 → 46 % de un núcleo con
audio en inglés), y aparece ~0.3 s después del texto original.

Si otro programa usa la GPU al mismo tiempo, la latencia puede duplicarse.

## Solución de problemas

| Síntoma | Qué hacer |
|---|---|
| No aparece texto y el medidor "Otros" no se mueve | Conectaste o cambiaste audífonos: pulsa **🔄 Reconectar**. Revisa que el audio salga por el dispositivo predeterminado de Windows |
| En reuniones no se transcribe a los demás | Teams o Zoom usan otro dispositivo que el predeterminado de Windows: deja ambos iguales |
| Tu voz aparece también como "Otros" | Estás usando parlantes: el micrófono y la salida se mezclan. Usa audífonos |
| "No se pudo cargar ningún modelo" | Falta descargar el modelo: ver [Primera vez](#primera-vez-descargar-el-modelo) |
| "No se pudo cargar el modelo de traducción" | Falta convertirlo una vez: ver [tools/convertir_traduccion.md](tools/convertir_traduccion.md) |
| No aparece la traducción | El selector de idioma debe estar en **English**: solo se traduce el audio en inglés |
| "GPU no disponible, usando CPU" | Falla CUDA (driver viejo o GPU no NVIDIA). El respaldo en CPU usa el modelo `small`, que también hay que descargar una vez con `MODO_OFFLINE = False` |
| Estado en amarillo ("Atrasado") seguido | La GPU no da abasto: cierra otros programas que la usen o prueba `MODELO = "medium"` |
| Ctrl+Shift+S no hace nada fuera de la ventana | Otra app ocupa el atajo (Ajustes dice "no disponible"): cámbialo en `atajo_subtitulos` |
| `Error in sitecustomize ... pandas` en la consola | Viene de una configuración global de Python del equipo, no de este programa; no afecta |

## Para desarrollar

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# Pruebas de la ventana: segundos, sin modelo ni audio
venv\Scripts\python.exe tests\pruebas_ventana.py

# Batería completa con el modelo real y audio sintético (~5 min). Correrla antes de cada commit:
# ningún cambio entra si sube el total de errores de palabra (hoy: 2 de 256)
venv\Scripts\python.exe tests\regresion.py

# Consumo de CPU, RAM, VRAM, GPU, RTF y latencia en 4 escenarios (~5 min, con la GPU libre)
venv\Scripts\python.exe tools\medir_recursos.py --etiqueta mi-cambio
```

- `tests\audio\`: clips sintéticos generados con las voces de Windows (`tests\generar_audio.ps1`).
- `mediciones\`: resultados de cada medición, para comparar antes y después de un cambio.
- `tools\bench_lote.py` y `tools\bench_gui.py`: microbenchmarks que justificaron decisiones
  (descartar el procesamiento en lote; elegir Tkinter frente a CustomTkinter o Qt).

## Licencia

[MIT](LICENSE)
