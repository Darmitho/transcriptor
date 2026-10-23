"""Captura del audio de una sola aplicación (o de todo menos una) en Windows.

Usa la API de "application loopback" de WASAPI (Windows 10 compilación 20348 o posterior):
ActivateAudioInterfaceAsync sobre el dispositivo virtual VAD\\Process_Loopback, con el PID
del proceso. Incluye los procesos hijos (Chrome y Discord reproducen desde uno hijo).
Basado en el ejemplo oficial de Microsoft "ApplicationLoopback".

CapturaApp(pid, excluir, al_recibir).iniciar() entrega bloques float32 mono a TASA_CAPTURA
llamando a al_recibir(bloque) desde su propio hilo; detener() la corta.
apps_con_audio() lista los programas con una sesión de audio en la salida predeterminada y
pid_de_app(nombre) encuentra el proceso raíz de un programa, cuyo árbol se captura.
"""

import ctypes
import os
import threading
import time
from ctypes import wintypes

import comtypes
import numpy as np
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

TASA_CAPTURA = 48000
CANALES = 2
PERIODO_ENTREGA = 0.05                  # Cada cuánto se convierte y entrega lo capturado
BUFFER_100NS = 200_000                  # 20 ms pedidos (Windows usa 10 ms igual en este modo)

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
ACTIVACION_LOOPBACK_PROCESO = 1          # AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
MODO_INCLUIR, MODO_EXCLUIR = 0, 1        # PROCESS_LOOPBACK_MODE_(INCLUDE|EXCLUDE)_TARGET_PROCESS_TREE
VT_BLOB = 65
DISPOSITIVO_LOOPBACK_PROCESO = "VAD\\Process_Loopback"


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class PARAMS_ACTIVACION(ctypes.Structure):
    """AUDIOCLIENT_ACTIVATION_PARAMS con AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS."""
    _fields_ = [("ActivationType", ctypes.c_int), ("TargetProcessId", wintypes.DWORD),
                ("ProcessLoopbackMode", ctypes.c_int)]


class PROPVARIANT_BLOB(ctypes.Structure):
    """PROPVARIANT con VT_BLOB (24 bytes en 64 bits, igual que el original)."""
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("cbSize", ctypes.c_ulong), ("pBlobData", ctypes.c_void_p)]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer", (["out"], ctypes.POINTER(ctypes.POINTER(ctypes.c_byte)), "datos"),
                  (["out"], ctypes.POINTER(ctypes.c_uint32), "cuadros"), (["out"], ctypes.POINTER(wintypes.DWORD), "flags"),
                  (["in"], ctypes.c_void_p, "posicion"), (["in"], ctypes.c_void_p, "qpc")),
        COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], ctypes.c_uint32, "cuadros")),
        COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], ctypes.POINTER(ctypes.c_uint32), "cuadros")),
    ]


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize", (["in"], ctypes.c_int, "modo"), (["in"], wintypes.DWORD, "flags"),
                  (["in"], ctypes.c_longlong, "duracion"), (["in"], ctypes.c_longlong, "periodo"),
                  (["in"], ctypes.POINTER(WAVEFORMATEX), "formato"), (["in"], ctypes.c_void_p, "sesion")),
        COMMETHOD([], HRESULT, "GetBufferSize", (["out"], ctypes.POINTER(ctypes.c_uint32), "cuadros")),
        COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], ctypes.POINTER(ctypes.c_longlong), "latencia")),
        COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], ctypes.POINTER(ctypes.c_uint32), "relleno")),
        COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], ctypes.c_int, "modo"),
                  (["in"], ctypes.POINTER(WAVEFORMATEX), "formato"), (["in"], ctypes.c_void_p, "parecido")),
        COMMETHOD([], HRESULT, "GetMixFormat", (["in"], ctypes.c_void_p, "formato")),
        COMMETHOD([], HRESULT, "GetDevicePeriod", (["in"], ctypes.c_void_p, "defecto"), (["in"], ctypes.c_void_p, "minimo")),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle", (["in"], wintypes.HANDLE, "evento")),
        COMMETHOD([], HRESULT, "GetService", (["in"], ctypes.POINTER(GUID), "iid"),
                  (["out"], ctypes.POINTER(ctypes.POINTER(IUnknown)), "servicio")),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetActivateResult", (["out"], ctypes.POINTER(HRESULT), "resultado"),
                  (["out"], ctypes.POINTER(ctypes.POINTER(IUnknown)), "interfaz")),
    ]


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted",
                  (["in"], ctypes.POINTER(IActivateAudioInterfaceAsyncOperation), "operacion")),
    ]


class IAgileObject(IUnknown):
    """Marca de objeto ágil: Windows exige que el manejador pueda llamarse desde cualquier hilo."""
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class _AlCompletar(comtypes.COMObject):
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.listo = threading.Event()

    def ActivateCompleted(self, operacion):
        self.listo.set()
        return 0


def _activar(pid, excluir):
    """Cliente de audio en modo loopback de un árbol de procesos (o de todo menos él)."""
    params = PARAMS_ACTIVACION(ACTIVACION_LOOPBACK_PROCESO, pid, MODO_EXCLUIR if excluir else MODO_INCLUIR)
    variante = PROPVARIANT_BLOB(vt=VT_BLOB, cbSize=ctypes.sizeof(params),
                                pBlobData=ctypes.cast(ctypes.pointer(params), ctypes.c_void_p))
    manejador = _AlCompletar()
    operacion = ctypes.POINTER(IActivateAudioInterfaceAsyncOperation)()
    activar = ctypes.WinDLL("Mmdevapi").ActivateAudioInterfaceAsync
    activar.restype = HRESULT
    activar.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(GUID), ctypes.POINTER(PROPVARIANT_BLOB),
                        ctypes.POINTER(IActivateAudioInterfaceCompletionHandler),
                        ctypes.POINTER(ctypes.POINTER(IActivateAudioInterfaceAsyncOperation))]
    activar(DISPOSITIVO_LOOPBACK_PROCESO, ctypes.byref(IAudioClient._iid_), ctypes.byref(variante),
            manejador.QueryInterface(IActivateAudioInterfaceCompletionHandler), ctypes.byref(operacion))
    if not manejador.listo.wait(5):
        raise RuntimeError("Windows no respondió a la activación de la captura por aplicación")
    resultado, interfaz = operacion.GetActivateResult()
    if resultado != 0:
        raise OSError(resultado, "No se pudo activar la captura por aplicación (¿Windows 10 20348 o posterior?)")
    return interfaz.QueryInterface(IAudioClient)


class CapturaApp(threading.Thread):
    """Hilo que captura el audio de un árbol de procesos y lo entrega como float32 mono."""

    def __init__(self, pid, al_recibir, excluir=False):
        super().__init__(daemon=True)
        self.pid, self.al_recibir, self.excluir = pid, al_recibir, excluir
        self.seguir = threading.Event()
        self.lista = threading.Event()      # Se marca al empezar a capturar (o al fallar)
        self.error = None

    def iniciar(self):
        self.seguir.set()
        self.start()
        self.lista.wait(10)
        if self.error:
            raise self.error
        return self

    def detener(self):
        self.seguir.clear()
        self.join(5)

    def run(self):
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        kernel32 = ctypes.windll.kernel32
        evento = kernel32.CreateEventW(None, False, False, None)
        try:
            cliente = _activar(self.pid, self.excluir)
            formato = WAVEFORMATEX(1, CANALES, TASA_CAPTURA, TASA_CAPTURA * CANALES * 2, CANALES * 2, 16, 0)
            flags = (AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                     | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY)
            cliente.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, BUFFER_100NS, 0, ctypes.byref(formato), None)
            cliente.SetEventHandle(evento)
            captura = cliente.GetService(ctypes.byref(IAudioCaptureClient._iid_)).QueryInterface(IAudioCaptureClient)
            cliente.Start()
        except Exception as e:
            self.error = e
            self.lista.set()
            kernel32.CloseHandle(evento)
            return
        self.lista.set()
        try:
            # Windows usa un búfer de 10 ms aunque se pida más: cada paquete se lee apenas llega
            # (esperar 40 ms perdía ~25 % del audio). Lo costoso (numpy y entregar el bloque) se
            # hace cada PERIODO_ENTREGA con los bytes acumulados, no por cada paquete.
            crudos, ultima = [], time.perf_counter()
            while self.seguir.is_set():
                if kernel32.WaitForSingleObject(evento, 200) != 0:   # 0 = WAIT_OBJECT_0
                    continue
                while captura.GetNextPacketSize():
                    datos, cuadros, flags_buf = captura.GetBuffer(None, None)
                    n = cuadros * CANALES * 2
                    crudos.append(bytes(n) if flags_buf & AUDCLNT_BUFFERFLAGS_SILENT else ctypes.string_at(datos, n))
                    captura.ReleaseBuffer(cuadros)
                if crudos and time.perf_counter() - ultima >= PERIODO_ENTREGA:
                    pcm = np.frombuffer(b"".join(crudos), np.int16).reshape(-1, CANALES)
                    self.al_recibir(pcm.mean(axis=1).astype(np.float32) / 32768)
                    crudos, ultima = [], time.perf_counter()
        finally:
            cliente.Stop()
            kernel32.CloseHandle(evento)


# ---------- Qué programas están sonando ----------
class IMMDevice(IUnknown):
    _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Activate", (["in"], ctypes.POINTER(GUID), "iid"), (["in"], wintypes.DWORD, "contexto"),
                  (["in"], ctypes.c_void_p, "params"), (["out"], ctypes.POINTER(ctypes.POINTER(IUnknown)), "interfaz")),
    ]


class IMMDeviceEnumerator(IUnknown):
    _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
    _methods_ = [
        COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], ctypes.c_int, "flujo"), (["in"], wintypes.DWORD, "estado"),
                  (["in"], ctypes.c_void_p, "lista")),
        COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], ctypes.c_int, "flujo"), (["in"], ctypes.c_int, "rol"),
                  (["out"], ctypes.POINTER(ctypes.POINTER(IMMDevice)), "dispositivo")),
    ]


class IAudioSessionControl2(IUnknown):
    _iid_ = GUID("{bfb7ff88-7239-4fc9-8fa2-07c950be9c6d}")
    _methods_ = [   # IAudioSessionControl (9 métodos) y luego los de la versión 2
        COMMETHOD([], HRESULT, "GetState", (["out"], ctypes.POINTER(ctypes.c_int), "estado")),
        *[COMMETHOD([], HRESULT, f"_reservado{i}") for i in range(8)],
        COMMETHOD([], HRESULT, "GetSessionIdentifier", (["in"], ctypes.c_void_p, "id")),
        COMMETHOD([], HRESULT, "GetSessionInstanceIdentifier", (["in"], ctypes.c_void_p, "id")),
        COMMETHOD([], HRESULT, "GetProcessId", (["out"], ctypes.POINTER(wintypes.DWORD), "pid")),
    ]


class IAudioSessionEnumerator(IUnknown):
    _iid_ = GUID("{E2F5BB11-0570-40CA-ACDD-3AA01277DEE8}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetCount", (["out"], ctypes.POINTER(ctypes.c_int), "n")),
        COMMETHOD([], HRESULT, "GetSession", (["in"], ctypes.c_int, "i"),
                  (["out"], ctypes.POINTER(ctypes.POINTER(IUnknown)), "sesion")),
    ]


class IAudioSessionManager2(IUnknown):
    _iid_ = GUID("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
    _methods_ = [   # IAudioSessionManager (2 métodos) y luego GetSessionEnumerator
        COMMETHOD([], HRESULT, "_reservado0"),
        COMMETHOD([], HRESULT, "_reservado1"),
        COMMETHOD([], HRESULT, "GetSessionEnumerator",
                  (["out"], ctypes.POINTER(ctypes.POINTER(IAudioSessionEnumerator)), "sesiones")),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def procesos():
    """{pid: (pid del padre, nombre del ejecutable)} de todos los procesos."""
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    foto = kernel32.CreateToolhelp32Snapshot(0x2, 0)    # TH32CS_SNAPPROCESS
    tabla, e = {}, PROCESSENTRY32W()
    e.dwSize = ctypes.sizeof(e)
    try:
        ok = kernel32.Process32FirstW(foto, ctypes.byref(e))
        while ok:
            tabla[e.th32ProcessID] = (e.th32ParentProcessID, e.szExeFile)
            ok = kernel32.Process32NextW(foto, ctypes.byref(e))
    finally:
        kernel32.CloseHandle(foto)
    return tabla


def pids_con_audio():
    """PIDs con una sesión de audio en la salida predeterminada (sin los sonidos del sistema)."""
    enumerador = comtypes.CoCreateInstance(GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"), IMMDeviceEnumerator,
                                           comtypes.CLSCTX_ALL)
    salida = enumerador.GetDefaultAudioEndpoint(0, 1)    # eRender, eMultimedia
    gestor = salida.Activate(ctypes.byref(IAudioSessionManager2._iid_), comtypes.CLSCTX_ALL, None)
    sesiones = gestor.QueryInterface(IAudioSessionManager2).GetSessionEnumerator()
    pids = set()
    for i in range(sesiones.GetCount()):
        pid = sesiones.GetSession(i).QueryInterface(IAudioSessionControl2).GetProcessId()
        if pid and pid != os.getpid():
            pids.add(pid)
    return pids


# Procesos de Windows que lanzan apps: nunca son la raíz de una app
PROCESOS_SISTEMA = {"explorer.exe", "svchost.exe", "services.exe", "wininit.exe", "winlogon.exe", "sihost.exe",
                    "runtimebroker.exe", "smss.exe", "system", "[system process]", "userinit.exe", "dllhost.exe"}
# Componentes que reproducen el audio en nombre de otra app (Teams y WhatsApp usan WebView2)
AUXILIARES = {"msedgewebview2.exe"}


def raiz(pid, tabla):
    """Proceso raíz de la app dueña del audio. Se sube al padre si es el mismo programa (las
    pestañas de Chrome, los procesos de Discord) o si el proceso es un auxiliar como WebView2
    (Teams, WhatsApp). Una app abierta por otra (Zoom desde un enlace de Chrome) es propia."""
    vistos = {pid}
    while True:
        padre = tabla[pid][0]
        if padre not in tabla or padre in vistos or tabla[padre][1].lower() in PROCESOS_SISTEMA:
            return pid
        exe = tabla[pid][1].lower()
        if tabla[padre][1].lower() != exe and exe not in AUXILIARES:
            return pid
        vistos.add(padre)
        pid = padre


def apps_con_audio(tabla=None, con_audio=None):
    """Nombres de los programas (su raíz) con una sesión de audio, para elegir cuál escuchar."""
    tabla = procesos() if tabla is None else tabla
    con_audio = pids_con_audio() if con_audio is None else con_audio
    return sorted({tabla[raiz(p, tabla)][1] for p in con_audio if p in tabla}, key=str.lower)


def _cadena(pid, tabla):
    """El proceso y sus ancestros dentro de la misma app, del más bajo a la raíz."""
    cadena, tope = [pid], raiz(pid, tabla)
    while pid != tope:
        pid = tabla[pid][0]
        cadena.append(pid)
    return cadena


def pid_de_app(nombre, tabla=None, con_audio=None):
    """PID desde el que capturar el programa (p. ej. 'Discord.exe'), o None si no está abierto.

    Windows captura el árbol completo del PID, y una app puede haber abierto otra (Zoom desde
    un enlace de Chrome): por eso, si la app está sonando, se usa el ancestro común más bajo
    de sus procesos con sesión de audio (en Chrome, su proceso de audio), y no su raíz. Si
    todavía no suena, su raíz (que incluye al proceso que sonará)."""
    tabla = procesos() if tabla is None else tabla
    if con_audio is None:
        try:
            con_audio = pids_con_audio()
        except (OSError, comtypes.COMError):
            con_audio = set()
    es = lambda pid: tabla[pid][1].lower() == nombre.lower()
    cadenas = [_cadena(p, tabla) for p in sorted(con_audio) if p in tabla and es(raiz(p, tabla))]
    if cadenas:
        comunes = set(cadenas[0]).intersection(*cadenas[1:])
        return next(p for p in cadenas[0] if p in comunes)  # El más bajo de los comunes
    abiertos = sorted({raiz(p, tabla) for p in tabla if es(p)})
    abiertos = [r for r in abiertos if es(r)]
    return abiertos[0] if abiertos else None
