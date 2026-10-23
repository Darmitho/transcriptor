"""Captura del audio de una sola aplicación (o de todo menos una) en Windows.

Usa la API de "application loopback" de WASAPI (Windows 10 compilación 20348 o posterior):
ActivateAudioInterfaceAsync sobre el dispositivo virtual VAD\\Process_Loopback, con el PID
del proceso. Incluye los procesos hijos (Chrome y Discord reproducen desde uno hijo).
Basado en el ejemplo oficial de Microsoft "ApplicationLoopback".

CapturaApp(pid, excluir, al_recibir).iniciar() entrega bloques float32 mono a TASA_CAPTURA
llamando a al_recibir(bloque) desde su propio hilo; detener() la corta.
"""

import ctypes
import threading
from ctypes import wintypes

import comtypes
import numpy as np
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

TASA_CAPTURA = 48000
CANALES = 2

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
            cliente.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 200_000, 0, ctypes.byref(formato), None)  # 20 ms
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
            while self.seguir.is_set():
                if kernel32.WaitForSingleObject(evento, 200) != 0:   # 0 = WAIT_OBJECT_0
                    continue
                while captura.GetNextPacketSize():
                    datos, cuadros, flags_buf = captura.GetBuffer(None, None)
                    if flags_buf & AUDCLNT_BUFFERFLAGS_SILENT:
                        bloque = np.zeros(cuadros, np.float32)
                    else:
                        crudo = np.ctypeslib.as_array(ctypes.cast(datos, ctypes.POINTER(ctypes.c_int16)),
                                                      shape=(cuadros * CANALES,))
                        bloque = crudo.reshape(-1, CANALES).mean(axis=1).astype(np.float32) / 32768
                    captura.ReleaseBuffer(cuadros)
                    self.al_recibir(bloque)
        finally:
            cliente.Stop()
            kernel32.CloseHandle(evento)
