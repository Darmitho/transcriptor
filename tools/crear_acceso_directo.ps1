# Crea el acceso directo "Transcriptor en vivo" en el escritorio y en el menú Inicio.
# Abre la app con pythonw.exe del venv: sin ventana de consola (el registro va a
# %LOCALAPPDATA%\Transcriptor\transcriptor.log). Como usa el código del proyecto, cualquier
# cambio se refleja al abrirla de nuevo, sin recompilar nada.
# Uso: powershell -ExecutionPolicy Bypass -File tools\crear_acceso_directo.ps1

$raiz = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $raiz "venv\Scripts\pythonw.exe"
$script = Join-Path $raiz "transcriptor.py"
$icono = Join-Path $raiz "assets\transcriptor.ico"
foreach ($ruta in @($pythonw, $script, $icono)) {
    if (-not (Test-Path $ruta)) { Write-Error "No existe: $ruta"; exit 1 }
}

$shell = New-Object -ComObject WScript.Shell
$destinos = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
foreach ($carpeta in $destinos) {
    $acceso = $shell.CreateShortcut((Join-Path $carpeta "Transcriptor en vivo.lnk"))
    $acceso.TargetPath = $pythonw
    # -E: ignora las variables PYTHON* del sistema. Un PYTHONPATH global con un sitecustomize.py
    # que falle (p. ej. por un import que no está en este venv) mata a pythonw.exe al arrancar,
    # porque sin consola no tiene dónde mostrar el error
    $acceso.Arguments = "-E `"$script`""
    $acceso.WorkingDirectory = $raiz
    $acceso.IconLocation = "$icono,0"
    $acceso.Description = "Subtítulos en vivo del audio del PC, 100 % local"
    $acceso.Save()
    Write-Output "Creado: $(Join-Path $carpeta 'Transcriptor en vivo.lnk')"
}
