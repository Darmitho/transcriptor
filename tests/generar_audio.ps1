# Regenera los audios de tests\audio con las voces TTS de Windows (Sabina es-MX, Zira en-US).
# Solo hace falta si se pierden o se quieren frases nuevas: los .wav ya están en el repo.
# Uso: powershell -ExecutionPolicy Bypass -File tests\generar_audio.ps1
# Luego convertir a 16 kHz mono (los tests los suben a 48 kHz al cargarlos).

$dest = Join-Path $PSScriptRoot "audio_48k"
New-Item -ItemType Directory -Force $dest | Out-Null
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$es = [ordered]@{
    "frase"    = "Buenos días a todos. Hoy vamos a revisar el presupuesto del proyecto y las fechas de entrega."
    "gracias"  = "Gracias."
    "gracias2" = "Muchas gracias."
    "parte0"   = "Primero revisamos el presupuesto del trimestre"
    "parte1"   = "luego vimos las fechas de entrega del proyecto"
    "parte2"   = "después hablamos de los riesgos principales"
    "parte3"   = "y finalmente acordamos los próximos pasos con el cliente"
    "parte4"   = "la reunión siguiente será el martes a las diez"
    "v1"       = "Esta es una prueba para Claude, para probar el micrófono."
    "v2"       = "Si el precio toca el order block, esperamos un pullback y miramos el RSI."
    "v3"       = "Saludos a toda la comunidad de CriptoBuzz que nos acompaña hoy."
    "v4"       = "bueno vamos ahora a ver qué pasa con el precio y nos vamos para arriba"
    "v5"       = "Mañana revisamos el contrato con el cliente y las fechas de pago."
    "continuo" = (Get-Content (Join-Path $PSScriptRoot "audio\continuo.txt") -Encoding UTF8 -Raw).Trim()
}
$en = [ordered]@{
    "en1" = "Today we are reviewing the quarterly budget with the client and the delivery dates."
    "en2" = "Let's check the order block and the RSI before we open a position on Bitcoin."
    "en3" = "Thank you."
    "en4" = "Could you share your screen so everyone can see the report?"
}

function Hablar($voz, $frases) {
    foreach ($k in $frases.Keys) {
        $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $s.SelectVoice($voz)
        $s.SetOutputToWaveFile((Join-Path $dest "$k.wav"), $fmt)
        $s.Speak($frases[$k])
        $s.Dispose()
    }
}
Hablar "Microsoft Sabina Desktop" $es
Hablar "Microsoft Zira Desktop" $en
