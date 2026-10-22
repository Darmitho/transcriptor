# Convertir los modelos de traducción a CTranslate2

Se hace **una sola vez**, con conexión a internet y en un entorno aparte: la conversión
necesita PyTorch y `transformers`, que la app no usa. Después el modelo funciona sin red.

```powershell
# Entorno temporal (ruta corta: algunas rutas de estos paquetes superan el límite de Windows)
py -3.13 -m venv $env:TEMP\ct2conv
& $env:TEMP\ct2conv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $env:TEMP\ct2conv\Scripts\python.exe -m pip install transformers sentencepiece sacremoses ctranslate2==4.8.3

# Destino: fuera del repo y de OneDrive
$destino = "$env:USERPROFILE\.cache\transcriptor\traduccion"
New-Item -ItemType Directory -Force $destino | Out-Null
cd $destino

# El elegido (Apache-2.0, 79 MB)
& $env:TEMP\ct2conv\Scripts\ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-es `
    --output_dir opus-mt-en-es --quantization int8 --copy_files source.spm target.spm

# Solo para comparar en tools\bench_traduccion.py
& $env:TEMP\ct2conv\Scripts\ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-en-es `
    --output_dir opus-mt-tc-big-en-es --quantization int8 --copy_files source.spm target.spm
& $env:TEMP\ct2conv\Scripts\ct2-transformers-converter --model facebook/nllb-200-distilled-600M `
    --output_dir nllb-200-distilled-600M --quantization int8 --copy_files sentencepiece.bpe.model

# Al terminar, el entorno temporal se puede borrar
Remove-Item -Recurse -Force $env:TEMP\ct2conv
```

| Modelo | Licencia | Tamaño int8 |
|---|---|---|
| Helsinki-NLP/opus-mt-en-es | Apache-2.0 | 79 MB |
| Helsinki-NLP/opus-mt-tc-big-en-es | CC-BY-4.0 | 228 MB |
| facebook/nllb-200-distilled-600M | CC-BY-NC-4.0 (no comercial: solo referencia) | 605 MB |
