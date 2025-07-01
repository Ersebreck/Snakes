import assemblyai as aai
from tqdm import tqdm
import os
from collections import defaultdict

# Configura tu API key
aai.settings.api_key = "5f2cf45d25f14dd3807065b5b05fa3ba"

# Directorio base
base_path = "/media/berrakero/FANY R/VIDEOS BV JUN 2025"

# Recorrer archivos y agrupar por carpeta principal (subfolder inmediato a base_path)
carpetas_videos = defaultdict(list)
for root, dirs, files in os.walk(base_path):
    for file in files:
        breakpoint()
        if file.endswith(".MOV") and "OTROS" in file:
            full_path = os.path.join(root, file)
            carpeta = os.path.relpath(root, base_path).split(os.sep)[0]
            carpetas_videos[carpeta].append(full_path)

# Procesar carpeta por carpeta
for carpeta, archivos in carpetas_videos.items():
    print(f"🔊 Transcribiendo carpeta: {carpeta} con {len(archivos)} archivos")
    transcripcion_total = ""

    for audio_file in tqdm(archivos, desc=f"Procesando {carpeta}"):
        try:
            transcriber = aai.Transcriber()
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.best,
                language_code="es",
                punctuate=True,
                format_text=True,
                speaker_labels=False,
            )
            transcript = transcriber.transcribe(audio_file, config)

            # Agrega nombre del archivo y su transcripción
            transcripcion_total += f"\n=== {os.path.basename(audio_file)} ===\n"
            transcripcion_total += transcript.text + "\n\n"

        except Exception as e:
            print(f"❌ Error con {audio_file}: {e}")

    # Guardar transcripción por carpeta
    carpeta_salida = os.path.join(base_path, f"{carpeta}.txt")
    with open(carpeta_salida, "w", encoding="utf-8") as f:
        f.write(transcripcion_total)

    print(f"✅ Archivo guardado en: {carpeta_salida}")
