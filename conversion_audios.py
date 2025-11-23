import os
from pydub import AudioSegment

# Recordar instalar pydub y ffmpeg si no lo tienes:
# pip install pydub
# sudo apt update && sudo apt install ffmpeg (o algo así)

# --- CONFIGURACIÓN ---
# Pon aquí la ruta donde están tus carpetas con los MP3
CARPETA_ORIGEN = "./audios" 

# Pon aquí donde quieres guardar los WAV
CARPETA_DESTINO = "./audios-wav"

# Crear la carpeta de destino si no existe
if not os.path.exists(CARPETA_DESTINO):
    os.makedirs(CARPETA_DESTINO)

print(f"Iniciando conversión de {CARPETA_ORIGEN} a {CARPETA_DESTINO}...")

# Recorrer la carpeta origen y todas sus subcarpetas
contador = 0

for root, dirs, files in os.walk(CARPETA_ORIGEN):
    for archivo in files:
        if archivo.lower().endswith(".mp3"):
            try:
                # 1. Construir rutas completas
                ruta_mp3_completa = os.path.join(root, archivo)
                
                # Definir nombre de salida (cambiamos la extensión a .wav)
                nombre_wav = os.path.splitext(archivo)[0] + ".wav"
                ruta_wav_completa = os.path.join(CARPETA_DESTINO, nombre_wav)
                
                # 2. Cargar y Convertir
                # print(f"Procesando: {archivo}...") # Descomenta si quieres ver cada archivo (puede llenar la consola)
                audio = AudioSegment.from_mp3(ruta_mp3_completa)
                
                # 3. Exportar
                audio.export(ruta_wav_completa, format="wav")
                
                contador += 1
                if contador % 10 == 0:
                    print(f"Llevamos {contador} audios convertidos...")

            except Exception as e:
                print(f"Error con el archivo {archivo}: {e}")

print(f"¡Listo! Se convirtieron {contador} archivos en total.")