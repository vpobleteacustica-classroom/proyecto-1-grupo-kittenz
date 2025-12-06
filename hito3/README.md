# Hito 3: Pipeline de Extracción de Características de Audio y red neuronal multicapa (MLP)

## Descripción General

Este hito implementa un pipeline automatizado para procesar el dataset **MTG-Jamendo** (subset mood/theme) que:

1. **Descarga** archivos comprimidos (TAR) con audios desde el repositorio del dataset  
2. **Extrae** características musicales de cada audio usando bibliotecas de análisis de audio  
3. **Asocia** cada audio con sus etiquetas de mood/theme  
4. **Genera** un CSV con todas las características extraídas para análisis posterior

Esto con el objetivo de entrenar un modelo MLP (una red neuronal multicapa), y así obtener mejores resultados que el primer intento en el hito 2, esto para reconocer la emocionalidad en música de videojuegos.

## Archivos Principales

- **`batch_download_and_extract.py`**: Script principal que ejecuta todo el pipeline  
- **emotion\_mlp.py :** Define una red neuronal profunda (MLP) con bloques residuales para clasificación de emociones.

- #### **`prepare_data.py`** \- Preparación de Datos

- #### **`train.py`** \- Entrenamiento del Modelo

## ¿Qué hace el código?

### batch\_download\_and\_extract.py 

### 1\. Descarga de Datos

- Lee la lista de archivos TAR del dataset desde los archivos de checksums  
- Descarga cada TAR desde el servidor especificado  
- Verifica la integridad usando checksums SHA256  
- Implementa sistema de checkpoints para reanudar descargas interrumpidas

### 2\. Extracción de Características

Para cada audio MP3 dentro de los TARs, el script extrae:

#### Características Temporales y Rítmicas (Librosa)

- **Tempo**: Velocidad de la música en BPM (beats por minuto)  
- **Time Signature**: Compás estimado (2/4, 3/4, 4/4) basado en el tempo  
- **Onset Strength**: Fuerza de los ataques/inicios de notas

#### Características Espectrales (Librosa)

- **MFCCs (13 coeficientes)**: Media y desviación estándar  
  - Representan el timbre y textura del audio  
- **Spectral Centroid**: "Brillo" del sonido (centro de masa del espectro)  
- **Spectral Rolloff**: Frecuencia por debajo de la cual está el 85% de la energía  
- **Chroma Features (12 valores)**: Distribución de energía en las 12 notas cromáticas  
- **Zero-Crossing Rate**: Tasa de cruces por cero (relacionado con ruido/percusión)

#### Características de Energía

- **RMS Energy**: Energía promedio de la señal  
- **Loudness (Essentia)**: Volumen percibido del audio

### 3\. Asociación con Metadata

- Carga el archivo `autotagging_moodtheme.tsv` del dataset  
- Asocia cada audio con sus etiquetas de mood/theme (ej: "happy", "dark", "energetic")  
- Utiliza el parser oficial del repositorio MTG-Jamendo para compatibilidad

### 4\. Generación del Dataset

- Guarda todas las características en un CSV con las siguientes columnas:  
  - Identificación: `tar`, `filename`, `basename`, `track_id`  
  - Características extraídas: todas las mencionadas arriba  
  - Metadata: `moodtheme_tags` (etiquetas separadas por `;`)  
  - Performance: `processing_time_ms` (tiempo de procesamiento por archivo)

## Requisitos

### Dependencias de Python

\`\`\`  
pip install pandas numpy librosa essentia-tensorflow requests tqdm  
\`\`\`

Cabe recalcar que el para el uso efectivo para extracción de los datos, es necesario ejecutarlo en un entorno de un sistema basado en linux/unix para poder utilizar de manera efectiva essentia y extraer los valores que únicamente se puede con este mismo.

### Dependencias del Sistema

- **ffmpeg**: Necesario para Librosa y pydub

### Dataset MTG-Jamendo

1. Clonar el repositorio oficial:  
     
   \`\`\`  
   git clone [https://github.com/MTG/mtg-jamendo-dataset.git](https://github.com/MTG/mtg-jamendo-dataset.git)  
   \`\`\`  
     
2. El script usa los archivos de metadata del repositorio clonado

## Uso

### Ejecución de script

\`\`\`  
DATA\_TYPE=audio-low REPO\_ROOT=./mtg-jamendo-dataset WORK\_DIR=./work\_low OUTPUT\_CSV=./features\_low.csv CHECKPOINT\_CSV=./processed\_low.csv MAX\_TARS=0 python3 batch\_download\_and\_extract.py  
\`\`\`

## Salidas Generadas

1. **`features_moodtheme.csv`**: Dataset final con todas las características  
2. **`processed_tars.csv`**: Checkpoint de TARs procesados  
3. **`work_moodtheme/`**: Directorio temporal (se limpia automáticamente)

### emotion\_mlp.py

Define una red neuronal profunda (MLP) con bloques residuales para clasificación de emociones:

**Componentes:**

- **ResidualBlock**: Bloques con conexiones residuales (skip connections)  
    
  - 2 capas lineales con Batch Normalization  
  - Activación GELU (Gaussian Error Linear Unit)  
  - Dropout para regularización  
  - Conexión residual: `output = activation(x + net(x))`


- **EmotionMLP**: Modelo principal  
    
  - Capa de entrada que proyecta features a dimensión oculta  
  - Stack de bloques residuales (profundidad configurable)  
  - Capa de salida para clasificación

**Parámetros configurables:**

- `input_dim`: Dimensión de las características de entrada  
- `num_classes`: Número de clases/emociones  
- `hidden_dim`: Dimensión de las capas ocultas (default: 128\)  
- `depth`: Número de bloques residuales (default: 4\)  
- `dropout`: Tasa de dropout para regularización (default: 0.3)

#### prepare\_data.py \- Preparación de Datos

Script para cargar, procesar y preparar los datos del CSV de características:

**Funciones principales:**

- **`read_features_csv()`**: Carga el CSV y valida columnas  
    
  - Features esperadas: tempo, loudness, time\_signature, spectral features, MFCCs, chroma


- **`build_tag_vocab()`**: Construye vocabulario de tags de mood/theme  
    
  - Parsea la columna `moodtheme_tags` (tags separados por `;`)  
  - Crea lista ordenada de todos los tags únicos


- **`encode_single_label()`**: Codificación para clasificación mono-etiqueta  
    
  - Usa el primer tag de cada audio como clase  
  - Filtra audios sin etiquetas


- **`encode_multi_label()`**: Codificación para clasificación multi-etiqueta  
    
  - Vector binario multi-hot de tamaño `len(vocab)`  
  - Permite múltiples emociones por audio


- **`build_X()`**: Construye matriz de características  
    
  - Normalización z-score por columna (opcional)  
  - Formula: `X_norm = (X - mean) / std`


- **`split_train_val_test()`**: División del dataset  
    
  - 70% entrenamiento, 15% validación, 15% test  
  - Split aleatorio con seed para reproducibilidad


- **`prepare()`**: Función principal de preparación  
    
  - Retorna tensores de PyTorch listos para entrenar  
  - Soporta modo `single` o `multi` label

#### 3\. `train.py` \- Entrenamiento del Modelo

Script principal para entrenar y evaluar el modelo de clasificación:

**Características principales:**

##### Manejo de Desbalanceo de Clases

- **`compute_pos_weight()`**: Calcula pesos para clases desbalanceadas  
  - Formula: `pos_weight_j = N_neg_j / N_pos_j`  
  - Útil para multi-label con BCEWithLogitsLoss

##### Optimización de Umbrales

- **`find_per_class_thresholds()`**: Ajusta umbrales por clase para maximizar F1  
  - Por defecto usa 0.5 como umbral  
  - Busca en grid \[0.1, 0.15, ..., 0.9\]  
  - Optimiza F1-score por clase independientemente

##### Funciones de Entrenamiento

- **`train_epoch()`**: Entrena una época con mini-batches  
- **`evaluate_single()`**: Evaluación para clasificación mono-etiqueta  
  - Métrica: Accuracy  
- **`evaluate_multi()`**: Evaluación para clasificación multi-etiqueta  
  - Métricas: F1-micro, F1-macro, Subset Accuracy, Label Accuracy Mean

##### Ejecutar

Para poder ejecutar el modelo se utiliza el comando:  
\`\`\`  
python3 train.py \--csv ./features\_low.csv \--mode multi \--epochs 30  
\`\`\`

Donde:

- **“csv”** :  El archivo de extensión .csv con el cual se va a entrenar el modelo, en nuestro caso “./features\_low.csv”, que son los datos que se generaron gracias a el código anterior.  
- **“epochs”** : Es la cantidad de veces que se va a ejecutar el entrenamiento y cuánto se va a considerar para la evaluación al final.  
- **“mode”** : modo del cual está funcionando el código, con dos opciones, donde:

  **Mono-etiqueta (single):**


- Accuracy: Porcentaje de predicciones correctas


  **Multi-etiqueta (multi):**


- **F1-micro**: F1 global considerando todas las predicciones  
- **F1-macro**: Promedio de F1 por clase (no ponderado)  
- **Subset Accuracy**: Porcentaje de predicciones exactas (todas las etiquetas correctas)  
- **Label Accuracy Mean**: Accuracy promedio por etiqueta

##### Salidas Generadas

- **`emotion_mlp.pt`**: Pesos del modelo entrenado  
- **`thresholds_per_class.npy`**: Umbrales optimizados (si se usa `--threshold_tuning`)

#### 4\. Archivos de Datos

- **`features_low.csv`**: CSV con características extraídas (audio-low quality)  
- **`processed_low.csv`**: Dataset procesado y listo para entrenar

---

### Resultados del modelo

Luego de realizar pruebas con el modelo en sí, únicamente se logró un accuracy máximo durante el entrenamiento de 27%, mientras que el modelo anterior había logrado un 40%.  A pesar del cambio de enfoque y de los datos utilizados, el nuevo modelo no presentó un rendimiento superior, no se logró realizar un modelo que muestre una buena precisión para la buena predicción sobre qué emoción proporciona una canción. Este valor tan pequeño de accuracy puede deberse a cómo está conformado los distintos tipos de dataset que consideramos, y a lo mismo los distintos tipos de datos que no se pueden unificar en algo común, haciendo que el modelo no pueda predecir de manera correcta la emoción.

## Referencias

- **Dataset**: [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset)  
- **Librosa**: Análisis de audio en Python  
- **Essentia**: Biblioteca de extracción de características de audio de Music Technology Group

