050_probe_npy.py: Recorre una carpeta de .npy y devuelve estadisticas que explican su forma, se uso para ver si los espectrogramas de mel descargados de jamendo eran comparables

100_build_metadata_single_label.py: Lee un archivo TSV (formato para archivos metadata en repositorio de jamendo), especificamente se usa para leer la metadata del archivo del subset autotag/moodtheme y revisa entre los archivos descargados cuale estan, obtiene la etiqueta y genera un registro con la ruta al .npy y su clase.
Crea un csv con todos los tracks que tienen su .npy descargado, y un JSON con el listado de clases encontradas, es decir que crea el dataset de entrenamiento. Se queda con el primer label si hay multiples

110_build_metadata_multilabel: Igual que 100_build_metadata_single_label pero para guardar multiples labels

200_split_70_15_15: Usa el CSV creado por 100_build_metadata_single_label para separar en train, val, y test, hace esto añadiendo una columna split, que dice a que subset pertenece cada .npy

210_split_70_15_15_multilabel: mismo proposito de 200_split_70_15_15 pero para el csv multilabel

300_train_cnn_single: Carga el csv dividido en train, val y test, y lo usa para entrenar el modelo definido en src_tiny_mel_cnn_single, se encarga de entrenar usando 3 semillas a travez de un numero de epocas, y gauardar el mejor modelo al final. Crea el dataset en si con dataset_npy_single

310_train_cnn_multilabel y 311_train_cnn_multilabel_minimal: Hacen lo mismo que la version single, pero con csv y modelos correspondientes para usar multilabel y multilabel con una version de poca complejidad. Crea el dataset en si con dataset_npy_multilabel

350_infer_one: Carga el mejor modelo entrenado he infiere la clase de un .npy

360_infer_one_multilabel: Mismo caso que 350_infer_one pero para multilabel

dataset_npy_single: Crea datalodaer con csv de splits para entrenamiento con single label

dataset_npy_multilabel_ Mismo proposito que dataset_npy_multilabel para entrenamiento con multilabel

src_tiny_mel_cnn_single: Modelo de cnn para single label
src_tiny_mel_cnn_multilabel: Modelo de cnn para multilabel
src_tiny_mel_cnn_minimal: Modelo de prueba con poca complejidad

Estos archivos estan pensados para trabajarse dentro del repositorio de https://github.com/MTG/mtg-jamendo-dataset/tree/master