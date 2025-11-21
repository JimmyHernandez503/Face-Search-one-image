Oceano — Motor de búsqueda facial sobre Qdrant

Nombre en clave: oceano

Oceano es un sistema de búsqueda y comparación de rostros a gran escala, pensado para trabajar con millones de personas a partir de una foto frontal por individuo (fondo blanco), utilizando:

FastAPI como backend y UI web sencilla.

Qdrant como base de datos vectorial.

InsightFace (buffalo_l) como modelo de embeddings faciales.

Docker para empaquetar y desplegar todo el entorno.

Este repositorio está preparado para subir a GitHub:
no incluye modelos, datos, thumbnails ni índices, solo código y configuración.

Características principales

🔍 Búsqueda facial top-K (por defecto, top-10 resultados más parecidos).

🧠 Modelo InsightFace buffalo_l (más preciso que buffalo_s).

🎛️ Test-Time Augmentation (TTA) en las consultas:

imagen original

flip horizontal

variaciones de brillo

rotación ligera ±15°
Se promedian los embeddings para una consulta más robusta.

📦 Ingesta incremental:

Recorre carpetas de imágenes.

Genera embeddings en GPU/CPU.

Guarda estado en SQLite para reanudar.

Crea thumbnails por persona.

⚙️ Qdrant / HNSW afinado para grandes volúmenes:

Distancia COSINE sobre vectores de 512 dimensiones.

Parámetros HNSW ajustables (m, ef_construct, hnsw_ef).

🌐 UI web sencilla:

Subes una foto.

Devuelve lista de candidatos con:

% de similitud.

DUI / ID (del nombre de archivo).

Ruta original.

Thumbnail.

Arquitectura

app/app/main.py
API FastAPI, endpoints:

GET / → formulario web.

POST /search → búsqueda facial.

GET /healthz → health-check.

GET /status → estado de la colección en Qdrant.

app/app/embeddings.py

Inicializa el modelo InsightFace (buffalo_l por defecto).

Lee y reescala imágenes.

Obtiene el embedding del rostro más grande.

Implementa best_face_embedding_tta (TTA ligero para consultas).

app/app/ingest.py

Ingesta masiva/incremental de imágenes.

Crea thumbnails.

Inserta vectores en Qdrant usando HNSW.

Guarda estado en SQLite (ingestion.db).

app/app/templates/index.html

UI web para subir una foto y ver resultados.

app/app/static/

CSS y recursos estáticos.

docker-compose.yml

Servicio api: FastAPI + modelo InsightFace.

Servicio qdrant: base vectorial.

Monta volúmenes (models, thumbs, qdrant_storage, logs, state).

Requisitos

Sistema operativo: Linux recomendado (probado en Ubuntu 24.x).

Docker y docker compose.

Opcional (GPU):

NVIDIA drivers instalados en el host.

nvidia-container-toolkit configurado.

La línea gpus: all en docker-compose.yml habilita el uso de la GPU.

Sin GPU, el proyecto puede funcionar en CPU (más lento) ajustando el código para usar solo CPUExecutionProvider.

Puesta en marcha rápida (instancia Oceano)

Clona el repositorio:

git clone https://github.com/TU_USUARIO/oceano.git
cd oceano


Crea directorios de trabajo (se pueden versionar o ignorar según tu flujo):

mkdir -p models thumbs qdrant_storage logs state


Levanta servicios (API + Qdrant):

docker compose up -d --build


Abre la UI:

Navegador → http://localhost:9100/

La primera vez que busques, InsightFace descargará automáticamente el modelo en ./models.

Ingesta de rostros

La ingesta recorre una carpeta de fotos (por ejemplo, una foto frontal por persona) y:

Calcula el embedding con buffalo_l.

Inserta el vector en Qdrant.

Crea un thumbnail por cada foto.

Guarda el estado en SQLite para poder reanudar.

Preparar el dataset

Una foto por persona (idealmente frontal, fondo neutro/blanco).

Nombres de archivo con el identificador de la persona, por ejemplo:

02239037-4.jpg

06024583-0.jpg

Ese nombre se usará como campo dui en el payload de Qdrant.

Comando de ingesta (desde el host)

Ejemplo: si tus fotos están en ~/Documentos/fotos_f:

cd /ruta/a/oceano

docker compose run --rm \
  api \
  python3 -m app.ingest \
  --path "/hosthome/Documentos/fotos_f" \
  --batch 256


Notas:

El docker-compose.yml monta ${HOME} del host como /hosthome en el contenedor.
Por eso la ruta dentro del contenedor es /hosthome/….

--batch 256 controla cuántos puntos se envían por lote a Qdrant.

La ingesta es incremental: guarda el estado en state/ingestion.db.
Si vuelves a lanzar el comando, solo procesará lo nuevo o lo modificado.

Uso de la interfaz web

Abre http://localhost:9100/.

Sube una foto con un rostro (idealmente frontal).

El sistema:

Detecta el rostro.

Aplica TTA y genera un embedding robusto.

Consulta Qdrant para obtener el top-K (por defecto, 10) más similares.

La página muestra, por cada coincidencia:

% de similitud (cosine similarity × 100).

DUI/ID (extraído del nombre de archivo original).

Ruta original.

Thumbnail (si existe).

Variables de entorno clave

Definidas en docker-compose.yml (servicio api):

QDRANT_URL / QDRANT_GRPC_URL
URLs internas para la instancia de Qdrant.

COLLECTION_NAME
Nombre de la colección en Qdrant (por defecto faces).

THUMBS_DIR
Directorio donde se guardan los thumbnails (montado como volumen).

SQLITE_DB
Ruta del archivo SQLite donde se guarda el estado de la ingesta.

MODEL_NAME
Modelo de InsightFace (buffalo_l por defecto).

DET_SIZE
Tamaño de entrada del detector de rostros (640,640 por defecto).

MAX_SIDE, DOWNSCALE_TO
Parámetros para reescalar imágenes muy grandes antes de pasar al modelo.

TOP_K
Número de resultados a devolver en cada búsqueda (por defecto 10).

SIM_THRESHOLD
Umbral de similitud mínima.

0.0 → no filtra nada, se muestran siempre los top-K.

Valores típicos para modo estricto: 0.40–0.50.

HNSW_EF
Parámetro de búsqueda HNSW (trade-off entre velocidad y recall).

TTA_SEARCH
1 para activar TTA en consultas, 0 para desactivarlo.

QUANTIZATION
none o scalar (cuantización INT8 en Qdrant para ahorrar RAM).

Modo sin GPU (CPU)

Si se ejecuta en una máquina sin GPU o sin drivers NVIDIA, es posible usar Oceano solo en CPU:

Ajustando embeddings.py para configurar InsightFace únicamente con CPUExecutionProvider.

Usando ctx_id=-1 en app.prepare().

Esto reduce la velocidad de ingesta y búsqueda, pero mantiene la funcionalidad básica.

Notas de privacidad y legales

Oceano es un sistema de búsqueda facial genérico. El uso en entornos reales debe respetar:

Legislación local sobre protección de datos personales y biometría.

Políticas internas de la organización para:

recolección de imágenes,

almacenamiento de rostros,

acceso a resultados y auditoría.

Este repositorio no incluye ningún dataset real de personas ni modelos entrenados propietarios. Los modelos de InsightFace se descargan desde sus repositorios oficiales, sujetos a sus propias licencias.
