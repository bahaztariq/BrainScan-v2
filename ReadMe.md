# 🧠 BrainScan-v2

Analyse automatisée d'images cérébrales : **classification** (GoogLeNet / TensorFlow-Keras),
**détection & localisation** des tumeurs (YOLOv8), **suivi d'expériences** (MLflow),
**orchestration** (Airflow) et **exposition des modèles** via une API REST (FastAPI)
consommée par une interface Streamlit.

Ce dépôt fournit l'infrastructure complète (dépendances, images Docker, orchestration,
API, interface, dossiers) ; le code algorithmique reste à écrire.

---

## 1. Services

| Service                 | Rôle                                            | URL                   |
|-------------------------|-------------------------------------------------|-----------------------|
| `postgres`              | Métadonnées Airflow + backend store MLflow      | `localhost:5432`      |
| `mlflow`                | Tracking server + Model Registry                | http://localhost:5000 |
| `airflow-apiserver`     | UI Airflow + API v2                             | http://localhost:8080 |
| `airflow-scheduler`     | Planification des DAGs                          | —                     |
| `airflow-dag-processor` | Analyse des DAGs                                | —                     |
| `api`                   | API REST (classification + détection)           | http://localhost:8000 |
| `streamlit`             | Interface web                                   | http://localhost:8501 |
| `trainer` (profil)      | Entraînement complet (ponctuel)                 | —                     |
| `airflow-cli` (profil)  | CLI Airflow                                     | —                     |

Connexion Airflow : `admin` / `admin` (variables `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD`).

---

## 2. Démarrage

```bash
cp .env.example .env
sed -i "s/^APP_UID=.*/APP_UID=$(id -u)/" .env
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env

docker compose up -d --build
docker compose ps
```

```bash
docker compose --profile training run --rm trainer     # pipeline d'entraînement
docker compose --profile debug run --rm airflow-cli dags list
docker compose logs -f api
docker compose down                                     # arrêt (volumes conservés)
docker compose down -v                                  # arrêt + suppression des volumes
```

Prérequis : Docker ≥ 24, Compose v2, 8 Go de RAM, ~20 Go de disque.
Python local : 3.11 / 3.12 / 3.13 (3.14 non compatible avec TensorFlow 2.21 / torch 2.14).

---

## 3. Arborescence

```
.
├── requirements.txt              # toutes les dépendances (versions figées)
├── dockerfile                    # image multi-cibles : app (défaut), airflow
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── .gitignore
├── scripts/
│   └── postgres/
│       └── init-databases.sh     # création des bases airflow + mlflow
├── data/                         # datasets (volume monté)
├── models/                       # modèles entraînés
├── artifacts/                    # rapports, figures, data.yaml, artefacts MLflow
├── logs/
└── src/
    ├── brainscan/
    │   ├── api/                  # API FastAPI      -> brainscan.api.app:app
    │   ├── ui/                   # interface Streamlit
    │   └── training/             # scripts d'entraînement
    └── dags/                     # DAGs Airflow
```

### Cibles du `dockerfile`

| Cible      | Image                | Usage                                        |
|------------|----------------------|----------------------------------------------|
| `app`      | `brainscan-v2/app`   | API FastAPI, Streamlit, MLflow, entraînement (défaut) |
| `airflow`  | `brainscan-v2/airflow` | Scheduler, API server, DAG processor        |

```bash
docker compose build api                       # cible app
docker compose build airflow-scheduler         # cible airflow
```

---

## 4. Fichiers à créer

| Fichier                                        | Contenu                                                     |
|------------------------------------------------|-------------------------------------------------------------|
| `src/brainscan/api/app.py`                      | FastAPI : `/health`, `/predict/classification`, `/predict/detection` |
| `src/brainscan/ui/app.py`                       | Streamlit : onglets Classification / Détection              |
| `src/brainscan/training/run_pipeline.py`        | Point d'entrée du pipeline (`python -m brainscan.training.run_pipeline`) |
| `src/brainscan/training/prepare.py`             | Préparation et split 70/15/15 des datasets                   |
| `src/brainscan/training/validate.py`            | Validation structure / qualité (annotations YOLO, classes)   |
| `src/brainscan/training/classification.py`      | Entraînement GoogLeNet + métriques + matrice de confusion     |
| `src/brainscan/training/detection.py`           | Préparation YOLO (`data.yaml`) + entraînement YOLOv8          |
| `src/brainscan/training/evaluation.py`          | Évaluation finale, comparaison des runs, Model Registry      |
| `src/dags/brainscan_pipeline.py`                | DAG Airflow (6 tâches)                                       |

Commandes attendues par le DAG et par `run_pipeline.py` :

```
python -m brainscan.training.prepare        --data-dir ... --classification-dir ... --detection-dir ...
python -m brainscan.training.validate       --data-dir ... --classification-dir ... --detection-dir ...
python -m brainscan.training.classification --data-dir ... --models-dir ... --artifacts-dir ...
python -m brainscan.training.detection      --data-dir ... --models-dir ... --artifacts-dir ...
python -m brainscan.training.evaluation     --data-dir ... --models-dir ... --artifacts-dir ...
```

---

## 5. Variables d'environnement

| Variable                     | Défaut                                              |
|------------------------------|-----------------------------------------------------|
| `APP_UID` / `AIRFLOW_UID`    | `1000` / `50000` (mettre `id -u` pour les droits)  |
| `POSTGRES_USER/PASSWORD`     | `brainscan`                                         |
| `AIRFLOW_ADMIN_USER/PASSWORD`| `admin` / `admin`                                   |
| `AIRFLOW_FERNET_KEY`         | à générer (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`) |
| `MLFLOW_TRACKING_URI`        | `http://mlflow:5000`                                |
| `MLFLOW_EXPERIMENT_NAME`     | `brainscan-v2`                                      |
| `CLASSIFICATION_MODEL_PATH`  | `/app/models/classification`                        |
| `DETECTION_MODEL_PATH`       | `/app/models/detection`                             |
| `CLASSIFICATION_CLASSES`     | `glioma,meningioma,pituitary,no_tumour`            |
| `DETECTION_CLASSES`          | `tumour`                                            |
| `API_BASE_URL`               | `http://api:8000`                                   |
| `API_PORT` / `STREAMLIT_PORT` / `AIRFLOW_PORT` / `MLFLOW_PORT` | `8000` / `8501` / `8080` / `5000` |

---

## 6. Notes techniques

* **GoogLeNet** : Keras 3 (et Keras 2) ne fournissent plus `keras.applications.googlenet`.
  Pistes : ONNX Model Zoo (`onnx` + `onnxruntime` + `onnx2keras` installé via
  `pip install --no-deps onnx2keras==0.0.25`), poids `torchvision.models.googlenet`,
  ou repli `tf_keras.applications.inception_v3`.
* **Airflow** : installé avec le fichier de contraintes officiel
  `constraints-3.3.2/constraints-3.12.txt` (versions alignées avec `requirements.txt`).
* **PyTorch** : wheels CPU uniquement (`download.pytorch.org/whl/cpu`) pour ne pas
  embarquer CUDA dans les images.
* **Augmentation** : `keras.layers.RandomFlip` / `RandomBlur` / `RandomGaussianNoise`
  (nativement disponibles, sans conflit OpenCV).

---

## 7. Installation locale (sans Docker)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export PYTHONPATH=$PWD/src
export MLFLOW_TRACKING_URI=http://localhost:5000

uvicorn brainscan.api.app:app --reload
streamlit run src/brainscan/ui/app.py
airflow standalone
```

---

## 8. Tester le modèle InceptionV3 pré-entraîné

Le notebook `notebooks/6-test-pretrained-classifier.ipynb` teste une image avec le modèle InceptionV3 pré-entraîné. Son checkpoint est inclus dans le dépôt à `models/classification/best_model.keras` ; après avoir cloné le dépôt, le notebook le charge automatiquement depuis cet emplacement.

Le modèle utilise `tf_keras` et `tensorflow_hub` (déjà listés dans `requirements.txt`), attend des images RGB `299 × 299` normalisées dans `[0, 1]` et produit les classes `glioma`, `meningioma`, `notumor`, `pituitary`. Au premier chargement, une connexion Internet peut être nécessaire pour que TensorFlow Hub télécharge le backbone InceptionV3.
