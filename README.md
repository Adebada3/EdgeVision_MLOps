# 🌿 EdgeVision — Classification de maladies des feuilles de tomate

**Projet MLOps : entraînement, optimisation, évaluation et déploiement sur iPhone 16 Plus.**

EdgeVision reconnaît trois classes à partir d'une photo de feuille de tomate : **alternariose** (`Tomato___Early_blight`), **mildiou** (`Tomato___Late_blight`) et **feuille saine** (`Tomato___healthy`). Le modèle est un **MobileNetV3 Small** fine-tuné sur un sous-ensemble de PlantVillage.

> **Limites :** prototype pédagogique, non destiné au diagnostic agronomique. Les résultats sur PlantVillage ne garantissent pas les performances sur des photos prises au champ.

## Démonstrations

- **Streamlit sur ordinateur** : import d'image, prédiction, confiance, choix du modèle et résultats de benchmark (`app.py`). L'inférence Streamlit s'exécute sur l'ordinateur hébergeant l'application.
- **EdgeVision Mobile sur iPhone** : interface `mobile_web/index.html`, inférence **locale dans Safari** via ONNX Runtime Web (WebAssembly), choix FP32 / INT8 dynamique et benchmark de 10 inférences. Le serveur Python distribue la page et les fichiers, mais ne calcule pas la prédiction. Cette version nécessite un accès réseau pour charger les ressources ; elle n'est pas une application iOS native ni une PWA hors ligne.

## Résultats

### Évaluation sur Mac — jeu de test (677 images)

| Modèle ONNX | Accuracy | Macro-F1 | Taille | Latence médiane Mac |
|---|---:|---:|---:|---:|
| FP32 | 98,97 % | 98,90 % | 6,09 Mo | 1,769 ms |
| INT8 dynamique | 98,97 % | 98,90 % | 4,33 Mo | 1,765 ms |
| INT8 statique | 42,39 % | 20,36 % | 1,72 Mo | 1,066 ms |

L'INT8 statique a été **écarté**, car la perte de précision est trop importante. L'INT8 dynamique réduit la taille d'environ **29 %** sans perte d'accuracy mesurée sur le jeu de test Mac.

### Benchmark sur iPhone 16 Plus — Safari / WASM

Même image pour les deux modèles ; trois séries de 10 inférences, après échauffement. Les valeurs ci-dessous sont les **médianes de chaque série**, et leur moyenne (pas la médiane globale des 30 inférences).

| Modèle | Série 1 | Série 2 | Série 3 | Moyenne des médianes |
|---|---:|---:|---:|---:|
| FP32 | 216 ms | 210 ms | 208 ms | **211,33 ms** |
| INT8 dynamique | 210 ms | 208 ms | 213 ms | **210,33 ms** |

Le gain observé est d'environ **0,47 %**, trop faible au regard de la variabilité pour conclure à une accélération significative. L'intérêt principal de l'INT8 dynamique est ici **la réduction de taille**. Une image d'alternariose a été reconnue par les deux modèles ; ce test isolé ne constitue pas une mesure d'accuracy sur iPhone.

**Attention :** ne pas comparer directement les latences Mac et iPhone (matériels et runtimes différents).

## Installation sur ordinateur

Prérequis : **Python 3.10 à 3.12 recommandé** et Git. Depuis la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sous Windows, activer l'environnement avec `.venv\\Scripts\\activate`.

### Lancer Streamlit

```bash
pip install streamlit
streamlit run app.py
```

Ouvrir l'URL locale affichée. Vérifier que `models/model_fp32.onnx`, `models/model_int8_dynamic.onnx` et `models/classes.json` sont présents. Si l'application utilise des chemins ou dépendances supplémentaires, se reporter au code de `app.py`.

### Lancer l'application mobile dans un navigateur

```bash
cd mobile_web
python3 -m http.server 8000 --bind 0.0.0.0
```

- Sur le même ordinateur : `http://localhost:8000`.
- Sur un iPhone connecté au **même Wi-Fi** : trouver l'adresse IP locale du Mac avec `ipconfig getifaddr en0`, puis ouvrir `http://ADRESSE_IP_DU_MAC:8000` dans Safari.
- Autoriser la connexion entrante Python sur le réseau privé si macOS le demande.
- Choisir FP32 ou INT8 dynamique, sélectionner une image et lancer l'analyse. Le benchmark mesure l'inférence uniquement, hors décodage/prétraitement.

**Fichiers requis dans `mobile_web/models/` :** `model_fp32.onnx`, `model_int8_dynamic.onnx`, `classes.json`. La bibliothèque ONNX Runtime Web est chargée depuis un CDN ; une connexion Internet est nécessaire au premier chargement.

> **Sécurité :** le serveur `http.server` est prévu pour un réseau local de confiance, pas pour être exposé directement sur Internet. Pour un professeur distant, il peut cloner le dépôt et lancer l'application sur son propre ordinateur ; il ne peut pas accéder à l'adresse `192.168.x.x` de votre réseau.

## Reproduire l'entraînement et les évaluations

Les images PlantVillage ne sont pas incluses dans le dépôt. Source : [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) ; respecter la licence et les conditions d'attribution. Placer les trois dossiers de classes dans `data/raw/` :

```text
data/raw/Tomato___Early_blight/
data/raw/Tomato___Late_blight/
data/raw/Tomato___healthy/
```

Puis exécuter depuis la racine :

```bash
python -m src.prepare --raw data/raw --out data/split --seed 42
python -m src.train --data data/split --epochs 8 --batch-size 32
python -m src.export_onnx
python -m src.quantize --data data/split/train
python -m src.evaluate --data data/split/test
python -m src.plot_results
```

**Note :** `src.prepare` refuse de remplacer un dossier `data/split` non vide. Sauvegarder ou choisir un autre dossier de sortie avant de recommencer. Le téléchargement et l'entraînement nécessitent plus de temps et de ressources que l'inférence seule.

## Organisation du projet

```text
EdgeVision_MLOps/
├── app.py                     # Interface Streamlit
├── mobile_web/
│   ├── index.html             # Interface Safari + benchmark
│   └── models/                # Modèles pour le navigateur
├── src/                       # Préparation, entraînement, export, évaluation
├── models/                    # Poids et modèles ONNX
├── results/                   # Benchmarks, graphiques, matrices de confusion
├── data/                      # Dataset local (non publié)
├── requirements.txt
└── README.md
```

## Reproductibilité et collaboration

- Les résultats de précision proviennent du **jeu de test Mac (677 images)** ; les temps iPhone proviennent de **mesures manuelles sur Safari**.
- Le modèle déployé sur iPhone est **INT8 dynamique** ; FP32 reste disponible pour la comparaison.
- Les modèles Core ML FP16/FP32 ont aussi été exportés à titre exploratoire, mais **le déploiement iPhone démontré utilise ONNX Runtime Web**, pas Core ML.
- Ne pas publier les données personnelles, mots de passe, fichiers `.env`, environnements virtuels ou les 4 500 images du dataset.

**Équipe :** à compléter avec les noms des deux étudiantes.  
**Cadre :** projet pédagogique MLOps.
