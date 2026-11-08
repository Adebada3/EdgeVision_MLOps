
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image

from src.common import TRANSFORM, labels


# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------

st.set_page_config(
    page_title="EdgeVision | Diagnostic des tomates",
    page_icon="🌿",
    layout="wide",
)

MODEL_DIR = Path("models")

MODELS = {
    "FP32 — Modèle original": MODEL_DIR / "model_fp32.onnx",
    "INT8 — Quantification dynamique": MODEL_DIR / "model_int8_dynamic.onnx",
    "INT8 — Quantification statique": MODEL_DIR / "model_int8_static.onnx",
}

CLASSES = labels()

CLASS_NAMES = {
    "Tomato___healthy": "Feuille saine",
    "Tomato___Early_blight": "Alternariose (Early Blight)",
    "Tomato___Late_blight": "Mildiou (Late Blight)",
}


# --------------------------------------------------
# Chargement des modèles ONNX
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


# --------------------------------------------------
# Prédiction
# --------------------------------------------------

def predict(image, model_path):
    session = load_model(str(model_path))

    # Même prétraitement que src.common.load_image()
    tensor = (
        TRANSFORM(image.convert("RGB"))
        .unsqueeze(0)
        .numpy()
        .astype(np.float32)
    )

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()

    logits = session.run(
        None,
        {input_name: tensor},
    )[0][0]

    latency_ms = (time.perf_counter() - start) * 1000

    # Softmax : conversion des logits en probabilités
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)

    return probabilities, latency_ms


# --------------------------------------------------
# En-tête
# --------------------------------------------------

st.title("🌿 EdgeVision")
st.subheader(
    "Détection intelligente des maladies des feuilles de tomates"
)

st.markdown(
    """
    **EdgeVision** est une application de classification d'images
    utilisant un réseau de neurones **MobileNetV3 Small**.

    Le modèle distingue trois catégories :
    - 🍃 Feuille saine
    - 🟤 Alternariose (Early Blight)
    - 🍂 Mildiou (Late Blight)

    Ce projet étudie également l'impact de la quantification
    sur la précision, la taille et la vitesse d'inférence.
    """
)

st.divider()


# --------------------------------------------------
# Barre latérale
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    selected_model = st.selectbox(
        "Choisir un modèle",
        list(MODELS.keys()),
    )

    st.caption(
        "Les prédictions sont exécutées localement "
        "sur l'ordinateur qui héberge Streamlit."
    )

    st.divider()

    st.header("📊 Performances mesurées")

    benchmark_path = Path("results/benchmark_pc.csv")

    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)

        st.dataframe(
            benchmark[
                [
                    "model",
                    "accuracy",
                    "size_mb",
                    "latency_median_ms",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Mesures effectuées sur 677 images de test "
            "avec ONNX Runtime sur Mac ARM64."
        )
    else:
        st.info("Les benchmarks ne sont pas encore disponibles.")


# --------------------------------------------------
# Importation de l'image
# --------------------------------------------------

st.header("📷 Analyser une feuille de tomate")

uploaded_file = st.file_uploader(
    "Téléverser une image de feuille",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG et PNG.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Impossible de lire l'image : {exc}")
        st.stop()

    col_image, col_result = st.columns(2, gap="large")

    with col_image:
        st.image(
            image,
            caption="Image sélectionnée",
            use_container_width=True,
        )

    with col_result:
        st.subheader("🔍 Résultat de l'analyse")

        if st.button(
            "Analyser l'image",
            type="primary",
            use_container_width=True,
        ):
            model_path = MODELS[selected_model]

            if not model_path.exists():
                st.error(
                    f"Fichier modèle introuvable : {model_path}"
                )
                st.stop()

            try:
                with st.spinner("Analyse en cours..."):
                    probabilities, latency_ms = predict(
                        image,
                        model_path,
                    )
            except Exception as exc:
                st.error(f"Erreur pendant l'inférence : {exc}")
                st.stop()

            best_index = int(np.argmax(probabilities))
            predicted_class = CLASSES[best_index]
            confidence = float(probabilities[best_index])

            display_name = CLASS_NAMES.get(
                predicted_class,
                predicted_class.replace("Tomato___", ""),
            )

            st.success("Analyse terminée !")

            st.metric(
                "Classe prédite",
                display_name,
            )

            st.metric(
                "Score de confiance",
                f"{confidence * 100:.2f} %",
            )

            st.metric(
                "Temps d'inférence",
                f"{latency_ms:.2f} ms",
            )

            st.subheader("Probabilités par catégorie")

            for class_name, probability in zip(
                CLASSES,
                probabilities,
            ):
                clean_name = CLASS_NAMES.get(
                    class_name,
                    class_name.replace("Tomato___", ""),
                )

                st.write(
                    f"**{clean_name}** : "
                    f"{float(probability) * 100:.2f} %"
                )

                st.progress(float(probability))

            if selected_model.startswith("INT8 — Quantification statique"):
                st.warning(
                    "Attention : ce modèle a obtenu seulement "
                    "42,39 % d'accuracy sur le jeu de test. "
                    "Ses prédictions sont donc peu fiables."
                )

            st.caption(
                "Le score de confiance est une probabilité "
                "softmax, pas une garantie de diagnostic."
            )

else:
    st.info(
        "Importe une image de feuille de tomate "
        "pour commencer l'analyse."
    )


# --------------------------------------------------
# Pied de page
# --------------------------------------------------

st.divider()

st.caption(
    "EdgeVision | Projet MLOps — MobileNetV3 Small, "
    "PyTorch, ONNX Runtime et quantification INT8."
)
