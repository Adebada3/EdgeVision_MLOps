
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import coremltools as ct
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.common import SIZE, labels


DATA_DIR = Path("data/split/val")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = labels()

MODELS = {
    "coreml_fp32": MODEL_DIR / "EdgeVision_FP32.mlpackage",
    "coreml_fp16": MODEL_DIR / "EdgeVision_FP16.mlpackage",
}


def prepare_image(path):
    # Même taille et interpolation que torchvision Resize
    with Image.open(path) as img:
        return img.convert("RGB").resize(
            (SIZE, SIZE),
            resample=Image.Resampling.BILINEAR,
        )


def evaluate_model(model_name, model_path, samples):
    print(f"\nÉvaluation : {model_name}")

    model = ct.models.MLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    input_name = model.get_spec().description.input[0].name
    output_name = model.get_spec().description.output[0].name

    y_true = []
    y_pred = []
    times = []

    for index, (path, target) in enumerate(samples, start=1):
        image = prepare_image(path)

        start = time.perf_counter()
        result = model.predict({input_name: image})
        elapsed_ms = (time.perf_counter() - start) * 1000

        logits = np.asarray(result[output_name]).reshape(-1)

        prediction = int(np.argmax(logits))

        y_true.append(target)
        y_pred.append(prediction)
        times.append(elapsed_ms)

        if index % 100 == 0:
            print(f"  {index}/{len(samples)} images")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"Latence médiane : {np.median(times):.3f} ms")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "latency_median_ms": float(np.median(times)),
        "samples": len(samples),
        "device": "Mac ARM64 - Core ML CPU",
    }


def main():
    samples = []

    for class_index, class_name in enumerate(CLASSES):
        folder = DATA_DIR / class_name

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, class_index))

    if not samples:
        raise RuntimeError("Aucune image de validation trouvée.")

    print(f"Images de validation : {len(samples)}")
    print(f"Ordre des classes : {CLASSES}")

    rows = []

    for name, path in MODELS.items():
        rows.append(evaluate_model(name, path, samples))

    output = RESULTS_DIR / "benchmark_coreml_val.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"\nRésultats enregistrés : {output}")


if __name__ == "__main__":
    main()
