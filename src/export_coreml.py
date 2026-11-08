
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct


MODEL_DIR = Path("models")
CLASSES = json.loads(
    (MODEL_DIR / "classes.json").read_text(encoding="utf-8")
)


# Recréer exactement l'architecture entraînée
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(CLASSES),
)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
)
model.eval()


# Prétraitement intégré dans le modèle Core ML
class EdgeVisionModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).reshape(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).reshape(1, 3, 1, 1),
        )

    def forward(self, image):
        normalized = (image - self.mean) / self.std
        return self.classifier(normalized)


wrapped_model = EdgeVisionModel(model).eval()

example_input = torch.rand(1, 3, 224, 224)

with torch.no_grad():
    traced_model = torch.jit.trace(
        wrapped_model,
        example_input,
    )

traced_model.eval()


def export_coreml(name, precision):
    print(f"Conversion de {name}...")

    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.ImageType(
                name="image",
                shape=example_input.shape,
                color_layout=ct.colorlayout.RGB,
                scale=1.0 / 255.0,
            )
        ],
        outputs=[
            ct.TensorType(name="logits")
        ],
        compute_precision=precision,
    )

    output_path = MODEL_DIR / f"{name}.mlpackage"
    coreml_model.save(str(output_path))

    print(f"Modèle enregistré : {output_path}")


export_coreml(
    "EdgeVision_FP32",
    ct.precision.FLOAT32,
)

export_coreml(
    "EdgeVision_FP16",
    ct.precision.FLOAT16,
)

print("Conversion Core ML terminée !")
print("Classes :", CLASSES)
