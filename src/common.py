from pathlib import Path
import json
import numpy as np
from PIL import Image
from torchvision import transforms

SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def load_image(path):
    return TRANSFORM(Image.open(path).convert('RGB')).unsqueeze(0).numpy().astype(np.float32)

def labels():
    return json.loads(Path('models/classes.json').read_text(encoding='utf-8'))
