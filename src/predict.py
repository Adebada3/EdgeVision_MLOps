import argparse
import numpy as np
import onnxruntime as ort
from .common import load_image,labels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/model_int8_static.onnx');ap.add_argument('--image',required=True);a=ap.parse_args()
    session=ort.InferenceSession(a.model,providers=['CPUExecutionProvider'])
    scores=session.run(None,{session.get_inputs()[0].name:load_image(a.image)})[0][0]
    exp=np.exp(scores-np.max(scores));probs=exp/exp.sum();index=int(probs.argmax())
    print(f'Prédiction : {labels()[index]} | confiance softmax : {probs[index]:.2%}')
    print('La confiance softmax ne représente pas une probabilité calibrée de diagnostic.')
if __name__=='__main__':main()
