from pathlib import Path
import torch
from torch import nn
from torchvision import models
from .common import labels

def main():
    classes=labels();model=models.mobilenet_v3_small(weights=None)
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(classes))
    model.load_state_dict(torch.load('models/best_model.pt',map_location='cpu',weights_only=True));model.eval()
    torch.onnx.export(model,torch.randn(1,3,224,224),'models/model_fp32.onnx',opset_version=17,input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
    print('Export : models/model_fp32.onnx')
if __name__=='__main__':main()
