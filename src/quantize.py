import argparse
from pathlib import Path
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, quantize_dynamic, quantize_static
from .common import load_image

class Reader(CalibrationDataReader):
    def __init__(self,paths):self.paths=iter(paths)
    def get_next(self):
        try:return {'input':load_image(next(self.paths))}
        except StopIteration:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data/split/train'));ap.add_argument('--calibration-count',type=int,default=100);a=ap.parse_args()
    src='models/model_fp32.onnx'
    paths=sorted(x for x in a.data.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png'})[:a.calibration_count]
    if not paths:raise SystemExit('Pas d’images de calibration')
    # Quantification dynamique : surtout pertinente pour Gemm/MatMul ; MobileNet est principalement convolutionnel.
    quantize_dynamic(src,'models/model_int8_dynamic.onnx',weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])
    print('Quantification dynamique terminée')
    quantize_static(src,'models/model_int8_static.onnx',Reader(paths),quant_format=QuantFormat.QDQ,activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,op_types_to_quantize=['Conv','MatMul','Gemm'],per_channel=False)
    print('Quantification statique terminée')
if __name__=='__main__':main()
