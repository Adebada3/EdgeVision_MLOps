import argparse, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import load_image,labels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data/split/test'));ap.add_argument('--models',nargs='+',default=['models/model_fp32.onnx','models/model_int8_dynamic.onnx','models/model_int8_static.onnx']);ap.add_argument('--output',default='results/benchmark_pc.csv');a=ap.parse_args()
    classes=labels(); samples=[]
    for i,c in enumerate(classes):
        for f in sorted((a.data/c).glob('*')):
            if f.suffix.lower() in {'.jpg','.jpeg','.png'}:samples.append((f,i))
    if not samples:raise SystemExit('Aucune image test trouvée')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for file in a.models:
        if not Path(file).exists():print('Absent, ignoré :',file);continue
        session=ort.InferenceSession(file,providers=['CPUExecutionProvider']);input_name=session.get_inputs()[0].name
        y_true=[];y_pred=[]
        for path,target in samples:
            logits=session.run(None,{input_name:load_image(path)})[0]
            y_true.append(target);y_pred.append(int(logits.argmax(axis=1)[0]))
        x=load_image(samples[0][0]);run=lambda:session.run(None,{input_name:x})
        for _ in range(10):run()
        times=[]
        for _ in range(100):
            start=time.perf_counter();run();times.append((time.perf_counter()-start)*1000)
        row=dict(model=Path(file).stem,accuracy=accuracy_score(y_true,y_pred),macro_f1=f1_score(y_true,y_pred,average='macro',zero_division=0),size_mb=Path(file).stat().st_size/1e6,latency_median_ms=float(np.median(times)),latency_p95_ms=float(np.percentile(times,95)),samples=len(samples),device=os.uname().machine if hasattr(os,'uname') else os.name)
        rows.append(row);print(row)
        fig,ax=plt.subplots(figsize=(max(6,len(classes)),max(5,len(classes))))
        ConfusionMatrixDisplay(confusion_matrix(y_true,y_pred,labels=list(range(len(classes)))),display_labels=classes).plot(ax=ax,xticks_rotation=90,colorbar=False)
        fig.tight_layout();fig.savefig('results/confusion_'+Path(file).stem+'.png',dpi=160);plt.close(fig)
    if rows:pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__':main()
