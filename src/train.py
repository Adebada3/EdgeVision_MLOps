import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from .common import TRANSFORM

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data/split'));ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    train=datasets.ImageFolder(a.data/'train',transform=TRANSFORM);val=datasets.ImageFolder(a.data/'val',transform=TRANSFORM)
    assert train.classes==val.classes,'Classes différentes entre train et val'
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0);va=DataLoader(val,batch_size=a.batch_size,num_workers=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    net.classifier[3]=nn.Linear(net.classifier[3].in_features,len(train.classes));net.to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4);loss_fn=nn.CrossEntropyLoss()
    Path('models').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
    Path('models/classes.json').write_text(json.dumps(train.classes,ensure_ascii=False,indent=2),encoding='utf-8')
    history=[];best=-1.
    for epoch in range(a.epochs):
        stats=[]
        for training,loader in [(True,tr),(False,va)]:
            net.train(training);total=correct=0;loss_sum=0.
            for x,y in loader:
                x=x.to(device);y=y.to(device)
                with torch.set_grad_enabled(training):
                    logits=net(x);loss=loss_fn(logits,y)
                    if training: opt.zero_grad();loss.backward();opt.step()
                loss_sum+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();total+=len(y)
            stats.append((loss_sum/total,correct/total))
        history.append(dict(epoch=epoch+1,train_loss=stats[0][0],train_accuracy=stats[0][1],val_loss=stats[1][0],val_accuracy=stats[1][1]))
        if stats[1][1]>best:best=stats[1][1];torch.save(net.cpu().state_dict(),'models/best_model.pt');net.to(device)
        print(history[-1])
    pd.DataFrame(history).to_csv('results/training_history.csv',index=False)
    print('Meilleure accuracy validation :',best)
if __name__=='__main__':main()
