from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    Path('results').mkdir(exist_ok=True)
    history=Path('results/training_history.csv')
    if history.exists():
        df=pd.read_csv(history);fig,axes=plt.subplots(1,2,figsize=(11,4))
        for metric,ax in [('loss',axes[0]),('accuracy',axes[1])]:
            ax.plot(df.epoch,df['train_'+metric],label='Train');ax.plot(df.epoch,df['val_'+metric],label='Validation');ax.set_title(metric);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig('results/training_curve.png',dpi=160);plt.close(fig)
    path=Path('results/benchmark_pc.csv')
    if path.exists():
        df=pd.read_csv(path);fig,axes=plt.subplots(1,3,figsize=(14,4))
        for col,ax in zip(['accuracy','size_mb','latency_median_ms'],axes):
            ax.bar(df.model,df[col]);ax.set_title(col);ax.tick_params(axis='x',rotation=35)
        fig.tight_layout();fig.savefig('results/comparison.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
