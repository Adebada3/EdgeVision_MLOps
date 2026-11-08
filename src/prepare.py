import argparse, random, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,default=Path('data/raw'));ap.add_argument('--out',type=Path,default=Path('data/split'));ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit('Dossier de sortie non vide : supprimez-le explicitement avant de recommencer.')
    rng=random.Random(a.seed)
    counts={}
    for folder in sorted(a.raw.iterdir()):
        if not folder.is_dir() or not folder.name.lower().startswith('tomato'): continue
        images=[x for x in folder.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png'}]
        rng.shuffle(images);n=len(images)
        if n<10: raise SystemExit(f'Pas assez d’images pour {folder.name}: {n}')
        ntrain=int(n*.70); nval=int(n*.15)
        for split, subset in [('train',images[:ntrain]),('val',images[ntrain:ntrain+nval]),('test',images[ntrain+nval:])]:
            dest=a.out/split/folder.name;dest.mkdir(parents=True,exist_ok=True)
            for f in subset: shutil.copy2(f,dest/f.name)
        counts[folder.name]=n
    if len(counts)<2: raise SystemExit('Il faut au moins deux classes Tomato dans data/raw')
    print('Répartition 70/15/15 :',counts)
if __name__=='__main__': main()
