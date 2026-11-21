"""Sprawdza, czy DogBreedClassifier daje tę samą dokładność co evaluate.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from classifier import DogBreedClassifier
from dataset import DogsDataset, eval_transform

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
clf = DogBreedClassifier(ckpt)
ds = DogsDataset("test", eval_transform())
step = len(ds) // n
ok = conf = 0
for i in range(0, step * n, step):
    path, label = ds.items[i]
    r = clf.predict(path, top_k=1)[0]
    ok += r.breed == clf.classes[label]
    conf += r.probability
# to samo zdjęcie przez Dataset (jak w evaluate.py), bez autocast
x, label = ds[0]
with torch.inference_mode():
    p = torch.softmax(clf.model(x.unsqueeze(0).to(clf.device)), 1)[0]
print(f"classifier: top-1 {ok / n:.3f}, średnia pewność {conf / n:.3f}")
print(f"dataset[0]: prawdziwa {clf.classes[label]}, model {clf.classes[p.argmax()]} {p.max():.3f}")
print("classifier na dataset[0]:", clf.predict(ds.items[0][0], top_k=1))
