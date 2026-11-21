"""Zdjęcia do prezentacji: po 2 zdjęcia każdej rasy ze zbioru TESTOWEGO (model ich nie widział)
oraz kilka ras spoza listy - do pokazania ostrzeżenia „Model nie jest pewny”.

    python tools/make_demo_set.py
Wynik: demo_zdjecia/ (oryginalna rozdzielczość z data/Images).
"""
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import DogBreedClassifier
from config import BREEDS, DATA_DIR, IMAGES_DIR, SPLITS_CSV

OUT = ROOT / "demo_zdjecia"
PER_BREED = 2
OTHER = {"pug": "mops", "French_bulldog": "buldog francuski", "Pomeranian": "szpic miniaturowy",
         "Chihuahua": "chihuahua", "Dalmatian": "dalmatynczyk"}


def safe(name):
    return name.replace(" ", "_").replace("ż", "z").replace("ó", "o")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    clf = DogBreedClassifier()
    classes = clf.classes

    with open(SPLITS_CSV, encoding="utf-8") as f:
        test = [r for r in csv.DictReader(f, delimiter=";") if r["split"] == "test"]

    picked = {c: 0 for c in classes}
    for r in test:
        breed = classes[int(r["label"])]
        if picked[breed] >= PER_BREED:
            continue
        orig = IMAGES_DIR / Path(r["path"]).relative_to("Images_288")
        res = clf.predict(orig, top_k=1)[0]
        # do prezentacji wyraźne zdjęcia: poprawnie rozpoznane z pewnością >= 80 %
        if res.breed == breed and res.probability >= 0.80:
            picked[breed] += 1
            shutil.copy(orig, OUT / f"{safe(breed)}_{picked[breed]}.jpg")
            print(f"{breed:22s} {res.percent:5.1f} %")

    folders = {d.name.split("-", 1)[1]: d for d in IMAGES_DIR.iterdir() if d.is_dir()}
    for key, pl in OTHER.items():
        if key not in folders:
            continue
        for img in sorted(folders[key].glob("*.jpg"))[:10]:
            res = clf.predict(img, top_k=1)[0]
            if res.probability < 0.5:   # przykład, na którym widać ostrzeżenie
                shutil.copy(img, OUT / f"spoza_listy_{safe(pl)}.jpg")
                print(f"spoza listy: {pl:18s} -> {res.breed} {res.percent:.1f} %")
                break
    print(f"Zapisano {len(list(OUT.iterdir()))} zdjęć w {OUT}")


if __name__ == "__main__":
    main()
