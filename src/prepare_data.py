"""Przygotowanie zbioru Stanford Dogs.

1. Rozpakowuje data/images.tar (jeśli trzeba).
2. Dzieli zdjęcia na train / val / test (70 / 15 / 15 %, stratyfikacja po rasie).
3. Zapisuje data/splits.csv i data/classes.txt.
4. Robi krótką analizę danych (EDA): liczność klas, rozmiary zdjęć -> reports/.

Uruchomienie:  python src/prepare_data.py
"""
import csv
import random
import tarfile
from collections import Counter
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from config import (BREEDS, CACHE_DIR, CACHE_SIDE, CLASSES_TXT, DATA_DIR, IMAGES_DIR,
                    REPORTS_DIR, SEED, SPLITS_CSV)


def extract_if_needed():
    if IMAGES_DIR.exists():
        return
    tar_path = DATA_DIR / "images.tar"
    if not tar_path.exists():
        raise SystemExit(
            f"Brak {tar_path}. Pobierz: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar")
    print("Rozpakowywanie images.tar ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_DIR, filter="data")


def check_images(paths):
    """Usuwa pliki, których nie da się otworzyć; zwraca rozmiary pozostałych."""
    good, sizes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append(im.size)
            good.append(p)
        except Exception as e:  # uszkodzony plik
            print(f"  pomijam {p.name}: {e}")
    return good, sizes


def resize_one(src):
    """Pomniejsza zdjęcie tak, by krótszy bok miał CACHE_SIDE px. Dekodowanie dużych JPEG
    w każdej epoce było wąskim gardłem (GPU czekało na dane), więc robimy to raz."""
    dst = CACHE_DIR / src.parent.name / src.name
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        scale = CACHE_SIDE / min(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
        im.save(dst, quality=95)
    return dst


def main():
    extract_if_needed()
    REPORTS_DIR.mkdir(exist_ok=True)

    all_folders = sorted(d for d in IMAGES_DIR.iterdir() if d.is_dir())
    print(f"Ras w całym zbiorze Stanford Dogs: {len(all_folders)}")
    # tylko rasy wybrane w config.BREEDS (kolejność jak w słowniku)
    by_key = {d.name.split("-", 1)[1]: d for d in all_folders}
    missing = [k for k in BREEDS if k not in by_key]
    if missing:
        raise SystemExit(f"Nie ma w zbiorze: {missing}")
    folders = [by_key[k] for k in BREEDS]
    classes = list(BREEDS.values())
    print(f"Ras: {len(classes)}")

    paths, labels = [], []
    for idx, folder in enumerate(folders):
        for img in sorted(folder.glob("*.jpg")):
            paths.append(img)
            labels.append(idx)
    print(f"Zdjęć: {len(paths)} - sprawdzanie plików ...")

    good, sizes = check_images(paths)
    keep = set(good)
    labels = [l for p, l in zip(paths, labels) if p in keep]
    paths = good

    print(f"Pomniejszanie do {CACHE_SIDE} px -> {CACHE_DIR.name} ...")
    with Pool() as pool:
        cached = pool.map(resize_one, paths, chunksize=64)
    by_orig = dict(zip(paths, cached))

    # 70 / 15 / 15 ze stratyfikacją - każda rasa w tej samej proporcji w każdym podzbiorze
    p_train, p_rest, y_train, y_rest = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    p_val, p_test, y_val, y_test = train_test_split(
        p_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)

    with open(SPLITS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["path", "label", "split"])
        for split, ps, ys in (("train", p_train, y_train), ("val", p_val, y_val),
                              ("test", p_test, y_test)):
            for p, y in zip(ps, ys):
                w.writerow([by_orig[p].relative_to(DATA_DIR).as_posix(), y, split])
    CLASSES_TXT.write_text("\n".join(classes), encoding="utf-8")
    print(f"train {len(p_train)} | val {len(p_val)} | test {len(p_test)}")

    # ---------- EDA ----------
    counts = Counter(labels)
    per_class = [counts[i] for i in range(len(classes))]
    order = sorted(range(len(classes)), key=lambda i: per_class[i])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([classes[i] for i in order], [per_class[i] for i in order], color="#4C72B0")
    for k, i in enumerate(order):
        ax.text(per_class[i] + 2, k, str(per_class[i]), va="center", fontsize=8)
    ax.set_xlabel("Liczba zdjęć")
    ax.set_title("Liczba zdjęć na rasę (wybrane rasy ze Stanford Dogs)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_class_counts.png", dpi=150)
    plt.close(fig)

    ws, hs = zip(*sizes)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ws, hs, s=2, alpha=0.3)
    ax.set_xlabel("Szerokość [px]")
    ax.set_ylabel("Wysokość [px]")
    ax.set_title("Rozmiary zdjęć")
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_image_sizes.png", dpi=150)
    plt.close(fig)

    # siatka przykładowych zdjęć
    rng = random.Random(SEED)
    sample = rng.sample(range(len(paths)), 16)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, i in zip(axes.flat, sample):
        with Image.open(paths[i]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(classes[labels[i]], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_samples.png", dpi=120)
    plt.close(fig)

    small = sum(1 for w_, h_ in sizes if min(w_, h_) < 224)
    stats = (
        f"Ras: {len(classes)}\n"
        f"Zdjęć: {len(paths)}\n"
        f"Zdjęć na rasę: min {min(per_class)} ({classes[order[0]]}), "
        f"max {max(per_class)} ({classes[order[-1]]}), "
        f"średnio {sum(per_class) / len(per_class):.1f}\n"
        f"Szerokość: min {min(ws)}, max {max(ws)}, średnio {sum(ws) / len(ws):.0f} px\n"
        f"Wysokość: min {min(hs)}, max {max(hs)}, średnio {sum(hs) / len(hs):.0f} px\n"
        f"Zdjęć z krótszym bokiem < 224 px: {small}\n"
        f"Podział: train {len(p_train)}, val {len(p_val)}, test {len(p_test)}\n"
    )
    (REPORTS_DIR / "eda_stats.txt").write_text(stats, encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
