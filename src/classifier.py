"""Klasyfikator ras psów - wersja bez PyTorch (tylko numpy + Pillow).

Sieć ResNet-50 jest liczona bezpośrednio w numpy, dlatego aplikacja działa na
komputerach, na których PyTorch się nie uruchamia (np. brak bibliotek Visual C++).
Wagi: models/dog_breed.npz (powstały z dog_breed.pt skryptem tools/export_numpy.py,
BatchNorm wtopiony w konwolucje - wynik identyczny jak w PyTorch).

    from classifier import DogBreedClassifier
    clf = DogBreedClassifier()
    for w in clf.predict("pies.jpg", top_k=5):
        print(f"{w.breed}: {w.percent:.1f} %")
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageOps

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "dog_breed.npz"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BLOCKS = (3, 4, 6, 3)   # liczba bloków w warstwach ResNet-50


@dataclass
class Prediction:
    breed: str          # nazwa rasy, np. "Golden retriever"
    probability: float  # 0..1

    @property
    def percent(self):
        return self.probability * 100


# ---------- operacje sieci (układ danych: wysokość x szerokość x kanały) ----------
def conv(x, w, b, k, stride=1, pad=0, relu=True):
    """Konwolucja jako mnożenie macierzy (im2col). w: (k*k*C_in, C_out)."""
    if pad:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    h, wd, c = x.shape
    oh, ow = (h - k) // stride + 1, (wd - k) // stride + 1
    if k == 1:
        cols = x[::stride, ::stride].reshape(-1, c)
    else:
        s0, s1, s2 = x.strides
        cols = as_strided(x, (oh, ow, k, k, c), (s0 * stride, s1 * stride, s0, s1, s2))
        cols = cols.reshape(oh * ow, k * k * c)
    y = cols @ w
    y += b
    if relu:
        np.maximum(y, 0, out=y)
    return y.reshape(oh, ow, -1)


def maxpool3x3s2(x):
    x = np.pad(x, ((1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
    h, w, c = x.shape
    oh, ow = (h - 3) // 2 + 1, (w - 3) // 2 + 1
    s0, s1, s2 = x.strides
    win = as_strided(x, (oh, ow, 3, 3, c), (s0 * 2, s1 * 2, s0, s1, s2))
    return win.max(axis=(2, 3))


class DogBreedClassifier:
    def __init__(self, checkpoint=DEFAULT_MODEL, device=None):
        checkpoint = Path(checkpoint)
        if checkpoint.suffix == ".pt":
            npz = checkpoint.with_suffix(".npz")
            if not npz.exists():
                raise ValueError("Ta wersja aplikacji nie używa PyTorch - wybierz plik .npz "
                                 "(konwersja: python tools/export_numpy.py model.pt model.npz)")
            checkpoint = npz
        with np.load(checkpoint) as f:
            self.p = {k: f[k] for k in f.files if k != "meta"}
            meta = json.loads(bytes(f["meta"]).decode("utf-8"))
        self.classes = meta["classes"]
        self.model_name = meta["model"]
        self.img_size = meta.get("img_size", 224)
        self.device = SimpleNamespace(type="cpu")

    # ---------- przygotowanie zdjęcia (jak torchvision: Resize 256, CenterCrop 224) ----------
    @staticmethod
    def load_image(image):
        """Ścieżka / bajty / PIL.Image -> RGB PIL.Image (z uwzględnieniem obrotu z EXIF)."""
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)
        img = ImageOps.exif_transpose(img)   # zdjęcia z telefonu bywają obrócone
        return img.convert("RGB")

    def preprocess(self, img):
        size = int(self.img_size * 256 / 224)
        w, h = img.size
        if w <= h:
            nw, nh = size, int(size * h / w)
        else:
            nw, nh = int(size * w / h), size
        img = img.resize((nw, nh), Image.BILINEAR)
        s = self.img_size
        left, top = int(round((nw - s) / 2.0)), int(round((nh - s) / 2.0))
        img = img.crop((left, top, left + s, top + s))
        x = np.asarray(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray((x - MEAN) / STD)

    # ---------- sieć ResNet-50 ----------
    def forward(self, x):
        p = self.p
        x = conv(x, p["stem.w"], p["stem.b"], 7, stride=2, pad=3)
        x = maxpool3x3s2(x)
        for li, n in enumerate(BLOCKS, start=1):
            for bi in range(n):
                q = f"layer{li}.{bi}"
                stride = 2 if (bi == 0 and li > 1) else 1
                y = conv(x, p[q + ".c1.w"], p[q + ".c1.b"], 1)
                y = conv(y, p[q + ".c2.w"], p[q + ".c2.b"], 3, stride=stride, pad=1)
                y = conv(y, p[q + ".c3.w"], p[q + ".c3.b"], 1, relu=False)
                if q + ".ds.w" in p:
                    x = conv(x, p[q + ".ds.w"], p[q + ".ds.b"], 1, stride=stride, relu=False)
                x = np.maximum(y + x, 0)
        feat = x.mean(axis=(0, 1))
        return feat @ p["fc.w"] + p["fc.b"]

    def predict(self, image, top_k=5):
        logits = self.forward(self.preprocess(self.load_image(image))).astype(np.float64)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = np.argsort(-probs)[:min(top_k, len(self.classes))]
        return [Prediction(self.classes[i], float(probs[i])) for i in idx]


if __name__ == "__main__":   # szybki test z wiersza poleceń: python src/classifier.py pies.jpg
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Użycie: python src/classifier.py <zdjęcie> [model.npz]")
    clf = DogBreedClassifier(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL)
    print(f"Model: {clf.model_name} (numpy, CPU)")
    for r in clf.predict(sys.argv[1]):
        print(f"{r.percent:6.2f} %  {r.breed}")
