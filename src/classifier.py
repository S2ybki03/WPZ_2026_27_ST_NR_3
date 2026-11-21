"""Interfejs modelu dla reszty aplikacji (GUI, logika, testy).

Użycie:
    from classifier import DogBreedClassifier

    clf = DogBreedClassifier("models/dog_breed.pt")   # raz, przy starcie aplikacji
    wyniki = clf.predict("pies.jpg", top_k=5)         # ścieżka, obiekt PIL.Image lub bajty
    for w in wyniki:
        print(w.breed, f"{w.percent:.1f} %")

predict() zwraca listę obiektów Prediction posortowaną malejąco wg prawdopodobieństwa.
Zdjęcie może mieć dowolny rozmiar i format obsługiwany przez Pillow - skalowanie
i normalizację robi klasa sama.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from dataset import eval_transform
from models import build_model

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "dog_breed.pt"


@dataclass
class Prediction:
    breed: str          # nazwa rasy, np. "Golden retriever"
    probability: float  # 0..1

    @property
    def percent(self):
        return self.probability * 100


class DogBreedClassifier:
    def __init__(self, checkpoint=DEFAULT_MODEL, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.model_name = ckpt["model"]
        self.model, _ = build_model(self.model_name, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(ckpt.get("img_size", 224))

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

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        x = self.transform(self.load_image(image)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x).float(), dim=1)[0]
        p, idx = probs.topk(min(top_k, len(self.classes)))
        return [Prediction(self.classes[i], float(v)) for v, i in zip(p.tolist(), idx.tolist())]


if __name__ == "__main__":   # szybki test z wiersza poleceń: python src/classifier.py pies.jpg
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Użycie: python src/classifier.py <zdjęcie> [model.pt]")
    clf = DogBreedClassifier(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL)
    print(f"Model: {clf.model_name}, urządzenie: {clf.device}")
    for r in clf.predict(sys.argv[1]):
        print(f"{r.percent:6.2f} %  {r.breed}")
