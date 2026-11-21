"""Zbiór danych PyTorch i transformacje obrazu."""
import csv

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import CLASSES_TXT, DATA_DIR, IMG_SIZE, MEAN, STD, SPLITS_CSV


def load_classes():
    return CLASSES_TXT.read_text(encoding="utf-8").splitlines()


def train_transform(img_size=IMG_SIZE):
    """Augmentacja: losowy kadr, odbicie, drobne zmiany kolorów - model nie uczy się
    'na pamięć' konkretnych zdjęć."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(img_size=IMG_SIZE):
    """Bez losowości - używane przy walidacji, teście i w aplikacji."""
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class DogsDataset(Dataset):
    def __init__(self, split, transform):
        self.items = []
        with open(SPLITS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row["split"] == split:
                    self.items.append((DATA_DIR / row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), label
