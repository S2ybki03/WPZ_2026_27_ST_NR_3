"""Wspólne ustawienia projektu: ścieżki, rozmiar obrazu, normalizacja."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "Images"          # po rozpakowaniu images.tar (Stanford Dogs)
CACHE_DIR = DATA_DIR / "Images_288"       # kopie pomniejszone do 288 px (krótszy bok)
CACHE_SIDE = 288
SPLITS_CSV = DATA_DIR / "splits.csv"      # ścieżka;klasa;podzbiór
CLASSES_TXT = DATA_DIR / "classes.txt"    # nazwy ras, jedna w linii, w kolejności indeksów
RUNS_DIR = ROOT / "runs"                  # wyniki eksperymentów
REPORTS_DIR = ROOT / "reports"            # wykresy z analizy danych i porównania modeli

IMG_SIZE = 224
SEED = 42

# Statystyki ImageNet - takiej normalizacji oczekują wstępnie wytrenowane sieci
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# Rasy rozpoznawane przez model (ustalenia zespołu): kilka najpopularniejszych,
# bez ras brachycefalicznych (płaski pysk - buldogi, mopsy itp.), które według
# artykułu [2] z przeglądu literatury są trudniejsze do rozpoznania.
# klucz = nazwa folderu w Stanford Dogs (bez prefiksu nXXXX-), wartość = nazwa w aplikacji
BREEDS = {
    "Labrador_retriever": "Labrador retriever",
    "golden_retriever": "Golden retriever",
    "German_shepherd": "Owczarek niemiecki",
    "beagle": "Beagle",
    "Siberian_husky": "Husky syberyjski",
    "Rottweiler": "Rottweiler",
    "Yorkshire_terrier": "Yorkshire terrier",
    "Border_collie": "Border collie",
    "Pembroke": "Welsh corgi pembroke",
    "standard_poodle": "Pudel duży",
}


def breed_name(folder: str) -> str:
    """'n02085620-Chihuahua' -> 'Chihuahua', 'n02088094-Afghan_hound' -> 'Afghan hound'."""
    name = folder.split("-", 1)[1].replace("_", " ")
    return name[0].upper() + name[1:]
