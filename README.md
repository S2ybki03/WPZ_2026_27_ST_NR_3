# Rozpoznawanie ras psów z wykorzystaniem SI

Aplikacja z graficznym interfejsem: użytkownik wczytuje zdjęcie psa z dysku, a model
sieci neuronowej wyświetla 5 najbardziej prawdopodobnych ras z wynikiem procentowym.

Model rozpoznaje **10 popularnych ras** (bez ras o płaskim pysku - buldogów, mopsów itp.):
Beagle, Border collie, Golden retriever, Husky syberyjski, Labrador retriever,
Owczarek niemiecki, Pudel duży, Rottweiler, Welsh corgi pembroke, Yorkshire terrier.

## Uruchomienie aplikacji

Wymagany Python 3.10+ (tkinter jest w standardowej instalacji).
```
pip install torch torchvision pillow
python src/gui.py
```
Karta graficzna nie jest potrzebna - na procesorze analiza zdjęcia trwa ok. 60 ms.
Aplikacja wczytuje model z `models/dog_breed.pt`; inny plik można wybrać przyciskiem
„Zmień model…” albo podać w wierszu poleceń: `python src/gui.py runs/mobilenet_v2/best.pt`.

## Struktura

```
src/
  gui.py            interfejs graficzny (tkinter)
  classifier.py     klasa DogBreedClassifier - jedyny punkt styku GUI z modelem
  config.py         ścieżki, stałe, lista ras (BREEDS)
  prepare_data.py   wybór ras, podział train/val/test, analiza danych (EDA)
  dataset.py        Dataset PyTorch + transformacje (augmentacja dla treningu)
  models.py         4 architektury: simple_cnn, mobilenet_v2, efficientnet_b0, resnet50
  train.py          trening jednego modelu
  evaluate.py       ocena na zbiorze testowym: metryki, macierz pomyłek, F1 dla ras
  compare.py        tabela porównawcza + wybór modelu -> models/dog_breed.pt
models/dog_breed.pt wybrany, wytrenowany model (ResNet-50)
runs/<model>/       wyniki każdego eksperymentu (wykresy, metryki)
reports/            wykresy z EDA, porównanie modeli, zrzuty ekranu GUI
tools/              skrypty diagnostyczne i test GUI
run_all.ps1         pełna seria eksperymentów jednym poleceniem
```

## Działanie aplikacji

1. Użytkownik uruchamia aplikację - model wczytuje się w tle (okno nie zawiesza się).
2. Wybiera zdjęcie psa z dysku (przycisk, kliknięcie w pole podglądu lub Ctrl+O).
3. Program przygotowuje zdjęcie: obrót wg EXIF, konwersja do RGB, skalowanie do 256 px,
   wycięcie środka 224 x 224, normalizacja.
4. Model zwraca prawdopodobieństwa dla 10 ras.
5. Aplikacja pokazuje 5 najbardziej prawdopodobnych ras z paskami i procentami.
   Gdy najwyższy wynik jest poniżej 50 %, wyświetla ostrzeżenie, że model nie jest pewny
   (np. rasa spoza listy, mieszaniec, brak psa na zdjęciu).

Model można też użyć bez GUI:
```python
from classifier import DogBreedClassifier

clf = DogBreedClassifier()                 # models/dog_breed.pt
for w in clf.predict("pies.jpg", top_k=5): # ścieżka, PIL.Image albo bajty
    print(f"{w.breed}: {w.percent:.1f} %")
```

## Dane

Zbiór **Stanford Dogs** (20 580 zdjęć, 120 ras). Wybrano 10 ras - łącznie 1666 zdjęć,
od 150 (Golden retriever) do 195 (Beagle) na rasę, więc klasy są zrównoważone.
Podział ze stratyfikacją: 70 % trening (1166), 15 % walidacja (250), 15 % test (250).
Zdjęcia mają rozmiary od 134 px do 2272 px; raz pomniejszamy je do 288 px (krótszy bok),
co przyspiesza trening. Wykresy: `reports/eda_*.png`, statystyki: `reports/eda_stats.txt`.

Rasy o płaskim pysku pominięto, bo według artykułu [2] z przeglądu literatury
skuteczność rozpoznawania jest dla nich wyraźnie niższa.

## Metoda

1. **simple_cnn** - własna sieć konwolucyjna uczona od zera (punkt odniesienia):
   5 bloków (2 x konwolucja 3x3 + BatchNorm + ReLU, MaxPool), uśrednianie globalne,
   dropout, warstwa wyjściowa. 80 epok.
2. **mobilenet_v2**, **efficientnet_b0**, **resnet50** - transfer learning: wagi wytrenowane
   na ImageNet, nowa warstwa wyjściowa na 10 ras.
   - faza 1 (3 epoki): zamrożony szkielet sieci, uczymy tylko warstwę wyjściową,
   - faza 2 (12 epok): fine-tuning całej sieci z mniejszym współczynnikiem uczenia.
   MobileNetV2 włączono do porównania, bo opisuje go artykuł [1] z przeglądu literatury.
3. Wspólne: optymalizator AdamW, harmonogram cosinusowy, label smoothing 0.1,
   mixed precision (fp16), augmentacja (losowy kadr, odbicie lustrzane, zmiana
   jasności/kontrastu/nasycenia), batch 32. Zapisywany jest model z najwyższą dokładnością
   na walidacji; zbiór testowy używany jest tylko raz, do końcowej oceny.

## Wyniki (zbiór testowy, 250 zdjęć)

| Model | Parametry [mln] | Rozmiar [MB] | Top-1 | F1 macro | Błędy | Czas CPU [ms] | Czas GPU [ms] |
|---|---|---|---|---|---|---|---|
| **ResNet-50** | 23.5 | 90.0 | **99.2 %** | 0.992 | 2 | 63 | 9.6 |
| EfficientNet-B0 | 4.0 | 15.6 | 98.0 % | 0.980 | 5 | 30 | 11.6 |
| MobileNetV2 | 2.2 | 8.7 | 96.8 % | 0.968 | 8 | 24 | 7.2 |
| Własna CNN (od zera) | 4.7 | 18.0 | 71.6 % | 0.723 | 71 | 33 | 1.5 |

Czas = klasyfikacja jednego zdjęcia (CPU: Intel i7-13700F, GPU: RTX 3060).
Szczegóły: `reports/comparison.md`, `runs/<model>/` (krzywe uczenia, macierz pomyłek,
F1 dla każdej rasy, najczęściej mylone pary).

Wnioski:
- Transfer learning daje ogromną przewagę: przy ok. 120 zdjęciach treningowych na rasę
  sieć uczona od zera osiąga 72 %, a sieci wstępnie wytrenowane 97-99 %.
  Własna CNN wyraźnie się przeucza: pod koniec treningu 87 % na zbiorze treningowym
  i tylko 67 % na walidacji - za mało danych, by nauczyć cech ras od zera.
- Do aplikacji wybrano **ResNet-50** (najwyższa dokładność). Różnice między trzema
  sieciami z transfer learningiem to 3-6 zdjęć na 250, więc lżejsze modele
  (EfficientNet-B0, MobileNetV2) są dobrą alternatywą, gdy liczy się rozmiar lub szybkość.
- Najczęstsze pomyłki dotyczą podobnych ras: Labrador / Golden retriever,
  Rottweiler / Labrador (czarna sierść), Corgi / Husky.
- Rasy spoza listy (np. mops, szpic) oraz obrazy bez psa dostają niski wynik
  (19-37 %), więc aplikacja pokazuje ostrzeżenie. Zrzuty: `reports/gui_*.png`.
- Najwyższe wyniki dla poprawnych ras to ok. 90-95 %, a nie 100 %: to efekt label
  smoothing, który celowo zmniejsza nadmierną pewność modelu.

## Odtworzenie treningu

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
```
Pobrać zbiór Stanford Dogs (http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar,
ok. 750 MB) do `data/images.tar`, potem:
```
.venv\Scripts\python src\prepare_data.py
powershell -ExecutionPolicy Bypass -File run_all.ps1
```
Na RTX 3060 cała seria (4 modele) trwa ok. 10 minut.

Uwaga techniczna: w PyTorch 2.14 z cuDNN 9 format pamięci `channels_last` spowalniał
trening ok. 20 razy (`tools/profile_gpu.py`), dlatego go nie używamy.

## Ograniczenia

- Model zna tylko 10 ras i zawsze wybiera jedną z nich; dla innych psów pomaga
  ostrzeżenie przy niskim wyniku, ale nie jest to pełne wykrywanie „nieznanej rasy”.
- Zbiór testowy jest mały (250 zdjęć), więc różnice rzędu 1 % między modelami
  nie są rozstrzygające.
- Sieci wstępnie wytrenowane uczyły się na ImageNet, a zdjęcia Stanford Dogs pochodzą
  z ImageNet, więc wyniki mogą być nieco wyższe niż dla zupełnie nowych zdjęć.

## Literatura

[1] An Intelligent Dog Breed Recognition System Using Deep Learning (2022),
    https://ijdiic.com/research/article/view/12
[2] Dog Breed Classification and Identification Using Convolutional Neural Networks (2023),
    https://ph01.tci-thaijo.org/index.php/ecticit/article/view/253728
