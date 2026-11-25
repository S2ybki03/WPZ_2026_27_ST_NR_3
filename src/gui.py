"""Graficzny interfejs aplikacji - rozpoznawanie ras psów.

    python src/gui.py

1. Użytkownik uruchamia aplikację (model wczytuje się w tle).
2. Wybiera zdjęcie psa z dysku (przycisk lub Ctrl+O).
3. Program przygotowuje zdjęcie do analizy (skalowanie, normalizacja - classifier.py).
4. Model SI określa najbardziej prawdopodobne rasy.
5. Aplikacja wyświetla 5 ras z wynikiem procentowym.
"""
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from classifier import DEFAULT_MODEL, DogBreedClassifier

TOP_K = 5
UNCERTAIN = 0.50          # poniżej tego progu ostrzegamy, że model nie jest pewny
PREVIEW = 440             # rozmiar podglądu zdjęcia [px]
IMG_TYPES = [("Zdjęcia", "*.jpg *.jpeg *.png *.bmp *.webp *.gif *.tif *.tiff"), ("Wszystkie pliki", "*.*")]

BG = "#f4f5f7"
CARD = "#ffffff"
ACCENT = "#2f6fdb"
BAR_BG = "#e3e7ee"
TEXT = "#1f2933"
MUTED = "#6b7785"
WARN = "#b35c00"


class ResultRow:
    """Jeden wiersz wyniku: miejsce, nazwa rasy, pasek i procent."""

    def __init__(self, parent, rank):
        self.frame = tk.Frame(parent, bg=CARD)
        self.frame.pack(fill="x", pady=6)
        top = tk.Frame(self.frame, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=f"{rank}.", width=3, anchor="w", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 11)).pack(side="left")
        self.name = tk.Label(top, text="—", anchor="w", bg=CARD, fg=TEXT,
                             font=("Segoe UI", 12, "bold" if rank == 1 else "normal"))
        self.name.pack(side="left", fill="x", expand=True)
        self.pct = tk.Label(top, text="", anchor="e", bg=CARD, fg=TEXT, font=("Segoe UI", 11))
        self.pct.pack(side="right")
        self.bar = tk.Canvas(self.frame, height=10, bg=CARD, highlightthickness=0)
        self.bar.pack(fill="x", padx=(28, 0), pady=(3, 0))
        self.bar.bind("<Configure>", lambda e: self._draw())
        self.value = 0.0
        self.color = ACCENT

    def set(self, name, value, color=ACCENT):
        self.name.config(text=name)
        self.pct.config(text=f"{value * 100:.1f} %" if name != "—" else "")
        self.value, self.color = value, color
        self._draw()

    def clear(self):
        self.set("—", 0.0)

    def _draw(self):
        w = self.bar.winfo_width()
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, w, 10, fill=BAR_BG, width=0)
        if self.value > 0:
            self.bar.create_rectangle(0, 0, max(2, int(w * self.value)), 10, fill=self.color, width=0)


class App(tk.Tk):
    def __init__(self, model_path=DEFAULT_MODEL):
        super().__init__()
        self.title("Rozpoznawanie ras psów")
        self.geometry("980x600")
        self.minsize(860, 540)
        self.configure(bg=BG)
        self.clf = None
        self.photo = None          # referencja, inaczej Tk usunie obraz z pamięci
        self.busy = False
        self._build()
        self.bind("<Control-o>", lambda e: self.open_image())
        self.load_model(Path(model_path))

    # ---------- budowa okna ----------
    def _build(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(bar, text="Rozpoznawanie ras psów", bg=BG, fg=TEXT,
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        style = ttk.Style(self)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        self.btn_open = ttk.Button(bar, text="Wczytaj zdjęcie…  (Ctrl+O)", style="Accent.TButton",
                                   command=self.open_image, state="disabled")
        self.btn_open.pack(side="right")
        ttk.Button(bar, text="Zmień model…", command=self.choose_model).pack(side="right", padx=8)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=6)

        left = tk.Frame(body, bg=CARD, highlightbackground="#d8dde5", highlightthickness=1)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg=CARD, highlightthickness=0, width=PREVIEW, height=PREVIEW)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda e: self._show_preview())
        self.canvas.bind("<Button-1>", lambda e: self.open_image())
        self.file_label = tk.Label(left, text="", bg=CARD, fg=MUTED, font=("Segoe UI", 9))
        self.file_label.pack(pady=(0, 8))
        self.image = None

        right = tk.Frame(body, bg=CARD, width=380, highlightbackground="#d8dde5", highlightthickness=1)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        inner = tk.Frame(right, bg=CARD)
        inner.pack(fill="both", expand=True, padx=18, pady=14)
        tk.Label(inner, text="Najbardziej prawdopodobne rasy", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.rows = [ResultRow(inner, i + 1) for i in range(TOP_K)]
        self.verdict = tk.Label(inner, text="", bg=CARD, fg=TEXT, wraplength=330, justify="left",
                                font=("Segoe UI", 10))
        self.verdict.pack(anchor="w", pady=(14, 0))
        self.known = tk.Label(inner, text="", bg=CARD, fg=MUTED, wraplength=330, justify="left",
                              font=("Segoe UI", 9))
        self.known.pack(side="bottom", anchor="w")

        self.status = tk.Label(self, text="", bg="#e8ebf0", fg=MUTED, anchor="w",
                               font=("Segoe UI", 9), padx=10, pady=4)
        self.status.pack(fill="x", side="bottom")
        self._show_preview()

    # ---------- model ----------
    def load_model(self, path):
        if not path.exists():
            self.set_status(f"Nie znaleziono modelu: {path}. Wybierz plik .npz przyciskiem „Zmień model…”.")
            return
        self.btn_open.config(state="disabled")
        self.set_status(f"Wczytywanie modelu {path.name} …")

        def work():
            try:
                clf = DogBreedClassifier(path)
                self.after(0, self._model_ready, clf, path)
            except Exception as e:
                self.after(0, self._error, f"Nie udało się wczytać modelu:\n{e}")

        threading.Thread(target=work, daemon=True).start()

    def _model_ready(self, clf, path):
        self.clf = clf
        self.btn_open.config(state="normal")
        self.known.config(text=f"Rasy rozpoznawane przez model ({len(clf.classes)}):\n"
                               + ", ".join(sorted(clf.classes)))
        self.set_status(f"Model: {clf.model_name} ({len(clf.classes)} ras) · {path.name} · "
                        f"urządzenie: {clf.device.type.upper()} · wczytaj zdjęcie psa")
        if self.image is not None:
            self.classify()

    def choose_model(self):
        path = filedialog.askopenfilename(title="Wybierz plik modelu", initialdir=DEFAULT_MODEL.parent,
                                          filetypes=[("Model (numpy)", "*.npz"), ("Wszystkie pliki", "*.*")])
        if path:
            self.load_model(Path(path))

    # ---------- zdjęcie ----------
    def open_image(self):
        if self.clf is None or self.busy:
            return
        path = filedialog.askopenfilename(title="Wybierz zdjęcie psa", filetypes=IMG_TYPES)
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.image = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            self._error(f"Nie można otworzyć pliku jako obrazu:\n{path}\n\n{e}")
            return
        self.file_label.config(text=f"{Path(path).name}  ·  {self.image.width}×{self.image.height} px")
        self._show_preview()
        self.classify()

    def _show_preview(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        if self.image is None:
            c.create_rectangle(20, 20, w - 20, h - 20, outline="#c5ccd6", dash=(6, 4), width=2)
            c.create_text(w / 2, h / 2 - 12, text="🐕", font=("Segoe UI Emoji", 40), fill=MUTED)
            c.create_text(w / 2, h / 2 + 40, text="Kliknij lub naciśnij Ctrl+O,\naby wczytać zdjęcie psa",
                          font=("Segoe UI", 11), fill=MUTED, justify="center")
            return
        img = self.image.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        c.create_image(w / 2, h / 2, image=self.photo)

    # ---------- klasyfikacja ----------
    def classify(self):
        self.busy = True
        self.btn_open.config(state="disabled")
        for r in self.rows:
            r.clear()
        self.verdict.config(text="Analiza zdjęcia…", fg=MUTED)
        image = self.image

        def work():
            try:
                t = time.perf_counter()
                res = self.clf.predict(image, top_k=TOP_K)
                self.after(0, self._show_results, res, (time.perf_counter() - t) * 1000)
            except Exception as e:
                self.after(0, self._error, f"Błąd podczas analizy zdjęcia:\n{e}")

        threading.Thread(target=work, daemon=True).start()

    def _show_results(self, res, ms):
        self.busy = False
        self.btn_open.config(state="normal")
        best = res[0]
        sure = best.probability >= UNCERTAIN
        for i, (row, r) in enumerate(zip(self.rows, res)):
            row.set(r.breed, r.probability, ACCENT if i == 0 and sure else "#9aa7b8")
        if sure:
            self.verdict.config(fg=TEXT, text=f"Najprawdopodobniej: {best.breed} ({best.percent:.1f} %).")
        else:
            self.verdict.config(fg=WARN, text=(
                f"Model nie jest pewny (najwyższy wynik {best.percent:.1f} %). Możliwe, że na zdjęciu "
                "jest mieszaniec, rasa spoza listy poniżej, kilka psów albo w ogóle nie ma psa."))
        self.set_status(f"Model: {self.clf.model_name} · analiza trwała {ms:.0f} ms")

    # ---------- pomocnicze ----------
    def set_status(self, text):
        self.status.config(text=text)

    def _error(self, text):
        self.busy = False
        if self.clf is not None:
            self.btn_open.config(state="normal")
        self.verdict.config(text="")
        self.set_status("Błąd - szczegóły w oknie komunikatu")
        messagebox.showerror("Błąd", text, parent=self)


if __name__ == "__main__":
    import sys
    App(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL).mainloop()
