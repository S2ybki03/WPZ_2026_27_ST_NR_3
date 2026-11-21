"""Automatyczny test GUI: wczytuje model, podaje zdjęcia testowe, robi zrzuty ekranu okna.

    python tools/gui_smoke.py runs/mobilenet_v2/best.pt zdjecie1.jpg [zdjecie2.jpg ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageGrab

import gui

out_dir = Path(__file__).resolve().parent.parent / "reports"
model, images = sys.argv[1], sys.argv[2:]
app = gui.App(model)
app.geometry("+40+40")
app.attributes("-topmost", True)   # żeby inne okna nie zasłaniały zrzutu
queue = list(images)
shots = []


def grab(name):
    app.update()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    ImageGrab.grab((x, y, x + app.winfo_width(), y + app.winfo_height())).save(out_dir / name)
    shots.append(name)


def step():
    if app.clf is None or app.busy:
        app.after(200, step)
        return
    if not shots:
        grab("gui_empty.png") if app.image is None else None
    if not queue:
        print("zrzuty:", shots)
        app.destroy()
        return
    path = queue.pop(0)
    with Image.open(path) as im:
        app.image = im.convert("RGB")
    app.file_label.config(text=Path(path).name)
    app._show_preview()
    app.classify()
    app.after(300, wait_result, Path(path).stem)


def wait_result(stem):
    if app.busy:
        app.after(200, wait_result, stem)
        return
    print(stem, "->", [(r.name.cget("text"), r.pct.cget("text")) for r in app.rows], "|", app.verdict.cget("text"))
    grab(f"gui_{stem}.png")
    app.after(100, step)


app.after(500, step)
app.mainloop()
