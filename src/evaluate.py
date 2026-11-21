"""Ocena wytrenowanego modelu na zbiorze testowym (zdjÄ™cia niewidziane przy treningu).

    python src/evaluate.py --run resnet50

Zapisuje w runs/<run>/: metrics.json, classification_report.txt, confusion_matrix.png,
f1_per_class.png, confused_pairs.txt.
"""
import argparse
import json
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

from config import RUNS_DIR
from dataset import DogsDataset, eval_transform
from models import build_model, count_params


def measure_latency(model, device, img_size, n=50):
    """Ĺšredni czas klasyfikacji jednego zdjÄ™cia (tak jak w aplikacji: batch = 1)."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    run_dir = RUNS_DIR / a.run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model, _ = build_model(ckpt["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    ds = DogsDataset("test", eval_transform(ckpt["img_size"]))
    dl = DataLoader(ds, batch_size=128, num_workers=a.workers, pin_memory=True)

    all_probs, all_y = [], []
    with torch.inference_mode():
        for x, y in dl:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x.to(device))
            all_probs.append(torch.softmax(out.float(), 1).cpu())
            all_y.append(y)
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_y).numpy()
    y_pred = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :5]

    metrics = {
        "run": a.run,
        "model": ckpt["model"],
        "params_mln": round(count_params(model) / 1e6, 2),
        "best_epoch": ckpt["epoch"],
        "val_acc": round(ckpt["val_acc"], 4),
        "test_size": int(len(y_true)),
        "test_top1": round(accuracy_score(y_true, y_pred), 4),
        "test_top5": round(float((top5 == y_true[:, None]).any(1).mean()), 4),
        "test_precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "test_recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "test_f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "latency_ms_gpu": round(measure_latency(model, device, ckpt["img_size"]), 2) if device.type == "cuda" else None,
        "latency_ms_cpu": round(measure_latency(model.float().cpu(), torch.device("cpu"), ckpt["img_size"], n=20), 2),
        "model_size_mb": round((run_dir / "best.pt").stat().st_size / 2**20, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    (run_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0),
        encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Macierz pomyĹ‚ek - {a.run}\n(liczba zdjÄ™Ä‡; kolor = odsetek w wierszu)")
    ax.set_xlabel("Rasa przewidziana")
    ax.set_ylabel("Rasa rzeczywista")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # F1 dla kaĹĽdej rasy (od najtrudniejszej)
    f1s = f1_score(y_true, y_pred, average=None, labels=range(len(classes)))
    worst = np.argsort(f1s)[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([classes[i] for i in worst][::-1], f1s[worst][::-1], color="#C44E52")
    for k, v in enumerate(f1s[worst][::-1]):
        ax.text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.08)
    ax.set_title(f"F1 dla poszczegĂłlnych ras - {a.run}")
    fig.tight_layout()
    fig.savefig(run_dir / "f1_per_class.png", dpi=130)
    plt.close(fig)

    # najczÄ™Ĺ›ciej mylone pary ras
    pairs = Counter((classes[t], classes[p_]) for t, p_ in zip(y_true, y_pred) if t != p_)
    lines = [f"{n:3d}  {t}  ->  {p_}" for (t, p_), n in pairs.most_common(20)]
    (run_dir / "confused_pairs.txt").write_text(
        "liczba  rasa rzeczywista  ->  rasa przewidziana\n" + "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
