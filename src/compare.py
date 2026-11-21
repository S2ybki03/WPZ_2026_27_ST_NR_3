"""Porównanie wszystkich eksperymentów i wybór modelu do aplikacji.

    python src/compare.py

Czyta runs/*/metrics.json, zapisuje reports/comparison.md i reports/comparison.png,
a najlepszy model (wg top-1 na teście) kopiuje do models/dog_breed.pt.
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import REPORTS_DIR, ROOT, RUNS_DIR


def main():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*/metrics.json"))]
    if not rows:
        raise SystemExit("Brak wyników - najpierw train.py i evaluate.py")
    rows.sort(key=lambda r: r["test_top1"], reverse=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    head = ("| Eksperyment | Parametry [mln] | Rozmiar [MB] | Val acc | Test top-1 | Test top-5 "
            "| F1 macro | Czas CPU [ms] | Czas GPU [ms] |")
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        gpu = r["latency_ms_gpu"] if r["latency_ms_gpu"] is not None else "-"
        lines.append(f"| {r['run']} | {r['params_mln']} | {r['model_size_mb']} | {r['val_acc']:.3f} "
                     f"| {r['test_top1']:.3f} | {r['test_top5']:.3f} | {r['test_f1_macro']:.3f} "
                     f"| {r['latency_ms_cpu']} | {gpu} |")
    best = rows[0]
    text = "\n".join(lines) + f"\n\nWybrany model: **{best['run']}** (najwyższa dokładność top-1 na zbiorze testowym).\n"
    (REPORTS_DIR / "comparison.md").write_text(text, encoding="utf-8")
    print(text)

    names = [r["run"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r["test_top1"] for r in rows], 0.4, label="top-1")
    ax.bar([i + 0.2 for i in x], [r["test_top5"] for r in rows], 0.4, label="top-5")
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["test_top1"] + 0.01, f"{r['test_top1']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r["test_top5"] + 0.01, f"{r['test_top5']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Dokładność na zbiorze testowym")
    ax.set_title("Porównanie modeli")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "comparison.png", dpi=130)
    plt.close(fig)

    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(RUNS_DIR / best["run"] / "best.pt", ROOT / "models" / "dog_breed.pt")
    print(f"Skopiowano runs/{best['run']}/best.pt -> models/dog_breed.pt")


if __name__ == "__main__":
    main()
