"""Trening jednego modelu.

PrzykĹ‚ady:
    python src/train.py --model simple_cnn
    python src/train.py --model mobilenet_v2
    python src/train.py --model efficientnet_b0
    python src/train.py --model resnet50

Wynik w runs/<model>/: best.pt (najlepszy model wg dokĹ‚adnoĹ›ci na walidacji),
history.json, curves.png, config.json.
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import IMG_SIZE, RUNS_DIR, SEED
from dataset import DogsDataset, eval_transform, load_classes, train_transform
from models import MODEL_NAMES, build_model, count_params, set_backbone_trainable

# Ustawienia domyĹ›lne dobrane osobno dla sieci od zera i dla transfer learningu
DEFAULTS = {
    "simple_cnn":      dict(epochs=80, freeze_epochs=0, lr=1e-3, ft_lr=None),
    "mobilenet_v2":    dict(epochs=15, freeze_epochs=3, lr=1e-3, ft_lr=1e-4),
    "efficientnet_b0": dict(epochs=15, freeze_epochs=3, lr=1e-3, ft_lr=1e-4),
    "resnet50":        dict(epochs=15, freeze_epochs=3, lr=1e-3, ft_lr=1e-4),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--name", help="nazwa eksperymentu (domyĹ›lnie = nazwa modelu)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--freeze-epochs", type=int, help="ile epok uczyÄ‡ tylko gĹ‚owÄ™")
    p.add_argument("--lr", type=float, help="LR dla gĹ‚owy / caĹ‚ej sieci od zera")
    p.add_argument("--ft-lr", type=float, help="LR dla szkieletu przy fine-tuningu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-pretrained", action="store_true", help="architektura bez wag ImageNet")
    a = p.parse_args()
    for k, v in DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.name = a.name or a.model
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, head_attr, lr, ft_lr, epochs, finetune):
    head = [p for n, p in model.named_parameters() if n.startswith(head_attr + ".")]
    body = [p for n, p in model.named_parameters() if not n.startswith(head_attr + ".")]
    if finetune:   # szkielet z mniejszym LR niĹĽ gĹ‚owa
        groups = [{"params": body, "lr": ft_lr}, {"params": head, "lr": ft_lr * 10}]
    elif all(p.requires_grad for p in body):   # sieÄ‡ od zera - wszystko jednakowo
        groups = [{"params": body + head, "lr": lr}]
    else:          # zamroĹĽony szkielet - uczymy tylko gĹ‚owÄ™
        groups = [{"params": head, "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    return opt, sched


def run_epoch(model, loader, criterion, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            # bez channels_last: w PyTorch 2.14 + cuDNN 9.24 spowalniaĹ‚o trening ~20x (tools/profile_gpu.py)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def plot_curves(hist, path, title):
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in hist], label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], label="val")
    a1.set_title("Funkcja straty"); a1.set_xlabel("Epoka"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in hist], label="train")
    a2.plot(ep, [h["val_acc"] for h in hist], label="val")
    a2.set_title("DokĹ‚adnoĹ›Ä‡"); a2.set_xlabel("Epoka"); a2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    a = parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"UrzÄ…dzenie: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    classes = load_classes()
    train_ds = DogsDataset("train", train_transform())
    val_ds = DogsDataset("val", eval_transform())
    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=True,
              persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model, head_attr = build_model(a.model, len(classes), pretrained=not a.no_pretrained)
    model = model.to(device)
    print(f"Model {a.model}: {count_params(model) / 1e6:.2f} mln parametrĂłw")

    out_dir = RUNS_DIR / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(a) | {"num_classes": len(classes), "img_size": IMG_SIZE,
                     "train_size": len(train_ds), "val_size": len(val_ds),
                     "params": count_params(model), "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    frozen = a.freeze_epochs > 0
    set_backbone_trainable(model, head_attr, trainable=not frozen)
    phase_epochs = a.freeze_epochs if frozen else a.epochs
    opt, sched = make_optimizer(model, head_attr, a.lr, a.ft_lr, phase_epochs, finetune=False)

    hist, best_acc, t_start = [], 0.0, time.time()
    for epoch in range(1, a.epochs + 1):
        if frozen and epoch == a.freeze_epochs + 1:   # przejĹ›cie do fine-tuningu
            print("--- odmraĹĽanie szkieletu, fine-tuning ---")
            set_backbone_trainable(model, head_attr, trainable=True)
            opt, sched = make_optimizer(model, head_attr, a.lr, a.ft_lr,
                                        a.epochs - a.freeze_epochs, finetune=True)
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, opt, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        sched.step()
        h = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                 val_acc=va_acc, seconds=time.time() - t0)
        hist.append(h)
        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": a.model, "state_dict": model.state_dict(), "classes": classes,
                        "img_size": IMG_SIZE, "epoch": epoch, "val_acc": va_acc},
                       out_dir / "best.pt")
            mark = "  * zapisano"
        print(f"[{epoch:2d}/{a.epochs}] train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {h['seconds']:.0f}s{mark}")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        plot_curves(hist, out_dir / "curves.png", a.name)

    print(f"Koniec. Najlepsza dokĹ‚adnoĹ›Ä‡ walidacyjna {best_acc:.4f}, "
          f"czas {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
