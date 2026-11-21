"""Diagnostyka: gdzie jest wąskie gardło - wczytywanie danych czy GPU."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from dataset import DogsDataset, train_transform
from models import build_model


def main():
    ds = DogsDataset("train", train_transform())
    t = time.perf_counter()
    for i in range(200):
        ds[i]
    print(f"1 proces: {200 / (time.perf_counter() - t):.0f} zdjęć/s")

    for workers in (0, 4, 8):
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=workers)
        it = iter(dl)
        next(it)
        t = time.perf_counter()
        for _ in range(20):
            next(it)
        print(f"DataLoader workers={workers}: {20 * 64 / (time.perf_counter() - t):.0f} zdjęć/s")
        del it, dl

    dev = torch.device("cuda")
    m, _ = build_model("mobilenet_v2", 120)
    m = m.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters())
    x = torch.randn(64, 3, 224, 224, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 120, (64,), device=dev)
    scaler = torch.amp.GradScaler()
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize(); t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    print(f"GPU (sam model, mobilenet_v2): {20 * 64 / (time.perf_counter() - t):.0f} zdjęć/s")


if __name__ == "__main__":
    main()
