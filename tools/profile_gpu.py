"""Diagnostyka szybkości GPU przy różnych ustawieniach."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from models import build_model


def bench(name, amp, cl, bench_flag, n=15):
    torch.backends.cudnn.benchmark = bench_flag
    dev = torch.device("cuda")
    m, _ = build_model(name, 120)
    m = m.to(dev)
    x = torch.randn(64, 3, 224, 224, device=dev)
    if cl:
        m = m.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, 120, (64,), device=dev)
    opt = torch.optim.AdamW(m.parameters())
    for i in range(n + 5):
        if i == 5:
            torch.cuda.synchronize(); t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"{name:16s} amp={amp!s:5} channels_last={cl!s:5} cudnn.benchmark={bench_flag!s:5}: "
          f"{n * 64 / (time.perf_counter() - t):6.0f} zdjęć/s", flush=True)


if __name__ == "__main__":
    print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0))
    for amp in (False, True):
        for cl in (False, True):
            for b in (False, True):
                bench("mobilenet_v2", amp, cl, b)
    bench("resnet50", True, False, True)
