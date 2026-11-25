"""Konwersja modelu ResNet-50 z PyTorch (.pt) do formatu numpy (.npz).
BatchNorm jest wtapiany w wagi konwolucji. Wymaga torch (tylko do konwersji).
    python tools/export_numpy.py models/dog_breed.pt models/dog_breed.npz
"""
import json
import sys

import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
if ck["model"] != "resnet50":
    raise SystemExit("Wersja numpy obsługuje tylko model resnet50")
sd = {k: v.float().numpy() for k, v in ck["state_dict"].items() if not k.endswith("num_batches_tracked")}
out = {}


def fold(conv, bn, name):
    w = sd[conv + ".weight"]
    s = sd[bn + ".weight"] / np.sqrt(sd[bn + ".running_var"] + 1e-5)
    b = sd[bn + ".bias"] - sd[bn + ".running_mean"] * s
    w = w * s[:, None, None, None]
    # (out, in, kh, kw) -> (kh*kw*in, out) dla układu NHWC
    o, i, kh, kw = w.shape
    out[name + ".w"] = np.ascontiguousarray(w.transpose(2, 3, 1, 0).reshape(kh * kw * i, o)).astype(np.float32)
    out[name + ".b"] = b.astype(np.float32)


fold("conv1", "bn1", "stem")
for li, n in zip(range(1, 5), (3, 4, 6, 3)):
    for bi in range(n):
        p = f"layer{li}.{bi}"
        for c in (1, 2, 3):
            fold(f"{p}.conv{c}", f"{p}.bn{c}", f"{p}.c{c}")
        if f"{p}.downsample.0.weight" in sd:
            fold(f"{p}.downsample.0", f"{p}.downsample.1", f"{p}.ds")
out["fc.w"] = sd["fc.weight"].T.astype(np.float32).copy()
out["fc.b"] = sd["fc.bias"].astype(np.float32)
meta = {"model": "resnet50", "classes": ck["classes"], "img_size": ck.get("img_size", 224)}
out["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
np.savez(dst, **out)
print("Zapisano", dst)
