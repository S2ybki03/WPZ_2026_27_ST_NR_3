"""Architektury porównywane w projekcie.

simple_cnn      - własna sieć konwolucyjna uczona od zera (punkt odniesienia)
mobilenet_v2    - lekka sieć, jak w artykule [1] z przeglądu literatury
efficientnet_b0 - nowocześniejsza lekka sieć
resnet50        - większa, klasyczna sieć resztkowa

Trzy ostatnie startują z wag wytrenowanych na ImageNet (transfer learning):
podmieniamy tylko ostatnią warstwę na 120 wyjść (liczba ras) i douczamy.
"""
import torch.nn as nn
from torchvision import models as tvm

MODEL_NAMES = ["simple_cnn", "mobilenet_v2", "efficientnet_b0", "resnet50"]


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """5 bloków (2 x konwolucja 3x3 + BatchNorm + ReLU, potem MaxPool), 224 -> 7 px,
    na końcu uśrednianie globalne i warstwa w pełni połączona."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256), _block(256, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name, num_classes, pretrained=True):
    """Zwraca (model, nazwa_atrybutu_głowy). Głowa = warstwa klasyfikująca, uczona zawsze."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes), "head"
    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, "classifier"
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, "classifier"
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, "fc"
    raise ValueError(f"Nieznany model: {name}. Dostępne: {MODEL_NAMES}")


def set_backbone_trainable(model, head_attr, trainable):
    """Zamraża / odmraża wszystko poza głową. Faza 1: uczymy tylko głowę (szybko,
    nie psujemy wag z ImageNet), faza 2: douczamy całość z małym LR (fine-tuning)."""
    for name, p in model.named_parameters():
        p.requires_grad = trainable or name.startswith(head_attr + ".")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
