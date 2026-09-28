"""Train a throwaway low-resolution/contrast-robust orientation candidate."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

OUTPUT_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = OUTPUT_DIR.parent
if (LOCAL_ROOT / "outputs" / "solution.py").exists():
    REPO = LOCAL_ROOT
    WORK_DIR = REPO / "work"
else:
    WORK_DIR = LOCAL_ROOT / "work"
    REPO = WORK_DIR / "Avito-cv-test"
ROOT = WORK_DIR
SPEC = importlib.util.spec_from_file_location("orientation_solution", REPO / "outputs" / "solution.py")
solution = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(solution)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)

train_path = ROOT / "textocr" / "cache" / "textocr_train_150000_42.npy"
val_path = ROOT / "textocr" / "cache" / "textocr_val_25000_42.npy"
train_images = np.load(train_path, mmap_mode="r")
val_images = np.load(val_path, mmap_mode="r")


class RotatedPairDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return 2 * len(self.images)

    def __getitem__(self, index):
        crop_index, rotated = divmod(index, 2)
        image = np.asarray(self.images[crop_index])
        if rotated:
            image = image[::-1, ::-1]
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.tensor(float(rotated))


def augment_training_batch(images: torch.Tensor) -> torch.Tensor:
    """Input [B,1,64,224] in [0,1]; label-preserving test-like degradation."""
    mean = images.mean(dim=(-2, -1), keepdim=True)
    contrast = 0.75 + 1.55 * torch.rand((len(images), 1, 1, 1), device=images.device)
    brightness = -0.10 + 0.20 * torch.rand((len(images), 1, 1, 1), device=images.device)
    images = ((images - mean) * contrast + mean + brightness).clamp_(0.0, 1.0)
    if torch.rand((), device=images.device).item() < 0.9:
        scale = 0.22 + 0.36 * torch.rand((), device=images.device).item()
        h, w = max(12, round(64 * scale)), max(40, round(224 * scale))
        images = F.interpolate(images, size=(h, w), mode="area")
        images = F.interpolate(images, size=(64, 224), mode="bilinear", align_corners=False)
    noise = torch.randn_like(images) * 0.012
    return (images + noise).clamp_(0.0, 1.0)


def validation_differences(model, raw_images: np.ndarray, scale: float | None,
                           contrast: float = 1.5) -> np.ndarray:
    diffs = []
    with torch.no_grad():
        for start in range(0, len(raw_images), 256):
            x = torch.from_numpy(np.asarray(raw_images[start:start + 256], dtype=np.float32))
            x = x.unsqueeze(1).to(device).div_(255.0)
            if contrast != 1.0:
                mean = x.mean(dim=(-2, -1), keepdim=True)
                x = ((x - mean) * contrast + mean).clamp_(0.0, 1.0)
            if scale is not None:
                h, w = max(12, round(64 * scale)), max(40, round(224 * scale))
                x = F.interpolate(x, size=(h, w), mode="area")
                x = F.interpolate(x, size=(64, 224), mode="bilinear", align_corners=False)
            raw = (x[:, 0].cpu().numpy() * 255.0).astype(np.float32)
            standardized = solution.standardize_crops(raw)
            x = torch.from_numpy(standardized).to(device).unsqueeze(1).float().div_(255.0)
            pair = torch.cat((x, x.flip((-2, -1))), dim=0)
            logits = model(pair).float().cpu().numpy()
            n = len(x)
            diffs.append((logits[:n] - logits[n:]) / 2.0)
    return np.concatenate(diffs)


def calibrate(differences: np.ndarray) -> tuple[float, float, float]:
    best = (float("inf"), 1.0, 0.0)
    for temperature in np.geomspace(0.35, 3.0, 60):
        p = 1.0 / (1.0 + np.exp(-np.clip(differences / temperature, -40, 40)))
        brier = float(np.mean(p * p))
        if brier < best[0]:
            best = (brier, float(temperature), float(np.mean(p < 0.5)))
    return best


def evaluate_suite(model) -> dict:
    suite = {"clean": (None, 1.0), "scale_0.25": (0.25, 1.5),
             "scale_0.35": (0.35, 1.5), "scale_0.50": (0.50, 1.5)}
    per_condition = {}
    pooled = []
    for name, (scale, contrast) in suite.items():
        d = validation_differences(model, val_images, scale, contrast)
        per_condition[name] = d
        pooled.append(d)
    pooled_d = np.concatenate(pooled)
    brier, temperature, accuracy = calibrate(pooled_d)
    by_condition = {}
    for name, d in per_condition.items():
        p = 1.0 / (1.0 + np.exp(-np.clip(d / temperature, -40, 40)))
        by_condition[name] = {
            "brier_at_shared_temperature": float(np.mean(p * p)),
            "accuracy_at_shared_temperature": float(np.mean(p < 0.5)),
            "brier_with_condition_temperature": calibrate(d)[0],
        }
    return {"suite_brier": brier, "shared_temperature": temperature,
            "suite_accuracy": accuracy, "conditions": by_condition}


model = solution._make_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
generator = torch.Generator().manual_seed(SEED)
loader = torch.utils.data.DataLoader(
    RotatedPairDataset(train_images), batch_size=256, shuffle=True,
    num_workers=0, pin_memory=device == "cuda", generator=generator,
)
baseline = solution._make_model().to(device).eval()
baseline_checkpoint = torch.load(REPO / "outputs" / "orientation_model.pt", map_location="cpu", weights_only=False)
baseline.load_state_dict(baseline_checkpoint["state_dict"])
baseline_metrics = evaluate_suite(baseline)
print("baseline", json.dumps(baseline_metrics), flush=True)

best_brier = float("inf")
output_path = OUTPUT_DIR / "orientation_model_testlike_candidate.pt"
metrics_path = OUTPUT_DIR / "testlike_candidate_metrics.json"
for epoch in range(1, 6):
    model.train()
    loss_sum = 0.0
    seen = 0
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
        images = augment_training_batch(images)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        seen += len(labels)
        if step % 300 == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={loss_sum / seen:.5f}", flush=True)
    model.eval()
    metrics = evaluate_suite(model)
    metrics["epoch"] = epoch
    metrics["train_loss"] = loss_sum / seen
    print("candidate", json.dumps(metrics), flush=True)
    if metrics["suite_brier"] < best_brier:
        best_brier = metrics["suite_brier"]
        torch.save({
            "state_dict": model.cpu().state_dict(),
            "temperature": metrics["shared_temperature"],
            "validation_pair_brier": best_brier,
            "validation_suite": "clean + scales 0.25/0.35/0.50 with contrast 1.5",
            "epoch": epoch,
            "seed": SEED,
            "image_width": 224,
            "image_height": 64,
            "input_normalization_std": 80.0,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }, output_path)
        model.to(device)
        metrics_path.write_text(json.dumps({"baseline": baseline_metrics, "best_candidate": metrics}, indent=2), encoding="utf-8")

print(f"saved {output_path}; best suite Brier={best_brier:.6f}")
