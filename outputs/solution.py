"""Train a small 0/180-degree text orientation model and make a submission."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 64
NORMALIZED_STD = 80.0
SEED = 42


def letterbox(image: Image.Image, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> np.ndarray:
    """Return a grayscale uint8 image, keeping the original aspect ratio."""
    image = image.convert("L")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), color=127)
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return np.asarray(canvas, dtype=np.uint8)


def standardize_crops(images: np.ndarray) -> np.ndarray:
    """Normalize each grayscale crop to mean 127 and std 80, rotation-invariantly."""
    values = np.asarray(images, dtype=np.float32)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("images must have shape (height, width) or (batch, height, width)")
    mean = values.mean(axis=(1, 2), keepdims=True)
    std = values.std(axis=(1, 2), keepdims=True)
    scale = NORMALIZED_STD / np.maximum(std, 8.0)
    normalized = np.clip((values - mean) * scale + 127.0, 0.0, 255.0).astype(np.float32)
    return normalized[0] if single else normalized


def p180_from_logits(logit_input: float, logit_rot180: float, temperature: float = 1.0) -> float:
    """Estimate whether the input is inverted, symmetrically scoring both rotations."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    value = (float(logit_input) - float(logit_rot180)) / (2.0 * temperature)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def validate_predictions(image_ids: Sequence[str], probabilities: Sequence[float]) -> tuple[list[str], list[float]]:
    """Validate prediction values and return ordinary lists in submission order."""
    ids = [str(value) for value in image_ids]
    values = [float(value) for value in probabilities]
    if len(ids) != len(values):
        raise ValueError("image_ids and probabilities must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("image_ids must be unique")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("every p_180 must be a finite number in [0, 1]")
    return ids, values


def _annotation_items(annotation_path: Path, images_root: Path) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Select nearly horizontal, legible TextOCR word boxes for orientation training."""
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data["imgs"]
    items: list[tuple[Path, tuple[float, float, float, float]]] = []
    resolved_paths: dict[str, Path] = {}
    for ann in data["anns"].values():
        text = ann.get("utf8_string", "")
        if not text or text == ".":
            continue
        bbox = ann.get("bbox", ())
        points = ann.get("points", ())
        if len(bbox) != 4 or len(points) < 8:
            continue
        x, y, w, h = map(float, bbox)
        if w < 12 or h < 7 or h > 180 or w / max(h, 1.0) < 1.2 or w / max(h, 1.0) > 12:
            continue
        # TextOCR polygon points start at the text's top-left and proceed clockwise.
        # Keep crops whose first edge is close to horizontal; the test boxes are horizontal.
        dx = float(points[2]) - float(points[0])
        dy = float(points[3]) - float(points[1])
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if dx <= 0 or angle > 12:
            continue
        image_id = ann["image_id"]
        if image_id not in resolved_paths:
            file_name = Path(images[image_id]["file_name"])
            resolved_paths[image_id] = resolve_textocr_image(images_root, file_name)
        image_path = resolved_paths[image_id]
        items.append((image_path, (x, y, w, h)))
    return items


def _crop_from_record(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox
    pad_x = max(1.0, w * 0.025)
    pad_y = max(1.0, h * 0.08)
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(image.width, int(math.ceil(x + w + pad_x)))
    bottom = min(image.height, int(math.ceil(y + h + pad_y)))
    return image.crop((left, top, right, bottom))


def _cache_name(split: str, limit: int, seed: int) -> str:
    return f"textocr_{split}_{limit}_{seed}.npy"


def resolve_textocr_image(images_root: Path, file_name: Path) -> Path:
    """Resolve either annotation paths or the official archive's *_images folders."""
    direct = images_root / file_name
    if direct.exists():
        return direct
    split = file_name.parts[0] if len(file_name.parts) > 1 else "train"
    for candidate in (images_root / f"{split}_images" / file_name.name, images_root / file_name.name):
        if candidate.exists():
            return candidate
    return direct


def group_annotations_by_image(items: Sequence[tuple[Path, tuple[float, float, float, float]]]):
    """Order chosen crops together so each source photo is decoded only once at a time."""
    return sorted(items, key=lambda item: str(item[0]).casefold())


def build_cache(annotation_path: Path, images_root: Path, cache_dir: Path, split: str,
                limit: int, seed: int, force: bool = False) -> Path:
    """Cache a deterministic sample of near-horizontal word crops as uint8 arrays."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / _cache_name(split, limit, seed)
    if destination.exists() and not force:
        return destination

    items = _annotation_items(annotation_path, images_root)
    random.Random(seed).shuffle(items)
    if limit > 0:
        items = items[:limit]
    items = group_annotations_by_image(items)
    if not items:
        raise ValueError(f"No usable annotations found in {annotation_path}")

    temporary = destination.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8,
        shape=(len(items), IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    valid = 0
    opened_path: Path | None = None
    source: Image.Image | None = None
    for index, (image_path, bbox) in enumerate(items):
        try:
            if opened_path != image_path:
                if source is not None:
                    source.close()
                source = Image.open(image_path).convert("RGB")
                opened_path = image_path
            crop = _crop_from_record(source, bbox)
            if crop.width < 4 or crop.height < 4:
                continue
            array[valid] = letterbox(crop)
            valid += 1
        except (OSError, ValueError, ZeroDivisionError):
            continue
        if (index + 1) % 25000 == 0:
            print(f"{split}: cached {index + 1:,}/{len(items):,} annotations", flush=True)
    if source is not None:
        source.close()
    array.flush()
    del array
    if valid == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Could not decode any crops from {annotation_path}")

    # Trim unused rows if some source images were missing/corrupt.
    if valid != len(items):
        complete = np.load(temporary, mmap_mode="r")[:valid].copy()
        np.save(temporary, complete, allow_pickle=False)
        del complete
    temporary.replace(destination)
    print(f"{split}: wrote {valid:,} crops to {destination}", flush=True)
    return destination


def _load_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise SystemExit("Install PyTorch first; see outputs/README.md") from error
    return torch, nn


def _make_model():
    torch, nn = _load_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class TinyOrientationNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(1, 16),
                ConvBlock(16, 24),
                ConvBlock(24, 40),
                ConvBlock(40, 64),
                ConvBlock(64, 80, stride=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.classifier = nn.Linear(80, 1)

        def forward(self, x):
            return self.classifier(self.features(x)).squeeze(1)

    return TinyOrientationNet()


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _load_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _calibrate(model, validation: np.ndarray, batch_size: int, device: str):
    torch, _ = _load_torch()
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            standardized = standardize_crops(validation[start:start + batch_size])
            original = torch.from_numpy(standardized)
            original = original.to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
            pair = torch.cat((original, original.flip((-2, -1))), dim=0)
            logits = model(pair).float().cpu().numpy()
            count = len(original)
            outputs.append((logits[:count] - logits[count:]) / 2.0)
    differences = np.concatenate(outputs)
    best_temperature, best_brier, best_accuracy = 1.0, float("inf"), 0.0
    for temperature in np.geomspace(0.35, 3.0, 60):
        clipped = np.clip(differences / temperature, -40, 40)
        p180_original = 1.0 / (1.0 + np.exp(-clipped))
        # Each original crop is the 0-degree member; its 180-degree partner
        # receives the complementary probability.
        brier = float(np.mean(np.square(p180_original)))
        accuracy = float(np.mean(p180_original < 0.5))
        if brier < best_brier:
            best_temperature, best_brier, best_accuracy = float(temperature), brier, accuracy
    return best_temperature, best_brier, best_accuracy


def train(args) -> None:
    torch, nn = _load_torch()
    _seed_torch(args.seed)
    data_dir = Path(args.textocr_dir)
    images_root = Path(args.images_dir) if args.images_dir else data_dir / "images"
    images_zip = Path(args.images_zip) if args.images_zip else data_dir / "train_val_images.zip"
    if not images_root.exists():
        if not images_zip.exists():
            raise SystemExit(f"Missing TextOCR images: {images_zip}")
        images_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(images_zip) as archive:
            root = images_root.resolve()
            for member in archive.infolist():
                target = (images_root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(images_root)

    train_json = Path(args.train_json) if args.train_json else data_dir / "TextOCR_0.1_train.json"
    val_json = Path(args.val_json) if args.val_json else data_dir / "TextOCR_0.1_val.json"
    if not train_json.exists() or not val_json.exists():
        raise SystemExit("TextOCR train and validation JSON files are required")
    cache_dir = Path(args.cache_dir)
    train_cache = build_cache(train_json, images_root, cache_dir, "train", args.train_limit, args.seed, args.rebuild_cache)
    val_cache = build_cache(val_json, images_root, cache_dir, "val", args.val_limit, args.seed, args.rebuild_cache)

    class RotatedPairDataset(torch.utils.data.Dataset):
        def __init__(self, path: Path):
            self.images = np.load(path, mmap_mode="r")

        def __len__(self):
            return len(self.images) * 2

        def __getitem__(self, index):
            crop_index, rotated = divmod(index, 2)
            image = np.asarray(self.images[crop_index])
            if rotated:
                image = image[::-1, ::-1]
            return torch.from_numpy(image.copy()).unsqueeze(0), torch.tensor(float(rotated))

    train_data = RotatedPairDataset(train_cache)
    val_images = np.load(val_cache, mmap_mode="r")
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
    )
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = _make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_brier = float("inf")
    best_temperature = 1.0
    best_epoch = 0
    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training on {device}; {parameter_count:,} parameters; {len(train_data):,} rotated examples", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
            if device == "cuda":
                contrast = 0.78 + 0.44 * torch.rand((len(images), 1, 1, 1), device=device)
                brightness = -0.10 + 0.20 * torch.rand((len(images), 1, 1, 1), device=device)
                noise = torch.randn_like(images) * 0.018
                images = (images * contrast + brightness + noise).clamp_(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)
            seen += len(labels)

        temperature, brier, accuracy = _calibrate(model, val_images, args.batch_size, device)
        mean_loss = loss_total / max(seen, 1)
        print(f"epoch {epoch}/{args.epochs} loss={mean_loss:.5f} val_pair_brier={brier:.5f} "
              f"val_pair_accuracy={accuracy:.4f} temperature={temperature:.3f}", flush=True)
        if brier < best_brier:
            best_brier = brier
            best_temperature = temperature
            best_epoch = epoch
            torch.save({
                "state_dict": model.cpu().state_dict(),
                "temperature": best_temperature,
                "validation_pair_brier": best_brier,
                "validation_pair_accuracy": accuracy,
                "epoch": best_epoch,
                "seed": args.seed,
                "image_width": IMAGE_WIDTH,
                "image_height": IMAGE_HEIGHT,
                "input_normalization_std": NORMALIZED_STD,
                "parameter_count": parameter_count,
            }, output_path)
            metrics_path = output_path.with_name("validation_metrics.json")
            metrics_path.write_text(json.dumps({
                "metric": "synthetic_pair_brier",
                "brier_score": best_brier,
                "accuracy_at_0.5": accuracy,
                "temperature": best_temperature,
                "best_epoch": best_epoch,
                "seed": args.seed,
                "train_word_crops": len(train_data) // 2,
                "validation_word_crops": len(val_images),
                "input_normalization_std": NORMALIZED_STD,
                "model_parameters": parameter_count,
            }, indent=2), encoding="utf-8")
            model.to(device)

    print(f"saved {output_path}; best synthetic-pair Brier={best_brier:.5f}; "
          f"temperature={best_temperature:.3f}", flush=True)


def predict(args) -> None:
    torch, _ = _load_torch()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = _make_model()
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device).eval()
    temperature = float(checkpoint.get("temperature", 1.0))
    test_zip = Path(args.test_zip)
    with zipfile.ZipFile(test_zip) as archive:
        sample_name = args.sample_csv or "sample_submission.csv"
        with archive.open(sample_name) as raw:
            rows = list(csv.DictReader(line.decode("utf-8-sig") for line in raw))
        image_ids = [row["image_id"] for row in rows]
        probabilities: list[float] = []
        for start in range(0, len(image_ids), args.batch_size):
            ids = image_ids[start:start + args.batch_size]
            originals: list[np.ndarray] = []
            for image_id in ids:
                image_path = f"test/images/{image_id}.png"
                with archive.open(image_path) as raw_image:
                    with Image.open(raw_image) as image:
                        originals.append(letterbox(image))
            standardized = standardize_crops(np.stack(originals))
            original = torch.from_numpy(standardized).to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
            pair = torch.cat((original, original.flip((-2, -1))), dim=0)
            with torch.no_grad():
                logits = model(pair).float().cpu().numpy()
            count = len(ids)
            probabilities.extend(
                p180_from_logits(float(logits[i]), float(logits[count + i]), temperature)
                for i in range(count)
            )
            if (start + len(ids)) % (args.batch_size * 10) == 0 or start + len(ids) == len(image_ids):
                print(f"predicted {start + len(ids):,}/{len(image_ids):,}", flush=True)

    ids, probabilities = validate_predictions(image_ids, probabilities)
    output = Path(args.submission_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "p_180"])
        writer.writerows(zip(ids, probabilities))
    print(f"wrote {len(ids):,} predictions to {output}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train from TextOCR and synthetic 180-degree pairs")
    train_parser.add_argument("--textocr-dir", default="work/textocr")
    train_parser.add_argument("--images-dir")
    train_parser.add_argument("--images-zip")
    train_parser.add_argument("--train-json")
    train_parser.add_argument("--val-json")
    train_parser.add_argument("--cache-dir", default="work/textocr/cache")
    train_parser.add_argument("--train-limit", type=int, default=150000)
    train_parser.add_argument("--val-limit", type=int, default=25000)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument("--model-out", default="outputs/orientation_model.pt")
    train_parser.add_argument("--rebuild-cache", action="store_true")
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict", help="write p_180 predictions for a test ZIP")
    predict_parser.add_argument("--test-zip", required=True)
    predict_parser.add_argument("--sample-csv")
    predict_parser.add_argument("--model", default="outputs/orientation_model.pt")
    predict_parser.add_argument("--submission-out", default="outputs/submission.csv")
    predict_parser.add_argument("--batch-size", type=int, default=256)
    predict_parser.add_argument("--cpu", action="store_true")
    predict_parser.set_defaults(run=predict)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
