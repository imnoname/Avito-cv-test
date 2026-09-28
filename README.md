# Text orientation submission

The training source is the [TextOCR Kaggle dataset](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset), using the official TextOCR v0.1 images and annotations. TextOCR contains natural scene text with word-level polygons; its upstream data is listed under CC BY 4.0. The dataset has no 0°/180° labels, so the training script keeps near-horizontal word boxes and creates paired examples by rotating each crop 180°. The official train and validation image splits remain separate.

The classifier is a small depthwise-separable CNN trained from scratch. Each letterboxed grayscale crop is standardized to mean 127 and standard deviation 80 (with clipping) before inference. The model scores every box in both orientations and symmetrizes the two logits, so rotating an input swaps its probability with `1 - p_180`. The exported checkpoint contains 97,729 model parameters and a validation-fitted temperature.

The implementation uses the open-source libraries PyTorch, NumPy, and Pillow at the pinned versions in `requirements.txt`. No pretrained model is used. The model architecture and training/inference code are implemented in `solution.py`; the only generated labels are the 0°/180° training pairs described above.

## Prepare TextOCR

The dataset is included under `data/TextOCR/` using Git LFS. Install Git LFS and clone/pull the LFS files first. The image archive is split into parts; reassemble it and copy the annotations into `work/textocr/` with the PowerShell commands in [`data/TextOCR/README.md`](data/TextOCR/README.md). Alternatively, download the three original files directly from the official TextOCR distribution:

- `train_val_images.zip` — https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
- `TextOCR_0.1_train.json` — https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
- `TextOCR_0.1_val.json` — https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json

```powershell
New-Item -ItemType Directory -Force work\textocr | Out-Null
curl.exe -L --retry 10 --retry-all-errors --continue-at - -o work\textocr\train_val_images.zip https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
curl.exe -L --retry 10 --retry-all-errors --continue-at - -o work\textocr\TextOCR_0.1_train.json https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
curl.exe -L --retry 10 --retry-all-errors --continue-at - -o work\textocr\TextOCR_0.1_val.json https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json
```

The image archive is about 6.6 GB. The first training run extracts it and builds a deterministic cache of up to 150,000 training and 25,000 validation word crops. The cache stays under `work/` and can be reused on later runs.

## Install and run

Install PyTorch for the available hardware using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the pinned Python dependencies. The verified local setup used PyTorch 2.11.0 with CUDA 12.8 on an RTX 5070.

From the project root:

```powershell
python -m pip install -r outputs/requirements.txt
python outputs/solution.py train --textocr-dir work/textocr --seed 42
python outputs/solution.py predict `
  --test-zip "C:\Users\theju\Downloads\test.zip" `
  --model outputs/orientation_model.pt `
  --submission-out outputs/submission.csv
```

To use CPU inference, append `--cpu`. Training defaults to five epochs, a batch size of 256, and a 150k/25k crop limit. Change those options on machines with different resources. Every stochastic step uses the fixed seed unless `--seed` is changed.

## Validation and limits

TextOCR labels text content and polygons, not whether a crop is upside down. Validation therefore uses held-out source images from TextOCR's official validation split and synthetic 0°/180° pairs. With per-crop standardization, the validation Brier score is 0.06502 and accuracy is 0.89796. This measures the synthetic task; it is not an estimate of the challenge test Brier score. No test images are manually labeled, and the solution uses no external inference API or large language/vision model.

## Files

- `solution.py` — deterministic cache preparation, training, and test inference.
- `orientation_model.pt` — trained weights and calibration temperature.
- `validation_metrics.json` — best synthetic-pair Brier score, accuracy, and run settings.
- `submission.csv` — exactly `image_id,p_180` for the 20,000 test boxes.
