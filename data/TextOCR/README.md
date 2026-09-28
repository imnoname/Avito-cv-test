# TextOCR training data

This directory contains the official TextOCR v0.1 training and validation annotations and image archive, stored as Git LFS files. The image ZIP is split into ordered 1.7 GB parts because Git LFS file-size limits are below the archive's 6.6 GiB size. Rejoin the parts in numeric order before training:

```powershell
New-Item -ItemType Directory -Force work\textocr | Out-Null
$parts = Get-ChildItem data\TextOCR\train_val_images.zip.part* | Sort-Object Name
$output = [System.IO.File]::Create('work\textocr\train_val_images.zip')
try { foreach ($part in $parts) { $input = [System.IO.File]::OpenRead($part.FullName); try { $input.CopyTo($output) } finally { $input.Dispose() } } } finally { $output.Dispose() }
Copy-Item data\TextOCR\TextOCR_0.1_*.json work\textocr\
```

The TextOCR dataset is distributed under CC BY 4.0. Source: [official TextOCR dataset page](https://textvqa.org/textocr/) and [Kaggle mirror](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset). Please retain attribution to the TextOCR authors and consult the upstream page for citation and license details. The downloaded data is included for reproducibility; the solution itself is in `outputs/`.
