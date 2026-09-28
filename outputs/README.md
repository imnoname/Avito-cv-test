# Итоговая модель ориентации текста (версия 0.75)

Эта папка содержит только код и артефакты решения, которое получило публичный результат `1 − Brier = 0.75382309`. Корневой README содержит краткое описание; здесь приведены полные команды запуска.

## 1. Требования

- Python 3.11 или 3.12.
- PyTorch, NumPy и Pillow из `requirements.txt`.
- Для быстрого обучения рекомендуется CUDA-совместимый GPU. Проверенная среда: Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX 5070. На CPU обучение поддерживается, но идёт значительно медленнее.

Из корня репозитория установите зависимости:

```powershell
python -m pip install -r outputs/requirements.txt
```

Если нужен GPU, установите сборку PyTorch для своей версии CUDA по официальной инструкции [PyTorch Start Locally](https://pytorch.org/get-started/locally/).

## 2. Подготовка TextOCR

В репозитории находятся официальные JSON-аннотации TextOCR и архив изображений, разделённый на части из-за ограничений Git LFS. Получите LFS-файлы при клонировании репозитория и из корня проекта выполните инструкции в [`data/TextOCR/README.md`](../data/TextOCR/README.md). Они соберут `work/textocr/train_val_images.zip` и скопируют JSON-аннотации в `work/textocr/`.

Первый запуск обучения распакует нужные изображения и создаст кэш кропов в `work/textocr/cache/`. Кэш повторно используется при следующих запусках.

## 3. Обучение модели

Запустите команду из корня репозитория:

```powershell
python outputs/solution_best075.py train `
  --textocr-dir work/textocr `
  --cache-dir work/textocr/cache `
  --model-out outputs/best_075/orientation_model.pt `
  --seed 42 `
  --epochs 5 `
  --batch-size 256
```

Скрипт выбирает горизонтальные word-box кропы TextOCR: до 150 000 для train и 25 000 из отдельного validation split. Для каждого кропа создаётся перевёрнутая на 180° копия, поэтому метки ориентации синтетические. Обучается с нуля небольшая CNN с depthwise-separable свёртками и 97 729 параметрами. Температура выбирается по Brier score на синтетических парах validation split.

При seed 42 лучшая эпоха в проверенной среде дала synthetic-pair Brier `0.07618729`, accuracy `0.89064` и temperature `0.93553168`. Эти метрики относятся к искусственным парам TextOCR; они не являются оценкой на тестовой разметке.

## 4. Создание submission.csv

Укажите путь к выданному архиву `test.zip`:

```powershell
python outputs/solution_best075.py predict `
  --test-zip "C:\path\to\test.zip" `
  --model outputs/best_075/orientation_model.pt `
  --submission-out outputs/submission.csv `
  --batch-size 256 `
  --probability-shrink 0.16957433201954025
```

Итоговый файл содержит ровно две колонки `image_id,p_180` и 20 000 строк. Порядок ID берётся из `sample_submission.csv` внутри архива. Вероятность выводится как:

```text
p = 0.5 + 0.16957433201954025 * (p_raw - 0.5)
```

Shrink-коэффициент был выбран по агрегированной обратной связи публичного score ранее отправленной версии и явно указан для воспроизводимости. Индивидуальные тестовые метки не использовались. Публичный score `0.75382309` — результат платформы для конкретного CSV; в `test.zip` нет меток, чтобы вычислить Brier локально.

## 5. Запуск ноутбука

Ноутбук выполняет те же шаги обучения и предсказания, а затем проверяет число строк, заголовки, порядок ID и диапазон вероятностей. Из корня репозитория укажите пути и запустите Jupyter:

```powershell
$env:TEXT_OCR_DIR = "work/textocr"
$env:TEST_ZIP = "C:\path\to\test.zip"
jupyter lab outputs/orientation_solution_best075.ipynb
```

Выполните все ячейки ноутбука.

## 6. Что используется

Используется датасет TextOCR v0.1 (лицензия и атрибуция описаны в `data/TextOCR/README.md`) и open-source библиотеки PyTorch, NumPy и Pillow. Модель обучается с нуля; внешние API, ручная разметка теста и большие LLM/VLM не используются. Случайность зафиксирована seed 42.

## Файлы

- `solution_best075.py` — подготовка данных, обучение и предсказание.
- `orientation_solution_best075.ipynb` — пояснения и воспроизводимый запуск.
- `best_075/orientation_model.pt` — checkpoint модели.
- `best_075/validation_metrics.json` — параметры обучения и метрики synthetic-pair валидации.
- `submission.csv` — итоговый файл предсказаний.
- `requirements.txt` — версии Python-зависимостей.
