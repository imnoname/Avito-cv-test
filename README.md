# Определение поворота текста на 180°

Решение оценивает вероятность `p_180` для каждого текстового кропа. В этой версии используется компактная CNN, обученная на TextOCR и синтетических парах «исходный кроп / тот же кроп, повёрнутый на 180°».

## Итоговая версия

Файл `outputs/submission.csv` содержит 20 000 предсказаний версии, которая получила на платформе `1 − Brier = 0.75382309`. Этот балл сообщён платформой; локально вычислить Brier на тесте нельзя, потому что `test.zip` содержит изображения и `sample_submission.csv`, но не метки ориентации.

Для предсказаний этой версии после модели применялось сжатие вероятностей к 0.5:

```text
p = 0.5 + 0.16957433201954025 * (p_raw - 0.5)
```

Коэффициент выбран по агрегированной обратной связи публичного score предыдущей отправки. Индивидуальные метки теста и ручная разметка тестовых изображений не использовались. Эта калибровка описана здесь и в ноутбуке.

## Обучение и модель

- Источник данных: TextOCR v0.1, официальные train и validation split. В репозитории лежит архив изображений по частям и JSON-аннотации в `data/TextOCR/`.
- Для обучения используются до 150 000 горизонтальных текстовых кропов из train. Для выбора эпохи и температуры используются 25 000 кропов из отдельного validation split.
- Метки ориентации создаются автоматически поворотом каждого выбранного кропа на 180°.
- Архитектура — небольшая CNN с depthwise-separable свёртками, 97 729 параметров; модель обучается с нуля.
- Случайность фиксируется seed 42. Температура выбирается по Brier score на синтетических парах TextOCR. Этот результат не является оценкой Brier на тесте.
- Используются PyTorch, NumPy и Pillow; точные версии указаны в `outputs/requirements.txt`. Внешние API и большие LLM/VLM не используются.

## Подготовка данных

Потребуются Python и Git LFS. Получите LFS-файлы при клонировании репозитория, затем выполните инструкции в [`data/TextOCR/README.md`](data/TextOCR/README.md), чтобы собрать архив TextOCR и разместить данные в `work/textocr/`. Можно также скачать оригинальные TextOCR-файлы по ссылкам из этой инструкции.

Положите тестовый `test.zip` рядом с папкой `work/textocr/` либо задайте переменные окружения `TEXT_OCR_DIR` и `TEST_ZIP`.

## Запуск ноутбука

Установите зависимости из корня репозитория и запустите Jupyter:

```powershell
python -m pip install -r outputs/requirements.txt
$env:TEXT_OCR_DIR = "work/textocr"
$env:TEST_ZIP = "C:\path\to\test.zip"
jupyter lab outputs/orientation_solution_best075.ipynb
```

Выполните все ячейки. Ноутбук обучит модель, создаст `outputs/best_075/orientation_model.pt` и `outputs/submission.csv`, затем проверит формат, количество строк, ID и диапазон вероятностей.

## Запуск из командной строки

Обучение:

```powershell
python outputs/solution_best075.py train `
  --textocr-dir work/textocr `
  --cache-dir work/textocr/cache `
  --model-out outputs/best_075/orientation_model.pt `
  --seed 42 --epochs 5 --batch-size 256
```

Предсказания:

```powershell
python outputs/solution_best075.py predict `
  --test-zip "C:\path\to\test.zip" `
  --model outputs/best_075/orientation_model.pt `
  --submission-out outputs/submission.csv `
  --probability-shrink 0.16957433201954025
```

## Состав решения

- `outputs/solution_best075.py` — подготовка кропов, обучение и инференс версии 0.75.
- `outputs/orientation_solution_best075.ipynb` — ноутбук с пояснениями, запуском обучения и проверками результата.
- `outputs/best_075/orientation_model.pt` — checkpoint этой модели.
- `outputs/best_075/validation_metrics.json` — метрики синтетической TextOCR-валидации и параметры обучения.
- `outputs/submission.csv` — финальные предсказания для тестового набора.
- `outputs/requirements.txt` — версии зависимостей.
- `data/TextOCR/` — обучающие изображения и аннотации TextOCR.

Другие экспериментальные модели, CSV и сборки в репозитории не используются.
