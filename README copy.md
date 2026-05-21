![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Assessment | Cat Detection — YOLO26 Training

## Overview

This is the Week 1 assessment for Unit 6. Over the past three days you've learned how a neural network is built, how it's trained, and how convolutional architectures specialise that toolkit for images. Today you'll apply that foundation to a real **object detection** problem on a dataset *you* helped collect earlier in the course using a YOLO detection framework — meaning each image comes with bounding-box annotations, not just a class label.

You will **train YOLO26** (Ultralytics, January 2026), the latest YOLO release, on the cat-detection dataset starting from its COCO-pretrained weights. YOLO26 is a fully convolutional detector engineered for edge deployment with **end-to-end, NMS-free inference** — its ONNX export produces final detections directly, which will make next Friday's containerised inference assessment dramatically simpler. Today's job is to train it and evaluate it as a real detector that outputs **bounding boxes**, not just class labels.

The dataset:

- [Cat Detection Dataset on Google Drive](https://drive.google.com/drive/folders/1qeGvkaK7UkNMYoESQHxGbV4DRH8EgEb0?usp=drive_link)

The folder contains the cleaned data under `DATA_CLEAN/`, with images plus YOLO-format annotations (one `.txt` per image with lines `class_id cx cy w h`, all values normalised to [0, 1]).

## Learning Goals

By the end of this assessment you should have demonstrated that you can:

- Load and inspect a YOLO-format detection dataset and visualise bounding boxes.
- Configure a YOLO data YAML file and produce reproducible train / val / test splits.
- Train a YOLO26 detector on a custom dataset (starting from COCO-pretrained weights) and read the training logs.
- Evaluate a detector with the right metrics: **mAP@0.5**, **mAP@0.5:0.95**, **precision**, **recall**.
- Visualise predictions on test images and qualitatively analyse failure cases.

## Prerequisites

- Python 3.10+
- The lessons and labs from days 1–3 of Unit 6 (CNN fundamentals especially)

```bash
pip install ultralytics torch torchvision matplotlib pandas pillow opencv-python
```

The `ultralytics` package (version 8.4.38 or later) gives you YOLO26 training, validation, prediction, and export in a single API.

## Why YOLO26 (and Why This Counts as "CNN Training")

YOLO26 is a fully convolutional detector. Its backbone is the same kind of **conv → BatchNorm → ReLU** block you met on Day 03, just stacked into a deeper, more sophisticated architecture optimised for detection. When you call `model.train(...)`, you are training a CNN — exactly the kind of model the lessons prepared you for. The Ultralytics API takes care of the detection-specific machinery (anchor-free heads, loss functions, NMS-free decoding) so you can focus on the data, training dynamics, and evaluation.

Two things that make YOLO26 a great fit for this course:

1. **End-to-end NMS-free inference** — the model directly outputs final detections, so the ONNX export is clean and the container next week will be simple.
2. **Fast on CPU** — YOLO26 is up to 43% faster on CPUs than previous YOLO versions, which means the leaderboard run on May 22 actually fits in the class period.

## Dataset

Download the dataset from the Google Drive link above. Inspect the layout — most likely you'll see something like:

```
DATA_CLEAN/
  images/
    img001.jpg
    img002.jpg
    ...
  labels/
    img001.txt
    img002.txt
    ...
```

Or images and labels may live side-by-side in a single folder. Document what you find in your notebook. The label file format (YOLO format) is one line per object:

```
<class_id> <cx> <cy> <w> <h>
```

All four coordinates are normalised to `[0, 1]` relative to the image dimensions.

> Place the dataset locally at `data/DATA_CLEAN/`. **Do not commit it** — add `data/` to `.gitignore`.

## Requirements

Work in a single Jupyter Notebook called **`m6-04-assessment.ipynb`** at the root of your repository. Push the notebook (and a small `data.yaml` config file you'll create) to your fork.

## Tasks

### Task 1 — Inspect the Dataset

1. Walk the dataset folder and report:
   - Number of images and number of label files
   - Whether every image has a matching label file (and vice versa)
   - The set of class IDs used and the per-class object count (a small table or bar chart is fine)
   - Image size statistics — pick 100 random images and report min/max/mean width and height
2. Pick **6 random images** and visualise them in a 2×3 grid with their bounding boxes drawn on top. Label each box with its class name.

> Implementation hint: you can read the YOLO label file, denormalise the coordinates, and use `matplotlib.patches.Rectangle` to draw boxes.

### Task 2 — Build Train / Val / Test Splits

YOLO uses simple text-file splits.

1. Shuffle the image list with a fixed seed (e.g. `random.seed(42)`) and split it into roughly **70 / 15 / 15** train / val / test.
2. Write three text files: `train.txt`, `val.txt`, `test.txt`, each containing one absolute or relative image path per line.
3. Create a `data.yaml` config file Ultralytics will read:

```yaml
path: ./data
train: DATA_CLEAN/train.txt
val: DATA_CLEAN/val.txt
test: DATA_CLEAN/test.txt
names:
  0: cat
  # add other classes if your label files use them
```

Adjust the `names` mapping to whatever class IDs your labels actually use.

### Task 3 — Pick a YOLO26 Variant and Train It

YOLO26 ships in five sizes — `n`, `s`, `m`, `l`, `x`. Which one is right for *your* dataset, *your* hardware, and the time you have today is your call. Here are the trade-offs:

| Variant | Params | COCO mAP@0.5:0.95 | CPU ONNX (ms) | When to pick it |
|---|---|---|---|---|
| `yolo26n` | 2.4 M | 40.9 | 38.9 | Quickest to train and easiest to fit on a laptop CPU; smallest checkpoint to ship next week. Good if your dataset is small (≤ 1–2k images) or your hardware is modest. |
| `yolo26s` | 9.5 M | 48.6 | 87.2 | Sweet spot for many real datasets — clearly better accuracy than `n` at modest extra cost. Default pick if you have a GPU and a few thousand images. |
| `yolo26m` | 20.4 M | 53.1 | 220.0 | Higher accuracy ceiling, but training is noticeably slower and the ONNX you ship next Friday will be ~80 MB. Worth it only if `s` is clearly bottlenecked by capacity. |
| `yolo26l` / `yolo26x` | 24.8 / 55.7 M | 55.0 / 57.5 | 286 / 526 | Overkill for a small custom dataset and **awkward to ship** in next week's container. Don't pick these unless you have a strong reason. |

(All numbers are from the YOLO26 model card on COCO at 640×640.)

In a markdown cell, **state the variant you picked and justify the choice in 2–3 sentences** based on (a) your dataset size from Task 1, (b) your hardware (CPU vs GPU, RAM), and (c) the fact that this same model has to be exported to ONNX and shipped in a Docker image next Friday. Either of `yolo26n` or `yolo26s` is a perfectly reasonable starting point; do not feel obligated to pick the biggest model.

1. Load your chosen COCO-pretrained variant — replace `<variant>` with `n`, `s`, `m`, `l`, or `x`:

```python
from ultralytics import YOLO
model = YOLO("yolo26<variant>.pt")  # e.g. "yolo26s.pt"
```

2. Train it on your `data.yaml` for **at least 30 epochs**:

```python
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,            # adjust to your hardware (smaller for bigger variants / less VRAM)
    project="runs",
    name="cats_v1",
    seed=42,
)
```

3. Inspect the training logs / `results.png`. Report:
   - Best mAP@0.5 and mAP@0.5:0.95 achieved on the validation set
   - The epoch at which the best validation mAP was reached
   - Visible signs of overfitting (training loss decreasing while validation mAP plateaus or drops)
   - One sentence reflecting on whether your variant choice felt right in hindsight (under-trained? over-fitting? plenty of headroom?). This will inform your Week-2 improvement runs.

### Task 4 — Evaluate on the Test Set

1. Run validation explicitly on the **test** split using the best checkpoint:

```python
metrics = model.val(data="data.yaml", split="test")
print(metrics.box.map, metrics.box.map50, metrics.box.mp, metrics.box.mr)
```

2. Report a small table with:

| Metric | Value |
|---|---|
| mAP@0.5 | … |
| mAP@0.5:0.95 | … |
| Mean precision | … |
| Mean recall | … |

3. In a markdown cell, briefly explain in your own words what each metric measures.

### Task 5 — Visualise Predictions

1. Run predictions on **6 test images** and overlay the predicted boxes (with class + confidence text) on the original images. Use a 2×3 grid.
2. For 3 of those images, also draw the **ground-truth boxes** so you can see where the model agrees and disagrees.
3. Show **at least 2 images where the model fails** — either missing a real cat (false negative) or boxing something that is not a cat (false positive). Provide a short markdown comment on each: what likely caused the mistake?

### Task 6 — Reflection (Required, Brief)

In a final markdown cell, write 5–7 sentences answering:

- What did you learn about training a real object detector versus a classifier?
- What part of the dataset (or labels) caused the most trouble?
- Looking at the failure cases, what single change would you try next? (Bigger backbone? More augmentation? Cleaning a few mislabelled images?)
- One thing you'll do differently in next Friday's improvement assessment.

This reflection is part of the grade and primes you for the Friday 22 May assessment, where you'll **improve this exact model**, convert it to ONNX, and ship it as a containerised inference service that competes on a class leaderboard.

## Submission

### What to submit

A pull request on your fork that contains:

- `m6-04-assessment.ipynb` — your completed notebook with all tasks, runs without errors top-to-bottom.
- `data.yaml` — your dataset config.
- `runs/cats_v1/weights/best.pt` — your best YOLO26 checkpoint (commit only if <50MB; otherwise upload to Drive and link).
- `README.md` at the repo root — minimal description of how to reproduce.
- `.gitignore` excluding the `data/` folder and large run artefacts.

### Definition of done (checklist)

- [ ] Dataset inspection: image and label counts, class distribution, 6-image visualisation with boxes.
- [ ] `data.yaml` and split text files created and validated.
- [ ] YOLO26 variant (your choice of `n`/`s`/`m`/`l`/`x`) chosen with a written justification, trained for at least 30 epochs, training results logged.
- [ ] Test-set metrics reported in a table (mAP@0.5, mAP@0.5:0.95, P, R).
- [ ] Prediction visualisations on at least 6 test images, including failures.
- [ ] Reflection written.
- [ ] Notebook runs end-to-end without errors.

### How to submit

Paste a link to your Pull Request in the Student Portal.

## Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| Data handling | 20% | Correct YOLO-format inspection, splits, and `data.yaml` |
| Training quality | 25% | Working YOLO26 training run, sensible hyperparameters, training logs interpreted |
| Evaluation rigour | 25% | Correct test-set evaluation; mAP and P/R reported and explained |
| Visualisation & failure analysis | 15% | Clear box overlays, ground-truth comparisons, thoughtful failure-case analysis |
| Code quality & communication | 10% | Clean notebook, markdown explanations, reproducible repo |
| Reflection | 5% | Specific, honest, looking forward to Week 2 |

## Extension — Task 7: Hard-Negative Augmentation

The base `cats_v1` model is trained only on cats, so it has no concept of "not a
cat" and fires confident false positives on visually similar animals — dogs
especially. **Task 7** in the notebook fixes this without adding a new class:

- 300 dog images are streamed from the Stanford Dogs dataset (250 used for
  training, 50 held out as an unseen false-positive test).
- The 250 training dogs get **empty label files**, so YOLO treats them as
  *background* images — teaching the detector that a dog is not a cat.
- `data_aug.yaml` is identical to `data.yaml` except `train` points at
  `train_aug.txt` (cat train split + the 250 dog negatives). `val`/`test` are
  unchanged, so all cat metrics stay comparable.
- The model is retrained with the same hyperparameters as `cats_v1`, producing
  `runs/cats_v2/weights/best.pt`.

This stays a single-class cat detector — it just makes far fewer mistakes on
non-cats. Reproduce by running the notebook top-to-bottom; the Task 7 cells are
idempotent (they skip the download and training if the artefacts already exist).

## Looking Ahead

Next week you'll keep working with this **same dataset** and the **same YOLO26 family**. You'll:

1. **Improve** the detector using Week-2 techniques (a different YOLO26 variant, stronger augmentation, transfer-learning recipes, hyperparameter tuning)
2. **Export** the trained model to ONNX (one line — YOLO26's NMS-free design makes this clean)
3. **Containerise** the inference and ship a Docker image to a public registry, with a fixed CLI the instructor will run
4. **Compete** on a live class leaderboard scored on an unseen cat-detection holdout

So write your code with re-use in mind — keep your data loading, evaluation, and prediction utilities clean.
