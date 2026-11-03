# Hand localization and validation protocol effects on static hand gesture classification

Code and results for the manuscript *Hand localization and validation protocol effects on static hand gesture classification* (Nurzhan Nurlybek, Astana IT University).

The study measures how three choices change the measured accuracy of a five-class static gesture classifier (fist, stop, like, peace, ok):
the way the hand is localized (Experiment A), the validation protocol (Experiment B) and the image source (Experiment C).
The data are 500 balanced images: 250 HaGRID crops (234 people) and 250 phone photographs taken by the author in 25 shooting situations.
Models: MobileNetV2 (ImageNet weights, two-stage fine-tuning) and a baseline of HOG features with an RBF SVM. Every experiment uses seeds 0, 1 and 2.

## Main results (accuracy, %)

| Experiment | Setting | Accuracy |
|---|---|---|
| A, localization (500 unseen HaGRID photographs) | annotated box, margin 0% | 94.3 |
| | annotated box, margin 25% (reference) | 91.9 |
| | noisy box, error 50% | 79.9 |
| | MediaPipe hand detector, margin 25% | 85.6 |
| | whole photograph, squeezed to a square | 24.8 |
| B, validation protocol (MobileNetV2) | random 5-fold cross-validation | 93.7 ± 1.8 |
| | group 5-fold cross-validation | 93.2 ± 1.0 |
| | single group-aware split (about 68 test images, 12 splits) | 92.5 ± 4.9 (range 82.5–98.2) |
| C, transfer (MobileNetV2) | HaGRID to own photographs | 79.6 ± 2.1 |
| | own photographs to HaGRID | 78.8 ± 2.4 |
| | in-domain, HaGRID / own photographs | 90.1 / 93.6 |
| | HaGRID to own photographs without low-resolution augmentation | 79.3 ± 2.3 |

± is the standard deviation over seeds (or over splits). All tables are in `results/tables/`.

## Repository contents

```
Experiments_A_B_C.ipynb   notebook of all experiments (Google Colab, T4 GPU); saves per-image predictions of every run
reproduce_tables.py       recomputes all tables and figures A, B, C from the saved predictions (no GPU, no images)
results/
  A_seed*.csv, B_seed*.csv, C_seed*.csv   predictions of every run (one row per test image and variant)
  A_raw_photos.csv, A_mediapipe_boxes.csv raw HaGRID test photographs of Experiment A, annotated and detected boxes
  run_info.json                           software and hardware of the reported run
  tables/                                 result tables
  figures/                                figures of the paper (600 dpi)
data/                     image metadata (see data/README.md)
```

## Reproducing the tables and figures

```
pip install -r requirements.txt
python reproduce_tables.py results
```

This rewrites `results/tables/*.csv` and the figures `fig_A_*`, `fig_B_*`, `fig_C_*` from the per-image predictions; the recomputed tables are identical to the published ones (the bootstrap uses a fixed seed).

## Re-running the experiments

Open `Experiments_A_B_C.ipynb` in Google Colab with a T4 GPU, put `dataset.zip` (500 images and the two metadata files) into `MyDrive/CV_Assignment1/` and run all cells.
The notebook downloads the public HaGRID sample (384 px) from Kaggle and the MediaPipe hand landmarker model, and takes about 100 minutes for three seeds.
Finished steps are saved and skipped when the notebook is run again.
The reported run used Python with PyTorch 2.11.0+cu130 and MediaPipe 0.10.32 on a Tesla T4.

The 250 own photographs show the author and are not published; they are available on request (241691@astanait.edu.kz).

## Data and licenses

- Code: MIT license (`LICENSE`).
- HaGRID images and annotations: CC BY-SA 4.0, Kapitanov et al., *HaGRID – HAnd Gesture Recognition Image Dataset*, WACV 2024.
- Result tables and figures: CC BY 4.0. The face of the person in Figure 1 is blurred.
