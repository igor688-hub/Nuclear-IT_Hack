# Brain Tissue Classification from Raman Spectra

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A machine learning pipeline that reads Raman scans of brain tissue and tells which experimental group a sample belongs to. Built by team **Bokom_m** at the Nuclear IT Hack hackathon.

## About

Raman spectroscopy shows the chemical makeup of tissue without damaging it. Every scan is a small map where each pixel has its own spectrum. We used scans of the cortex, striatum and cerebellum to tell apart three groups:

- **Control**: healthy tissue
- **Endo**: tissue under stress, with the cell's own HSP70 response
- **Exo**: tissue under stress, with HSP70 added from outside

HSP70 is a heat shock protein that helps cells survive stress. The question behind the project: does its protective effect leave a trace in the spectrum that a model can pick up?

Scans come in two spectral windows that we treat separately:

- **1500 cm⁻¹**: proteins and nucleic acids
- **2900 cm⁻¹**: lipids, mostly cell membranes

## How it works

1. **Resampling.** Every spectrum is put onto the same wavenumber grid.
2. **Cleaning.** ALS baseline correction removes the fluorescence background, a Savitzky-Golay filter smooths out noise, and SNV normalization evens out differences in overall intensity.
3. **Pixel classification.** A LightGBM model predicts the group for every pixel.
4. **Sample diagnosis.** Pixel probabilities are averaged into one answer for the whole sample.

The train/test split is done by file, so pixels from the same tissue sample never appear on both sides. Without this the model would just memorize individual slices and the scores would look much better than they really are.

## What the data looks like

Average spectra of each group after cleaning:

<p>
  <img src="images/mean_spectra_1500.png" width="49%">
  <img src="images/mean_spectra_2900.png" width="49%">
</p>

The difference from Control shows where the groups actually diverge:

<p>
  <img src="images/diff_spectra_1500.png" width="49%">
  <img src="images/diff_spectra_2900.png" width="49%">
</p>

The differences are subtle, and PCA shows that the groups overlap heavily. That is why we went with gradient boosting instead of a simple linear model.

<details>
<summary>PCA projection (protein window)</summary>
<img src="images/pca_1500.png" width="60%">
</details>

## Results

Accuracy on held-out samples:

| Window | Accuracy |
|---|---|
| Protein (1500) | 83% |
| Lipid (2900) | 79% |

![Confusion matrices](images/confusion_matrices.png)

The two windows complement each other. The protein model never missed a healthy sample, and the lipid model never missed an Exo sample. This suggests a two-step check: use the protein window to screen out healthy tissue first, then use the lipid window to tell Endo from Exo.

The model relies most on peaks around 1290 and 1435 cm⁻¹ (protein structure) and around 2880 cm⁻¹ (lipid chains in membranes), which matches what you would expect from changes in proteins and membranes under stress.

> The test set is small (24 samples per window), so treat these numbers as a rough estimate.
> The bundled `model_1500.pkl` is the tuned version from the notebook. It uses balanced class weights and scored a bit lower on the same split than the base model behind the 83% figure.

## Quick start

```bash
git clone https://github.com/igor688-hub/Nuclear-IT_Hack.git
cd Nuclear-IT_Hack
pip install -r requirements.txt
python predict.py path/to/scan.txt
```

The script accepts:

- a raw `.txt` export from the microscope with columns `X Y Wave Intensity` and one header line
- a `.csv` with one spectrum per row and wavenumbers as column names

The spectral window is detected automatically. The script prints how the pixels voted and the final diagnosis for the sample, then saves per-pixel predictions next to the input file as `<name>_predictions.csv`.

## Training

Everything from raw scans to saved models is in [`notebooks/model_training.ipynb`](notebooks/model_training.ipynb).

The raw scans are not included in this repository. To retrain, put the `.txt` files into `data/`. File names should contain the group (`control`, `endo`, `exo`) and the window (`center1500` or `center2900`).

## Repository structure

```
├── predict.py                  inference on a new scan
├── models/                     trained models and wavenumber grids
├── notebooks/
│   └── model_training.ipynb    preprocessing, training, evaluation
├── docs/
│   ├── report_ru.pdf           full report (Russian)
│   └── presentation_ru.pptx    hackathon slides (Russian)
└── images/                     figures for this README
```

## Team

**Bokom_m**: Anastasia Ilkaeva, Evgeny Utushkin, Igor Yatsun

## License

[MIT](LICENSE)
