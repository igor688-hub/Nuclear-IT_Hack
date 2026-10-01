import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent / "models"
CLASSES = ["Control", "Endo", "Exo"]


def baseline_als(y, lam=10**5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(spectra):
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    processed = np.zeros_like(spectra)

    for i, y in enumerate(spectra):
        if np.all(y == 0):
            continue
        corrected = y - baseline_als(y)
        processed[i] = savgol_filter(corrected, window_length=11, polyorder=3, mode="nearest")

    mean = processed.mean(axis=1, keepdims=True)
    std = processed.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return np.nan_to_num((processed - mean) / std, nan=0.0)


def read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=["X", "Y", "Wave", "Intensity"])
    coords = df[["X", "Y"]].drop_duplicates()
    n_pixels = len(coords)
    n_waves = len(df) // n_pixels

    waves = df["Wave"].values[:n_waves]
    matrix = df["Intensity"].values.reshape(n_pixels, n_waves)
    if waves[0] > waves[-1]:
        waves = waves[::-1]
        matrix = matrix[:, ::-1]

    wide = pd.DataFrame(matrix, columns=waves)
    wide.insert(0, "X", coords["X"].values)
    wide.insert(1, "Y", coords["Y"].values)
    return wide


def load_spectra(path):
    if path.suffix.lower() == ".txt":
        return read_txt(path)
    return pd.read_csv(path)


def wave_columns(df):
    columns, waves = [], []
    for c in df.columns:
        try:
            waves.append(float(c))
            columns.append(c)
        except (TypeError, ValueError):
            pass
    return columns, np.array(waves)


def main():
    if len(sys.argv) > 1:
        raw_path = sys.argv[1]
    else:
        raw_path = input("Path to a spectra file (.txt or .csv): ").strip().strip("\"'")
    path = Path(raw_path)

    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = load_spectra(path)
    columns, waves = wave_columns(df)
    if not columns:
        sys.exit("No wavenumber columns found in the file")

    window = "2900" if waves.mean() > 2200 else "1500"
    model = joblib.load(MODELS_DIR / f"model_{window}.pkl")
    train_waves = np.array(joblib.load(MODELS_DIR / f"cols_{window}.pkl"), dtype=float)

    print(f"Spectral window: {window} cm-1")
    print(f"Spectra in file: {len(df)}")
    print("Preprocessing...")

    interpolate = interp1d(waves, df[columns].values, axis=1, bounds_error=False, fill_value="extrapolate")
    X = preprocess(interpolate(train_waves))

    probs = model.predict_proba(X)
    pixel_preds = probs.argmax(axis=1)
    sample_probs = probs.mean(axis=0)
    verdict = CLASSES[sample_probs.argmax()]

    print()
    print("Pixel votes:")
    for i, name in enumerate(CLASSES):
        share = (pixel_preds == i).mean() * 100
        print(f"  {name:<8} {share:5.1f}%")
    print()
    print(f"Sample diagnosis: {verdict.upper()} ({sample_probs.max() * 100:.1f}%)")

    result = pd.DataFrame({
        "prediction": [CLASSES[p] for p in pixel_preds],
        "confidence": probs.max(axis=1).round(4),
    })
    if {"X", "Y"}.issubset(df.columns):
        result.insert(0, "X", df["X"].values)
        result.insert(1, "Y", df["Y"].values)

    out_path = path.with_name(f"{path.stem}_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"Per-pixel results saved to {out_path}")


if __name__ == "__main__":
    main()
