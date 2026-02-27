import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.fft import rfft, rfftfreq

def extract_features(x, fs=5120, f0=50):
    """
    Hand-crafted features for electrical signals:
    - RMS, mean, std, peak-to-peak, crest factor
    - Band energies around fundamental + harmonics (50, 150, 250 Hz)
    - THD-like proxy: energy(harmonics)/energy(fundamental)
    """
    x = np.asarray(x, dtype=float)
    eps = 1e-12

    rms = np.sqrt(np.mean(x**2))
    mean = np.mean(x)
    std = np.std(x)
    ptp = np.ptp(x)
    peak = np.max(np.abs(x))
    crest = peak / (rms + eps)

    # FFT-based energies
    n = len(x)
    X = np.abs(rfft(x))**2
    freqs = rfftfreq(n, d = 1/fs)

    def band_energy(center_hz, bw = 10.0):
        m = (freqs >= center_hz - bw) & (freqs <= center_hz + bw)
        return float(np.sum(X[m]))

    e1 = band_energy(f0)
    e3 = band_energy(3*f0)
    e5 = band_energy(5*f0)

    thd_proxy = (e3 + e5) / (e1 + eps)

    return [rms, mean, std, ptp, crest, e1, e3, e5, thd_proxy]

def load_and_build_xy(csv_path = "data/signals.csv", fs = 5120, f0 = 50):
    df = pd.read_csv(csv_path)
    x_cols = [c for c in df.columns if c.startswith("x")]

    feats = []
    for _, row in df.iterrows():
        x = row[x_cols].to_numpy(dtype=float)
        feats.append(extract_features(x, fs = fs, f0 = f0))

    X = np.array(feats, dtype=float)
    y = df["label"].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()
    return X, y, split

def main(csv_path = "data/signals.csv", out_dir = "output", fs = 5120, f0 = 50, C = 10.0, gamma = "scale"):
    os.makedirs(out_dir, exist_ok=True)

    X, y, split = load_and_build_xy(csv_path, fs, f0)
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C = C, gamma = gamma, kernel = "rbf"))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict = True)
    with open(os.path.join(out_dir, "svm_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels = np.unique(y))
    disp = ConfusionMatrixDisplay(cm, display_labels = np.unique(y))
    plt.figure(figsize=(7, 6))
    disp.plot(values_format = "d")
    plt.title("SVM Baseline - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "svm_confusion.png"), dpi = 150)
    plt.close()

    print("SVM done.")
    print("Saved:", os.path.join(out_dir, "svm_report.json"))
    print("Saved:", os.path.join(out_dir, "svm_confusion.png"))
    print("Macro F1:", report["macro avg"]["f1-score"])
    print("Accuracy:", report["accuracy"])

if __name__ == "__main__":
    main()