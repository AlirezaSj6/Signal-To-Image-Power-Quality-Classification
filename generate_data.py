import os
import numpy as np
import pandas as pd

def base_signal(n=1024, fs=5120, f0=50, noise_std=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    t = np.arange(n) / fs
    x = np.sin(2*np.pi*f0*t)
    x += rng.normal(0, noise_std, size=n)
    return x

def add_harmonics(x, fs=5120, f0=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    t = np.arange(n) / fs
    a3 = rng.uniform(0.08, 0.25)
    a5 = rng.uniform(0.05, 0.18)
    return x + a3*np.sin(2*np.pi*3*f0*t) + a5*np.sin(2*np.pi*5*f0*t)

def apply_sag(x, rng):
    y = x.copy()
    n = len(y)
    start = rng.integers(n//6, n//2)
    length = rng.integers(n//10, n//4)
    depth = rng.uniform(0.25, 0.6)  # amplitude reduction
    y[start:start+length] *= (1.0 - depth)
    return y

def apply_swell(x, rng):
    y = x.copy()
    n = len(y)
    start = rng.integers(n//6, n//2)
    length = rng.integers(n//10, n//4)
    inc = rng.uniform(0.25, 0.6)
    y[start:start+length] *= (1.0 + inc)
    return y

def apply_transient(x, rng):
    y = x.copy()
    n = len(y)
    idx = rng.integers(n//5, 4*n//5)
    amp = rng.uniform(0.5, 1.2) * (1 if rng.random() > 0.5 else -1)
    width = rng.integers(3, 10)
    for k in range(width):
        if idx + k < n:
            y[idx + k] += amp * np.exp(-k / max(1, width/2))
    return y

def generate_dataset(out_path = "data/signals.csv", n_points = 1024, fs = 5120, f0 = 50, n_per_class = 500, test_ratio = 0.2, seed = 42):
    rng = np.random.default_rng(seed)
    classes = ["normal", "harmonics", "sag", "swell", "transient"]
    rows = []

    for label in classes:
        for _ in range(n_per_class):
            x = base_signal(n = n_points, fs = fs, f0 = f0, rng = rng)

            if label == "harmonics":
                x = add_harmonics(x, fs = fs, f0 = f0, rng = rng)
            elif label == "sag":
                x = apply_sag(x, rng)
            elif label == "swell":
                x = apply_swell(x, rng)
            elif label == "transient":
                x = apply_transient(x, rng)

            rows.append([label, *x.tolist()])

    df = pd.DataFrame(rows, columns = ["label"] + [f"x{i}" for i in range(n_points)])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cut = int(len(df) * (1 - test_ratio))
    df["split"] = "train"
    df.loc[cut:, "split"] = "test"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "| rows:", len(df), "| train:", cut, "| test:", len(df)-cut)

if __name__ == "__main__":
    generate_dataset()