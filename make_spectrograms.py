import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def save_spec_image(x, fs, out_png):
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=256, noverlap=192)
    S = np.log10(Sxx + 1e-10)

    
    plt.figure(figsize = (2.24, 2.24), dpi = 100)  # ~224x224
    plt.axis("off")
    plt.imshow(S, aspect = "auto", origin = "lower")
    plt.tight_layout(pad = 0)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches = "tight", pad_inches = 0)
    plt.close()

def main(csv_path = "data/signals.csv", img_root = "images", fs=5120):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("x")]

    for i, row in df.iterrows():
        label = row["label"]
        split = row["split"]
        x = row[feature_cols].to_numpy(dtype=float)

        out_png = os.path.join(img_root, split, label, f"{i}.png")
        save_spec_image(x, fs, out_png)

    print("Done. Images saved under:", img_root)

if __name__ == "__main__":
    main()