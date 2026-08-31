#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def compute_bic(model, X):
    log_likelihood = model.score(X)
    n_samples, n_features = X.shape
    n_states = model.n_components

    n_params = (
        n_states * (n_states - 1) +
        (n_states - 1) +
        n_states * n_features +
        n_states * (n_features * (n_features + 1) / 2)
    )

    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic


def main():

    parser = argparse.ArgumentParser(
        description="Compute BIC for multiple window sizes and HMM states"
    )

    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    # OUTPUT FILE
    bic_output_file = os.path.join(args.outdir, "bic_values.txt")

    # BUAT FILE DARI AWAL
    with open(bic_output_file, "w") as f:
        f.write("window_size\tstate\tbic\n")

    total_written = 0

    for file in args.input:

        print(f"\n[INFO] Processing file: {file}")

        try:
            df = pd.read_csv(file, sep="\t")
        except Exception as e:
            print(f"[ERROR] Cannot read file: {e}")
            continue

        df.columns = df.columns.str.strip()
        features = ["feature_1", "feature_2", "feature_3"]

        # CLEAN DATA
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features)

        if df.empty:
            print("[WARNING] Empty after cleaning, skip.")
            continue

        # FILTER CHROM
        if "CHROM" in df.columns:
            counts = df["CHROM"].value_counts()
            valid = counts[counts > 1].index
            df = df[df["CHROM"].isin(valid)]

        if df.empty:
            print("[WARNING] Empty after CHROM filter, skip.")
            continue

        X = df[features].values

        if len(X) < 2:
            print("[WARNING] Not enough samples.")
            continue

        # REMOVE ZERO VAR
        var = np.var(X, axis=0)
        X = X[:, var > 0]

        if X.shape[1] == 0:
            print("[WARNING] All features zero variance.")
            continue

        # SCALE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_scaled = X_scaled[np.isfinite(X_scaled).all(axis=1)]

        if len(X_scaled) < 2:
            print("[WARNING] Not enough valid samples after scaling.")
            continue

        print(f"[INFO] Data shape: {X_scaled.shape}")

        # SAFE WINDOW NAME
        try:
            window_size = os.path.basename(file).split("_")[1].split(".")[0]
        except:
            window_size = os.path.basename(file)

        # SIMPAN UNTUK PLOT
        bic_values = []

        # ==============================
        # LOOP STATES
        # ==============================
        for k in args.states:

            if len(X_scaled) <= k:
                print(f"[WARNING] Skip state {k}")
                bic_str = "NaN"
                bic_values.append(np.nan)

            else:
                try:
                    model = GaussianHMM(
                        n_components=k,
                        covariance_type="full",
                        n_iter=200,
                        random_state=42
                    )

                    model.fit(X_scaled)
                    bic_val = compute_bic(model, X_scaled)

                    bic_str = str(bic_val)
                    bic_values.append(bic_val)

                    print(f"[INFO] State={k}, BIC={bic_val:.2f}")

                except Exception as e:
                    print(f"[ERROR] State {k}: {e}")
                    bic_str = "NaN"
                    bic_values.append(np.nan)

            # WRITE KE FILE
            with open(bic_output_file, "a") as f:
                f.write(f"{window_size}\t{k}\t{bic_str}\n")

            total_written += 1

        # ==============================
        # PLOT PER WINDOW (FIX)
        # ==============================
        valid_states = [s for s, b in zip(args.states, bic_values) if not np.isnan(b)]
        valid_bic = [b for b in bic_values if not np.isnan(b)]

        if len(valid_bic) > 0:
            plt.plot(valid_states, valid_bic, marker="o", label=f"{window_size} bp")

    print(f"\n[INFO] Total rows written: {total_written}")
    print(f"[INFO] Output saved: {bic_output_file}")

    # ==============================
    # FINAL PLOT
    # ==============================
    plt.xlabel("Number of HMM States")
    plt.ylabel("BIC (lower is better)")
    plt.title("BIC Comparison Across Window Sizes")
    plt.legend(title="Window Size")
    plt.grid(True)

    out_plot = os.path.join(args.outdir, "bic_multi_window.png")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Plot saved: {out_plot}")
    print("[DONE]")


if __name__ == "__main__":
    main()
