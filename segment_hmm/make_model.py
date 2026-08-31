#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def main():

    parser = argparse.ArgumentParser(
        description="Build Gaussian HMM and perform genome segmentation"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--n_states", type=int, required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    features = ["feature_1", "feature_2", "feature_3"]

    df = df.dropna(subset=features)

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=args.n_states,
        covariance_type="full",
        n_iter=300,
        random_state=42
    )

    model.fit(X_scaled)
    df["state"] = model.predict(X_scaled)

    # ==============================
    # OUTPUT 1: SEGMENTATION
    # ==============================
    out_state = os.path.join(args.outdir, "segmentation_result.tsv")
    df.to_csv(out_state, sep="\t", index=False)

    # ==============================
    # Merge consecutive windows with the same state
    # ==============================

    merged = []

    current = df.iloc[0]

    chrom = current["CHROM"]
    state = current["state"]

    # Support either uppercase or lowercase column names
    start_col = "START" if "START" in df.columns else "start"
    end_col = "END" if "END" in df.columns else "end"

    start = current[start_col]
    end = current[end_col]

    f1 = [current["feature_1"]]
    f2 = [current["feature_2"]]
    f3 = [current["feature_3"]]

    n_windows = 1

    for i in range(1, len(df)):

        row = df.iloc[i]

        if row["CHROM"] == chrom and row["state"] == state:

            end = row[end_col]

            f1.append(row["feature_1"])
            f2.append(row["feature_2"])
            f3.append(row["feature_3"])

            n_windows += 1

        else:

            merged.append([
                chrom,
                start,
                end,
                end - start,
                state,
                np.mean(f1),
                np.mean(f2),
                np.mean(f3),
                n_windows
            ])

            chrom = row["CHROM"]
            state = row["state"]

            start = row[start_col]
            end = row[end_col]

            f1 = [row["feature_1"]]
            f2 = [row["feature_2"]]
            f3 = [row["feature_3"]]

            n_windows = 1

    # Save last region
    merged.append([
        chrom,
        start,
        end,
        end - start,
        state,
        np.mean(f1),
        np.mean(f2),
        np.mean(f3),
        n_windows
    ])

    merged_df = pd.DataFrame(
        merged,
        columns=[
            "CHROM",
            "START",
            "END",
            "LENGTH",
            "state",
            "mean_feature_1",
            "mean_feature_2",
            "mean_feature_3",
            "n_windows"
        ]
    )

    out_merge = os.path.join(args.outdir, "merged_segmentation.tsv")
    merged_df.to_csv(out_merge, sep="\t", index=False)

    print(f"[INFO] Merged segmentation saved to {out_merge}")

    # ==============================
    # OUTPUT 2: SUMMARY PER STATE
    # ==============================
    summary = []

    for state in sorted(df["state"].unique()):
        subset = df[df["state"] == state]

        mean_f1 = subset["feature_1"].mean()
        mean_f2 = subset["feature_2"].mean()
        mean_f3 = subset["feature_3"].mean()
        n_windows = len(subset)

        summary.append([state, mean_f1, mean_f2, mean_f3, n_windows])

    summary_df = pd.DataFrame(summary, columns=[
        "state", "mean_feature_1", "mean_feature_2", "mean_feature_3", "n_windows"
    ])

    out_summary = os.path.join(args.outdir, "result_summary.txt")
    summary_df.to_csv(out_summary, sep="\t", index=False)

    print(f"[INFO] Summary saved to {out_summary}")

    # ==============================
    # VISUALIZATION
    # ==============================

    # 1-3: BOXPLOTS
    for feature in features:
        plt.figure()
        df.boxplot(column=feature, by="state")
        plt.title(f"Boxplot of {feature} by State")
        plt.suptitle("")
        plt.xlabel("State")
        plt.ylabel(feature)

        out_box = os.path.join(args.outdir, f"boxplot_{feature}.png")
        plt.savefig(out_box, dpi=300, bbox_inches="tight")
        plt.close()

    # ==============================
    # 4: EMISSION (MEAN PER STATE)
    # ==============================
    means = model.means_

    plt.figure()
    for i in range(means.shape[1]):
        plt.plot(range(args.n_states), means[:, i], marker='o', label=f"Feature {i+1}")

    plt.xlabel("State")
    plt.ylabel("Mean Value")
    plt.title("Emission Means")
    plt.legend()

    out_emission = os.path.join(args.outdir, "emission_means.png")
    plt.savefig(out_emission, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================
    # 5: TRANSITION MATRIX
    # ==============================
    transmat = model.transmat_

    plt.figure()
    plt.imshow(transmat)
    plt.colorbar()
    plt.title("Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")

    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            plt.text(j, i, f"{transmat[i, j]:.2f}",
                     ha="center", va="center", color="white")

    out_trans = os.path.join(args.outdir, "transition_matrix.png")
    plt.savefig(out_trans, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Segmentation saved to {out_state}")
    print("[INFO] All visualizations saved")
    print("[DONE] Model training completed")


if __name__ == "__main__":
    main()
