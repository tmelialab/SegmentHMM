#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from collections import defaultdict
import os


def safe_mean(arr):
    if len(arr) == 0:
        return np.nan
    return np.mean(arr)


def allele_freq(genotypes):

    alleles = []

    for g in genotypes:
        a, b = g[0], g[1]

        if a >= 0:
            alleles.append(a)
        if b >= 0:
            alleles.append(b)

    if len(alleles) == 0:
        return np.nan

    return np.mean(alleles)


def compute_windows(vcf_file, group1, group2, window_sizes):

    print("[INFO] Opening VCF...")
    vcf = VCF(vcf_file)

    samples = vcf.samples

    g1_idx = [samples.index(x) for x in group1]
    g2_idx = [samples.index(x) for x in group2]

    print(f"[INFO] Group1 samples: {len(g1_idx)}")
    print(f"[INFO] Group2 samples: {len(g2_idx)}")

    windows = {w: defaultdict(lambda: defaultdict(list)) for w in window_sizes}

    snp_counter = 0

    for variant in vcf:

        snp_counter += 1

        if snp_counter % 100000 == 0:
            print(f"[INFO] Processed SNPs: {snp_counter}")

        chrom = variant.CHROM
        pos = variant.POS

        gts = variant.genotypes

        g1 = [gts[i] for i in g1_idx]
        g2 = [gts[i] for i in g2_idx]

        p1 = allele_freq(g1)
        p2 = allele_freq(g2)

        if np.isnan(p1) or np.isnan(p2):
            continue

        fst = (p1 - p2) ** 2
        dxy = abs(p1 - p2)

        for w in window_sizes:

            win = pos // w

            windows[w][chrom][win].append((fst, dxy))

    print(f"[INFO] Total SNP processed: {snp_counter}")

    return windows


def build_dataframe(windows, window_size):

    records = []

    for chrom in windows:

        for win in sorted(windows[chrom].keys()):

            start = win * window_size
            end = start + window_size

            data = windows[chrom][win]

            fst_vals = [x[0] for x in data]
            dxy_vals = [x[1] for x in data]

            snp_count = len(data)

            if snp_count == 0:
                continue

            feature_1 = safe_mean(fst_vals)
            feature_2 = safe_mean(dxy_vals)
            feature_3 = snp_count

            records.append(
                [chrom, start, end, feature_1, feature_2, feature_3]
            )

    df = pd.DataFrame(
        records,
        columns=["CHROM", "START", "END", "feature_1", "feature_2", "feature_3"],
    )

    print(f"[INFO] Windows before cleaning ({window_size}): {len(df)}")

    for col in ["feature_1", "feature_2", "feature_3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)

    df = df.dropna(subset=["feature_1", "feature_2", "feature_3"])

    removed = before - len(df)

    if removed > 0:
        print(f"[INFO] Removed {removed} NaN windows ({window_size})")

    counts = df["CHROM"].value_counts()

    bad_chrom = counts[counts == 1].index

    if len(bad_chrom) > 0:

        print("[INFO] Removing chromosomes with only one window:")

        for c in bad_chrom:
            print(f"   - {c}")

        df = df[~df["CHROM"].isin(bad_chrom)]

    print(f"[INFO] Final windows ({window_size}): {len(df)}")

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--vcf", required=True)
    parser.add_argument("--group1", nargs="+", required=True)
    parser.add_argument("--group2", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", type=int, required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    windows = compute_windows(
        args.vcf,
        args.group1,
        args.group2,
        args.windows,
    )

    for w in args.windows:

        print(f"\n[INFO] Building dataframe for window {w}")

        df = build_dataframe(windows[w], w)

        outfile = f"{args.outdir}/window_{w}.tsv"

        df.to_csv(outfile, sep="\t", index=False)

        print(f"[INFO] Saved: {outfile}")


if __name__ == "__main__":
    main()
