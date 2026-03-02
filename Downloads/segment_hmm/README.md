# SegmentHMM

A Python-based Hidden Markov Model (HMM) pipeline for genomic window segmentation using *DXY (absolute divergence) with small-n smoothing* as Feature 2 and automatic model selection using BIC.

This tool is designed for large VCF datasets and HPC (SLURM) environments.

---

## Overview

segment_hmm performs:

1. Preprocessing of large VCF files  
2. Window-based feature computation  
3. DXY (absolute divergence) with small-n smoothing  
4. HMM model fitting  
5. Automatic BIC model selection  
6. Segmentation of genomic regions  
7. BIC visualization including number of windows  

---

## Features

- Handles large VCF files
- Designed for HPC cluster usage
- SLURM compatible
- Automatic BIC state selection
- Publication-ready BIC plots
- Python wheel distribution support

---

## Installation

### Option 1  Install from Wheel

bash
pip install segment_hmm-1.0.0-py3-none-any.whl


### Option 2  Install from Source

bash
git clone https://github.com/tmelialab/SegmentHMM
cd segment_hmm
pip install .


---

## Input Requirements

- VCF file
- Two varieties
- 3 samples per variety
- Indexed VCF recommended
- Large datasets supported

---

## Example Data (Test Dataset)

An example dataset is included inside:


example/test_data.vcf


This dataset contains:
- 2 varieties
- 3 samples each
- Small genomic region
- Suitable for testing installation

You can use it to verify installation.

---

## Usage

### Step 1  Preprocess VCF

bash
segment_hmm preprocess \
--vcf init_all_combine_snp_filter2.vcf \
--group1 Sampel1 Sampel2 Sampel3 \
--group2 Sampel1 Sampel2 Sampel3 \
--window 1000 5000 10000 20000 50000 \
--outdir output_preprocess


Output:

features.tsv


Contains:
- Window positions
- SNP counts
- DXY (absolute divergence with smoothing)

---

### Step 2  Model Selection (BIC)

bash
segment_hmm compute-bic \
--input output_preprocess/window_1000.tsv \
--input output_preprocess/window_5000.tsv \
--input output_preprocess/window_10000.tsv \
--input output_preprocess/window_20000.tsv \
--input output_preprocess/window_50000.tsv \
--outdir output_bic \
--states 2 3 4 5 6 7 8


Output:
- bic_plot.png
- Includes:
  - BIC curve
  - Number of windows
  - Optimal state selection

---

### Step 3  Build Final Model

bash
segment_hmm make-model \
--input output_preprocess/window_50000.tsv \
--n_states 8 \
--output_dir output_model


Output:

segmentation.tsv


Contains:
- Window
- Assigned HMM state

---

## Running on HPC (SLURM)

Example SLURM script:

bash
#!/bin/bash
#SBATCH --job-name=segment_hmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log.out

source ~/build_env/bin/activate

segment_hmm.preprocess \
--vcf init_all_combine_snp_filter2.vcf \
--group1 Sampel1 Sampel2 Sampel3 \
--group2 Sampel1 Sampel2 Sampel3 \
--window 1000 5000 10000 20000 50000 \
--outdir output_preprocess

segment_hmm.compute_bic \
--input output_preprocess/window_1000.tsv \
--input output_preprocess/window_5000.tsv \
--input output_preprocess/window_10000.tsv \
--input output_preprocess/window_20000.tsv \
--input output_preprocess/window_50000.tsv \
--outdir output_bic \
--states 2 3 4 5 6 7 8

segment_hmm.make_model \
--input output_preprocess/window_50000.tsv \
--n_states 8 \
--output_dir output_model


Submit job:

bash
sbatch run_segment.sh


---

## Mathematical Background

### DXY (Absolute Divergence)

For each genomic window:

\[
D_{XY} = \frac{1}{L} \sum (p_1 - p_2)^2
\]

Where:
- \( p_1 \) = allele frequency in variety 1
- \( p_2 \) = allele frequency in variety 2
- \( L \) = number of sites

Small-n smoothing is applied to stabilize allele frequency estimates in low sample size conditions.

---

## Output Files

| File | Description |
|------|------------|
| features.csv | Window-based computed features |
| bic_plot.png | BIC model selection plot |
| segmentation.csv | Final HMM state assignments |

---

## Requirements

- Python  3.9
- numpy
- pandas
- matplotlib
- scikit-learn
- hmmlearn

---

## Building the Wheel (For Developers)

Inside project root:

bash
python -m build


Generated files:


dist/
    segment_hmm-1.0.0-py3-none-any.whl
    segment_hmm-1.0.0.tar.gz


---

## Citation

If you use this tool in your research, please cite:


Author et al. (2026)
segment_hmm: HMM-based genomic segmentation using DXY divergence.


---

## License

MIT License

---

## Contact

For issues and contributions:

Open a GitHub Issue.

---

## Project Status

Active development  
Version: 1.0.0  
HPC-ready  
SLURM compatible  
Journal-ready
