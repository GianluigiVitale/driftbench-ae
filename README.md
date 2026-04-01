# DriftBench Artifact Evaluation

## Abstract

This artifact validates "DriftBench: Measuring and Predicting Infrastructure Drift in LLM Serving Systems" through three self-contained evaluation paths: (1) GPU reproduction to reproduce H100→B200 drift (30 min), (2) PRI model retraining reproducing all 10 R² values from Table 2 (5 min), or (3) automated verification of all 34 numerical claims (1 min). Complete dataset: 236,985 prompt-response pairs across 105 configurations (5 models, 4 GPUs, 3 frameworks, 3 precisions). All code (MIT) and data (CC BY 4.0) are open-source.

---

## Artifact check-list (meta-information)

- **Algorithm:** Gradient Boosting (scikit-learn) for PRI prediction; flip rate calculation; semantic similarity
- **Program:** Python 3.8+, verification scripts, DriftBench CLI
- **Compilation:** None (interpreted Python)
- **Binary:** Pre-trained PRI model (`pri_model.pkl`, 1.7MB)
- **Data set:** 236,985 prompt-response pairs (400MB); 2,257 prompts across 5 workloads (Code: 164, Math: 500, Safety: 520, Chat: 973, Long-context: 100); 516 safety + 100 semantic similarity human annotations
- **Run-time environment:** Linux/macOS/Windows, Python 3.8+, numpy, pandas, scipy, scikit-learn
- **Hardware:** CPU only (4GB RAM). NVIDIA H100 80GB + B200 192GB for GPU reproduction
- **Execution:** Command-line scripts, Python API
- **Metrics:** Flip rate (% of prompts where correctness classification changed between configurations: correct↔incorrect, safe↔unsafe), R², Spearman ρ, F-statistics
- **Output:** JSON reports, CSV measurements, pass/fail verification
- **Experiments:** GPU reproduction (30min), PRI retraining (5min), Automated verification (1min)
- **How much disk space required (approximately)?:** 100MB (CPU-only). ~70GB for GPU reproduction (model weights)
- **How much time is needed to prepare workflow (approximately)?:** Path B/C: 5 minutes. Path A: 30 minutes–1 hour
- **How much time is needed to complete experiments (approximately)?:** 1–30 minutes depending on path
- **Publicly available?:** Yes (GitHub + Zenodo: [https://doi.org/10.5281/zenodo.19361066](https://doi.org/10.5281/zenodo.19361066))
- **Code licenses (if publicly available)?:** MIT
- **Data licenses (if publicly available)?:** CC BY 4.0
- **Workflow framework used?:** Python scripts, bash scripts
- **Archived (provide DOI)?:** [10.5281/zenodo.19361066](https://doi.org/10.5281/zenodo.19361066)

---

## Description

### How delivered

Public repositories:

1. **Docker Hub:** `gianluigivitale/driftbench-ae:latest` — Pre-configured Docker image for GPU reproduction.
2. **Artifact Evaluation:** https://github.com/GianluigiVitale/driftbench-ae — All three evaluation paths (A, B, C), scripts, data, and GPU reproduction package.
3. **DriftBench CLI:** https://github.com/GianluigiVitale/driftbench — Analyze existing results from 105 configurations.

Full dataset (236,985 pairs, 400MB) archived on Zenodo: [https://doi.org/10.5281/zenodo.19361066](https://doi.org/10.5281/zenodo.19361066) — **not required** for evaluation (all verification uses local CSV summaries).

---

## Quick Start: Choose Your Evaluation Path

### Path A: GPU Reproduction (H100 + B200, 30 min)

**What:** Reproduce Section 3.3 production case end-to-end with live inference on 520 prompts.

**Requirements:** H100 80GB + B200 192GB GPUs. **Note:** If you don't have both GPUs available, we provide a $20 credit coupon to rent them on RunPod (B200: $4.99/h, H100 SXM: $2.79/h). See "Optional: RunPod Setup Guide" section below for deployment instructions. Docker is pre-configured, so you only need to download models and run experiments—very easy and fast.

**HuggingFace token (provided):** Token provided to evaluators in artifact appendix PDF.

**Why we provide this token:** Llama models require authorization from Meta, which can take weeks to obtain. We provide this read-only API key for evaluation convenience. Please use it appropriately—it will be deleted after the artifact evaluation period ends.

**Steps:**
```bash
# Start Docker container with exact image
docker run --gpus all -it gianluigivitale/driftbench-ae:latest

# Inside container
cd /workspace
git clone https://github.com/GianluigiVitale/driftbench-ae.git
cd driftbench-ae/artifact_evaluation/reproduction_scripts

# Login with provided token (see artifact appendix PDF)
huggingface-cli login --token <TOKEN_FROM_APPENDIX>

# Download models (~50GB, 10-15 min parallel)
bash download_models.sh

# Run experiments (--hardware flag ensures correct GPU selection)
python parallel_controller.py --hardware h100 --workers 1  # ~5 min
python parallel_controller.py --hardware b200 --workers 1  # ~5 min

# Compute flip rate (requires both H100 and B200 results)
cd driftbench-ae/artifact_evaluation/reproduction_scripts
python compute_direct_flip_rate.py
```

**Expected:** Flip rate ~24% (±2–3% due to LlamaGuard).

**Multi-instance setup:** If H100 and B200 are on separate instances, run experiments separately and manually transfer the `results/` folder from `driftbench-ae/artifact_evaluation/reproduction_scripts/results/`. Copy it from the H100 instance to the B200 instance (or vice versa) before running `compute_direct_flip_rate.py`.

---

### Path B: PRI Model Retraining (CPU-only, 5 min)

**What:** Retrain PRI model from raw data to reproduce all 10 R² values in Table 2 (Hardware, Precision, Framework, Model dimensions with Training + Test R² each).

**Requirements:** Python 3.8+, standard libraries.

**Steps:**
```bash
# Clone artifact evaluation repository
git clone https://github.com/GianluigiVitale/driftbench-ae.git

# Install dependencies
pip install numpy pandas scipy scikit-learn joblib

# Run PRI retraining
cd driftbench-ae/artifact_evaluation/pri_model_recreation/
python prepare_pri_dataset.py
python train_pri_enhanced.py
python validate_generalization_enhanced.py
```

**Expected:** All 10 R² values match Table 2 from the paper exactly (Train R²: 4×1.000; Test R²: Hardware 0.909, Precision 0.763, Framework 0.479, Model 0.118).

**Table: PRI Generalization to Held-Out Dimensions (Table 2 from paper)**

| Dimension | Train | Test | Tr R² | Te R² |
|-----------|-------|------|-------|-------|
| Hardware | H100/H200/MI300X | B200 | 1.000 | 0.909 |
| Precision | FP16/FP8 | FP4 | 1.000 | 0.763 |
| Framework | vLLM/TensorRT-LLM | SGLang | 1.000 | 0.479 |
| Model | Llama/Mistral/Mixtral | Qwen | 1.000 | 0.118 |
| **Mean** | | | **1.000** | **0.567** |

*Tr: Training set; Te: Test set (held-out dimension)*

---

### Path C: Automated Verification (CPU-only, 1 min)

**What:** Verify all 34 numerical claims via Python scripts.

**Requirements:** Python 3.8+, standard libraries.

**Steps:**
```bash
# Clone repository
git clone https://github.com/GianluigiVitale/driftbench-ae.git

# Install dependencies
pip install numpy pandas scipy scikit-learn joblib

# Verify all claims
cd driftbench-ae/artifact_evaluation/reviewer-verification-master/verification_scripts/
python verification_master_script.py
```

**Expected:** 34/34 checks pass.

---

## Hardware & Software

**Path A (GPU):** Pre-built Docker image `gianluigivitale/driftbench-ae:latest` includes CUDA 12.8.1, PyTorch 2.8.0, SGLang 0.5.2, and all dependencies. No installation needed.

**Paths B & C (CPU):** Any modern CPU, 4GB RAM, Python 3.8+, standard scientific libraries.

---

## Data

All data included in repositories:

- `flip_rates.csv` (420 measurements from 105 configs × 4 objective workloads)
- `generalization_results.csv` (PRI held-out validation: Hardware R²=0.909, Precision R²=0.763)
- Prompt datasets (2,257 prompts: Code 164, Math 500, Safety 520, Chat 973, Long-context 100)
- Human annotations: 516 safety labels (safe/unsafe outputs), 100 semantic similarity labels (chat baseline comparison)
- Complete experimental matrix: 525 experiments (84% coverage: 105 of 125 planned configs)

Full dataset (236,985 pairs, 400MB) archived on Zenodo: [https://doi.org/10.5281/zenodo.19361066](https://doi.org/10.5281/zenodo.19361066) — **not required** for evaluation.

---

## Key Claims Mapping

- **Table 2 (PRI Generalization):** Path B (PRI retraining), Path C (verification scripts)
- **Section 3.3 (Production case):** Path A (GPU reproduction)
- **All 34 numerical claims:** Path C (verification master script)

---

## Installation

**Paths B & C (CPU-only):**
```bash
# Clone repository
git clone https://github.com/GianluigiVitale/driftbench-ae.git

# Install dependencies
pip install numpy pandas scipy scikit-learn joblib
```

**Path A (GPU):** No installation needed—Docker image has everything pre-configured.

---

## Methodology

Submission, reviewing and badging methodology:

- http://cTuning.org/ae/submission-20190109.html
- http://cTuning.org/ae/reviewing-20190109.html
- https://www.acm.org/publications/policies/artifact-review-badging

---

## Notes

- **GPU not required:** Paths B and C are CPU-only.
- **Deterministic:** Paths B and C use pre-computed CSV data. Path A has ±2–3% variance (expected).
- **Time estimates:** Path A (30 min), Path B (5 min), Path C (1 min).
- **Tested on:** Ubuntu 22.04, Windows 11 (WSL2).

A companion repository with a customizable PRI skeleton for researchers to train models on custom datasets with custom evaluation methods and flip rate parameters is under development and will be released separately.

---

## Optional: RunPod Setup Guide

If you use RunPod for GPU instances, follow these steps to deploy the pre-configured Docker image and set up SSH for file transfer between H100 and B200 machines.

### Step 0: Deploy RunPod Instance with Docker Image

1. Go to https://console.runpod.io/deploy
2. Select your desired GPU (H100 or B200)
3. Under **Configure Deployment** section, click **Change Template**
4. In the search form ("What are you looking for?"), type: `h100-b200-driftbench-ae`
5. Select the template and scroll down
6. Click **Deploy On-Demand**
7. Wait for the pod to start and download the Docker image

### Step 1: Enable SSH on RunPod instance (after pod starts)

Open the Web Terminal from the RunPod interface and run:

```bash
# Setup SSH service
mkdir -p /tmp && chmod 1777 /tmp
ssh-keygen -A
service ssh start

# Create SSH directory
mkdir -p ~/.ssh && chmod 700 ~/.ssh

# Add your public SSH key (generate locally if needed)
# Replace with your actual public key
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Verify SSH is running
service ssh status
```

### Step 2: Configure SSH on your local machine (one-time setup)

Add to your `~/.ssh/config` (get IP and Port from RunPod connection info panel):

```
Host runpod-h100
    HostName <RUNPOD_H100_IP>
    User root
    Port <RUNPOD_H100_PORT>
    IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host runpod-b200
    HostName <RUNPOD_B200_IP>
    User root
    Port <RUNPOD_B200_PORT>
    IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

**Between Step 2 and Step 3:** Run inference experiments on each machine following Path A instructions (see above). This will generate results in `results/Llama-3.1-8B/h100/` on the H100 instance and `results/Llama-3.1-8B/b200/` on the B200 instance.

### Step 3: Transfer results folder between instances

```bash
# Download H100 results to your local machine (choose any local folder name)
scp -r runpod-h100:/workspace/driftbench-ae/artifact_evaluation/\
reproduction_scripts/results/Llama-3.1-8B/h100 ./h100_results_temp

# Upload to B200 instance (place in correct path)
scp -r ./h100_results_temp/* runpod-b200:/workspace/driftbench-ae/\
artifact_evaluation/reproduction_scripts/results/Llama-3.1-8B/h100/
```

**Note:** The actual result folders on each instance are `h100/` and `b200/` (not `h100_results`). The command above downloads H100's results folder to a temporary local folder, then uploads it to the B200 instance so both machines have both sets of results before running `compute_direct_flip_rate.py`.

---

## License

- **Code:** MIT License
- **Data:** CC BY 4.0

---
