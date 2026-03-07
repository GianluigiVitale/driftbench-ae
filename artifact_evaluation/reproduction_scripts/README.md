# DriftBench Production Validation - Artifact Evaluation Package

Complete reproduction package for the DriftBench production validation experiment (Section 3.3 of the paper).

**Experiment:** H100/FP16/SGLang → B200/FP8/SGLang safety drift detection using Llama-3.1-8B-Instruct and AdvBench.

**Status:** ✓ Tested and validated on RTX 5090

---

## Quick Start

```bash
# Step 1: Setup environment (10-15 minutes)
chmod +x setup_sglang.sh
./setup_sglang.sh

# Step 2: Download models (one-time, ~35GB)
# Option A: Use provided script (recommended)
chmod +x download_models.sh
./download_models.sh

# Option B: Manual download with huggingface-cli
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b
# huggingface-cli download meta-llama/LlamaGuard-3-8B --local-dir models/llama-guard-3-8b

# Step 3: Run H100/FP16/SGLang experiment
source /workspace/venv/bin/activate
python parallel_controller.py --hardware h100 --workers 1

# Step 4: Run B200/FP8/SGLang experiment
python parallel_controller.py --hardware b200 --workers 1

# Step 5: Compute production validation
# This automatically:
#   - Loads LlamaGuard-3-8B classifier
#   - Classifies all H100 outputs (520 prompts) as safe/unsafe
#   - Classifies all B200 outputs (520 prompts) as safe/unsafe
#   - Compares classifications to find flips
#   - Calculates flip rate and makes deployment decision
python compute_direct_flip_rate.py
```

**Expected result:** Flip rate showing how many prompts changed safety classification between H100 and B200 configurations

---

## Folder Structure

```
reproduction_scripts/
├── README.md                        # This file
├── PRODUCTION_VALIDATION.md         # Production validation guide (START HERE)
├── requirements.txt                 # Python dependencies
├── setup_sglang.sh                  # Environment setup script
├── experiment_matrix.csv            # Experiment definitions
│
├── parallel_controller.py           # Experiment runner
├── compute_direct_flip_rate.py      # Main validation script
│
├── models/                          # Model downloads (git-ignored)
│   ├── README.md                    # Download instructions
│   ├── llama-3.1-8b/                # Llama-3.1-8B-Instruct (~16GB)
│   └── llama-guard-3-8b/            # LlamaGuard-3-8B (~19GB)
│
├── data/
│   └── safety/
│       └── advbench_prompts.jsonl   # 520 safety prompts
│
├── scripts/
│   ├── run_experiment.py            # Core inference execution
│   ├── evaluate_safety.py           # LlamaGuard-3-8B classifier
│   └── manifest_utils.py            # Provenance tracking
│
└── results/                         # Experiment outputs (git-ignored)
    ├── llama-3.1-8b/h100/fp16/sglang/safety/  # H100 outputs (520 JSON)
    ├── llama-3.1-8b/b200/fp8/sglang/safety/   # B200 outputs (520 JSON)
    └── production_validation_results.json      # Final results
```

---

## Documentation

- **📖 [PRODUCTION_VALIDATION.md](PRODUCTION_VALIDATION.md)** - Production validation workflow (START HERE)

---

## Hardware Requirements

**For testing (FP16 only):**
- NVIDIA RTX 3090 / 4090 / 5090 (24GB+ VRAM)
- CUDA 12.8+
- 32GB RAM

**For production validation (H100→B200 comparison):**
- H100 GPU (80GB) - Baseline configuration
- B200 GPU (192GB) - Target configuration
- FP8 tensor core support required

**Note:** FP8 quantization is not supported on consumer GPUs (RTX series). Tested on RTX 5090 with FP16 only.

---

## Key Metrics (RTX 5090 Test Run)

**Inference Performance:**
- Model: Llama-3.1-8B-Instruct
- Time: 278 seconds for 520 prompts
- Speed: 1.87 prompts/second

**Safety Classification:**
- Total: 520 outputs classified
- Safe: 201 (38.7%)
- Unsafe: 319 (61.3%)
- LlamaGuard-3-8B processing: ~8 classifications/second

**Expected Production Result (H100→B200):**
- Flip rate: ~24% (significant safety drift detected)
- Decision: DEPLOYMENT BLOCKED ⚠️

---

## Files Summary

| Category | Files | Size |
|----------|-------|------|
| **Scripts** | 7 Python files | ~90 KB |
| **Config** | 4 files | ~12 KB |
| **Data** | 1 dataset | 140 KB |
| **Docs** | 4 markdown files | ~33 KB |
| **Results** | 521 JSON files | 2.2 MB |
| **Total** | 537 files | 2.5 MB |

---

## Pipeline Overview

```
Step 1: Setup Environment
   └─> setup_sglang.sh

Step 2: Run H100/FP16/SGLang Inference
   └─> parallel_controller.py --hardware h100
   └─> Generates: results/llama-3.1-8b/h100/fp16/sglang/safety/*.json (520 files)

Step 3: Run B200/FP8/SGLang Inference  
   └─> parallel_controller.py --hardware b200
   └─> Generates: results/llama-3.1-8b/b200/fp8/sglang/safety/*.json (520 files)

Step 4: Production Validation (Automatic)
   └─> compute_direct_flip_rate.py does:
       1. Load LlamaGuard-3-8B classifier
       2. Classify all 520 H100 outputs → safe/unsafe labels
       3. Classify all 520 B200 outputs → safe/unsafe labels
       4. Compare classifications between H100 and B200
       5. Count flips (safe↔unsafe changes)
       6. Calculate flip rate percentage
       7. Make deployment decision (BLOCK if >5% flip rate)
   └─> Generates: production_validation_results.json
```

---

## Troubleshooting

**FP8 fails on RTX 5090:**
- Expected - FP8 requires H100/H200/B200 datacenter GPUs
- Use FP16 for testing on consumer hardware

**LlamaGuard not found:**
- Download with: `huggingface-cli download meta-llama/LlamaGuard-3-8B`
- Requires HuggingFace account with Meta model access

**Import errors:**
- Run scripts from the reproduction_scripts/ root directory
- Scripts use relative imports and expect to be run from root

---

## Citation

If you use this reproduction package, please cite the DriftBench paper:

```bibtex
@inproceedings{driftbench2026,
  title={DriftBench: A Benchmark for Infrastructure Drift in LLM Inference},
  author={...},
  booktitle={MLSys 2026},
  year={2026}
}
```

---

## License

See main DriftBench repository for license information.

---

## Contact

For questions about this artifact evaluation package, please refer to the paper or contact the authors.

**MLSys 2026 Artifact Evaluation** ✓
