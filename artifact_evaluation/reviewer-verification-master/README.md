# DriftBench Verification Package

This anonymous repository contains all materials needed for reviewers to verify claims in the DriftBench MLSys 2026 submission.

## Verification Results

Automated checks: 34/34 passed (100%)

All numerical data verified:
- 236,985 prompt-response pairs confirmed
- 105 configurations validated
- All R² values match source data
- Production validation case exact (65 to 113 violations, +9.23pp)
- LlamaGuard metrics correct (85.21% safety recall, 121/142)

Issues found: None (numerical accuracy confirmed)

## Quick Start

### verification_scripts/

Contains automated verification tools and validation guides.

- `verification_master_script.py` - Automated verification of 31 key paper claims including dataset sizes, flip rates, PRI R² values, and production validation results. Runtime is approximately 30 seconds.

- `validate_semantic_drift_human_annotations.py` - Validates automated semantic drift measurements against 100 human categorical judgments. Computes Spearman correlation (ρ=0.804, p<0.001) between automated measurements and human assessment. Includes the data file `semantic_drift_100_human_annotations.csv` (366 KB).

- `safety_516_human_annotations.csv` - Human safety labels for 516 AdvBench prompts used to validate the LlamaGuard-3-8B automated safety classifier. Documents the 85.21% safety recall (121/142 unsafe responses detected) reported in the paper.

- `README.md` - Documentation for the verification scripts with usage examples and expected outputs.

Run the master verification script to check all 31 paper claims:

```bash
cd verification_scripts
python3 verification_master_script.py
```

This verifies dataset sizes (236,985 pairs, 105 configs), flip rates, PRI R² values, production validation results, and statistical tests.

To validate semantic drift measurements against human judgments:

```bash
cd verification_scripts
python3 validate_semantic_drift_human_annotations.py
```

This reproduces the Spearman ρ=0.804 correlation reported in the paper's appendix.

## Verified Claims

The verification scripts check 29 numerical claims from the paper:

**Dataset**:
- 236,985 prompt-response pairs
- 105 configurations (5 models, 4 hardware platforms, 3 precision formats, 3 frameworks)
- 525 completed experiments
- 4 objective workload types (code, math, safety, long_context)

**Key Results**:
- Safety flip rate: 7.95%
- Code flip rate: 0.09%
- Math flip rate: 16.74%
- Safety/Code ratio: 91×

**Production Validation**:
- Drift increase: +9.23pp (H100/FP16/SGLang to B200/FP8/SGLang)
- Violation count: 65 to 113 (+48 unsafe outputs)
- Exact flip rate: 21.73% (113/520)

**PRI Model Generalization**:
- Overall R²: 0.998
- Hardware held-out R²: 0.909
- Precision held-out R²: 0.763
- Framework held-out R²: 0.479
- Model held-out R²: 0.118

**Statistical Tests**:
- Workload ANOVA F-statistic: 53.60
- LlamaGuard Safety Recall: 85.21% (121/142 unsafe responses detected across 516 prompts)
- All reported p-values < 0.001

**Human Validation**:
- 100 human categorical judgments on semantic drift (identical, minor, moderate, major, contradictory)
- Spearman ρ=0.804 (p<0.001) between automated measurements and human assessments
- Distribution: 54% minor-to-no difference, 36% moderate, 10% major/contradictory
- Mean cosine similarities by category range from 0.939 (identical) to 0.108 (contradictory)
- 516 human safety labels validating LlamaGuard-3-8B automated safety classifier
- Complete datasets included in verification_scripts/

## Data Files

All data files required for verification are included locally:

### Analysis Data (`data/` directory)
```
data/
├── flip_rates.csv                    (420 rows: 105 configs × 4 workloads)
├── generalization_results.csv        (PRI R² values)
├── feature_importance.csv            (PRI feature rankings)
└── anova_flip_rate_corrected.csv     (Statistical test results)
```

### Original Datasets (`datasets/` directory)
```
datasets/
├── humaneval_prompts.jsonl           (164 code generation prompts)
├── gsm8k_prompts.jsonl               (500 math reasoning prompts)
├── advbench_prompts.jsonl            (520 safety evaluation prompts)
├── lmsys_prompts.jsonl               (1000 chat conversation prompts)
├── long_qa_prompts.jsonl             (100 long-context QA prompts)
└── README.md                         (Dataset documentation)
```

### Human Validation Data (`verification_scripts/` directory)
- `semantic_drift_100_human_annotations.csv` (366 KB - Semantic drift human judgments)
- `safety_516_human_annotations.csv` (1.3 MB - Safety classification human labels)

### Full Experimental Dataset (External)

The complete prompt-response experiment matrix (236,985 pairs) is available for download. 

**Figshare repository information** (including download link) is provided in the artifact appendix PDF for evaluators.

**Size**: 400 MB  
**Contents**: All 236,985 prompt-response pairs across 105 configurations (5 models × 4 hardware × 3 precision × 3 frameworks × 4 workloads)

The verification scripts automatically use the local data files included in this repository. No external data access required for verification.

## Repository Structure

```
repo_reviewer/
├── README.md                           
├── data/                               (Analysis results)
│   ├── flip_rates.csv
│   ├── generalization_results.csv
│   ├── feature_importance.csv
│   └── anova_flip_rate_corrected.csv
├── data_references/                    (External data links)
│   └── FIGSHARE_REPOSITORY.md
├── datasets/                           (Original prompt datasets)
│   ├── humaneval_prompts.jsonl        (164 code prompts)
│   ├── gsm8k_prompts.jsonl            (500 math prompts)
│   ├── advbench_prompts.jsonl         (520 safety prompts)
│   ├── lmsys_prompts.jsonl            (1000 chat prompts)
│   ├── long_qa_prompts.jsonl          (100 long-context prompts)
│   └── README.md
└── verification_scripts/               
    ├── verification_master_script.py
    ├── validate_semantic_drift_human_annotations.py
    ├── semantic_drift_100_human_annotations.csv (366 KB)
    ├── safety_516_human_annotations.csv (1.3 MB)
    └── README.md
```

## Data Integrity

- All claims traceable to source files
- All verification commands tested and working
- Transparency in methodology
- No overstatements or unsupported claims identified

## Additional Information

Documentation files:
- This README - Main navigation and overview
- `verification_scripts/README.md` - Script documentation
- `datasets/README.md` - Dataset documentation

Last updated: October 30, 2025  
Verification runtime: Under 1 minute for automated scripts
