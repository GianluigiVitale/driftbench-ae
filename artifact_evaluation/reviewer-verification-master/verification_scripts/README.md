# Verification Scripts

This directory contains verification scripts and datasets for validating paper claims.

## Available Scripts

1. **verification_master_script.py**
   - Verification of all paper metrics
   - Validates R² values, flip rates, and generalization results
   - Cross-references claims with experimental data

2. **validate_semantic_drift_human_annotations.py**
   - Validates automated semantic drift measurement against human judgments
   - Analyzes 100 human categorical assessments
   - Computes Spearman correlation (ρ=0.804, p<0.001)
   - Includes required data file

### Data Files

- **`semantic_drift_100_human_annotations.csv`** (366 KB)
  - 100 production prompts from LMSYS-Chat dataset
  - Comparison: H100/FP16/vLLM vs B200/FP8/TensorRT-LLM
  - Model: Llama-3.1-70B-Instruct
  - Human categorical judgments: identical, minor_diff, moderate_diff, major_diff, contradictory
  - Automated cosine similarity scores for validation

- **`safety_516_human_annotations.csv`** (1.3 MB)
  - 516 safety prompts from AdvBench dataset (4 uncertain excluded from 520)
  - Baseline responses from H100/FP16/SGLang configuration
  - Human safety labels for validation against LlamaGuard-3-8B automated classifier
  - Used to validate LlamaGuard Safety Recall: 85.21% (121/142 unsafe responses detected)
  - Columns: prompt_id, prompt, baseline_response, llamaguard_label, human_label, confidence, notes

## Usage

### Quick Start: Semantic Drift Validation

```bash
cd verification_scripts
python3 validate_semantic_drift_human_annotations.py
```

**Expected Output:**
```
Total Annotated Prompts: 100

Categorical Judgment Distribution:
  identical           :  25 ( 25.0%)
  minor_diff          :  29 ( 29.0%)
  moderate_diff       :  36 ( 36.0%)
  major_diff          :   5 (  5.0%)
  contradictory       :   5 (  5.0%)

Mean Automated Score by Category:
  identical           : 0.939
  minor_diff          : 0.897
  moderate_diff       : 0.709
  major_diff          : 0.524
  contradictory       : 0.108

Spearman ρ: 0.804
P-value: < 0.001 (highly significant)
```

### Dependencies

```bash
pip install scipy pandas numpy
```

## What Gets Validated

### Semantic Drift Human Validation

The script validates the automated semantic drift measurement methodology reported in:
- **Paper Section**: Appendix → Semantic Drift Human Validation
- **Table**: tab:semantic_validation_app

Key metrics verified:
1. Category distribution (25% identical, 29% minor, 36% moderate, 5% major, 5% contradictory)
2. Mean automated scores decrease monotonically with judgment severity
3. Spearman correlation ρ=0.804 (p<0.001) indicating strong monotonic relationship
4. Distribution summary: 54% minor-to-no difference, 36% moderate, 10% major/contradictory

The monotonic relationship shows that as human annotators judge response pairs to be more different (identical → contradictory), the automated cosine similarity scores decrease correspondingly.

## Dataset Details

### Human Annotation Methodology

**Setup:**
- **Baseline Configuration**: H100 GPU, FP16 precision, vLLM framework
- **Test Configuration**: B200 GPU, FP8 precision, TensorRT-LLM framework
- **Model**: Llama-3.1-70B-Instruct (frozen weights)
- **Prompts**: 100 randomly sampled from LMSYS-Chat-1M (production workload)

**Annotation Process:**
Each response pair received a categorical judgment:
- **Identical**: No meaningful difference in content or meaning
- **Minor Difference**: Slight wording changes, same core information
- **Moderate Difference**: Same information but different style/emphasis
- **Major Difference**: Substantial content differences
- **Contradictory**: Conflicting or incompatible information

**Automated Measurement:**
- Cosine similarity of sentence embeddings (all-mpnet-base-v2)
- Range: 0.0 (completely different) to 1.0 (identical)
- Computed for all 236,985 prompt-response pairs in full benchmark

### CSV File Format

**Columns:**
- `prompt_id`: Unique identifier for the prompt
- `prompt`: The input prompt text
- `response_baseline`: Model response from baseline config (H100/FP16/vLLM)
- `response_target`: Model response from test config (B200/FP8/TensorRT)
- `cos_sim`: Automated cosine similarity score (0.0-1.0)
- `sdr`: Semantic drift rate (1 - cos_sim)
- `length_ratio`: Length ratio between responses
- `manual_similarity_score`: Derived score (not directly annotated)
- `manual_judgment`: Human categorical judgment (5 categories)
- `notes`: Annotator notes explaining the judgment

## Paper References

**Main Text:**
- Section 2.3: Methodology Validation
  - "Human validation on 100 production prompts shows strong monotonic agreement between categorical judgments and automated scores (Spearman ρ=0.804, p<0.001)."

**Contributions:**
- "(5) Artifacts: ... 100 semantic drift categorical annotations validated with Spearman ρ=0.804"

**Appendix:**
- Subsection: Semantic Drift Human Validation (label: app:validation:semantic)
- Table: tab:semantic_validation_app
- Full methodology, results, and interpretation

## Contact

For questions about verification or data:
- See main repository README
- Check paper appendices for detailed methodology
- Full dataset information available in artifact appendix PDF

## Human Annotation Datasets

This directory includes two human annotation datasets:

1. **Semantic Drift Validation** (`semantic_drift_100_human_annotations.csv`)
   - 100 prompts with human categorical judgments
   - Validates automated semantic drift measurement
   - Spearman ρ=0.804 correlation with automated scores

2. **Safety Classification Validation** (`safety_516_human_annotations.csv`)
   - 516 safety prompts with human safety labels
   - Validates LlamaGuard-3-8B automated safety classifier
   - 85.21% safety recall (121/142 unsafe responses detected)
