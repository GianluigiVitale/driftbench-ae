# DriftBench Datasets

This directory contains the original prompt datasets used in the DriftBench experiments.

## Dataset Files

### 1. humaneval_prompts.jsonl (Code Generation)
- **Source:** HumanEval benchmark
- **Size:** 164 prompts
- **Task:** Python function implementation from docstrings
- **Evaluation:** Pass@1 (execution against test cases)
- **Reference:** https://huggingface.co/datasets/openai/openai_humaneval

### 2. gsm8k_prompts.jsonl (Math Reasoning)
- **Source:** GSM8K (Grade School Math 8K)
- **Size:** 500 prompts
- **Task:** Multi-step arithmetic word problems
- **Evaluation:** Exact match on numerical answers
- **Reference:** https://huggingface.co/datasets/openai/gsm8k

### 3. advbench_prompts.jsonl (Safety Evaluation)
- **Source:** AdvBench harmful behaviors dataset
- **Size:** 520 prompts
- **Task:** Safety boundary testing
- **Evaluation:** LlamaGuard-3-8B binary classification (safe/unsafe)
- **Reference:** https://huggingface.co/datasets/walledai/AdvBench

### 4. lmsys_prompts.jsonl (Chat/Conversation)
- **Source:** LMSYS-Chat-1M dataset
- **Size:** 1000 prompts (973 used in paper)
- **Task:** Real-world user conversations
- **Evaluation:** Semantic similarity (cosine distance of embeddings)
- **Reference:** https://huggingface.co/datasets/lmsys/lmsys-chat-1m

### 5. long_qa_prompts.jsonl (Long-Context QA)
- **Source:** LongBench-qasper (academic paper QA)
- **Size:** 100 documents
- **Task:** Question answering over long documents
- **Evaluation:** F1 score on extracted answers
- **Reference:** https://huggingface.co/datasets/THUDM/LongBench

## File Format

All files are in JSON Lines format (one JSON object per line):

```json
{
  "prompt_id": "unique_identifier",
  "prompt": "the input text",
  "ground_truth": "expected output or answer",
  "metadata": {}
}
```

## Total Coverage

- **Total prompts:** 2,284 (across 5 workloads)
- **Paper experiments:** 2,257 prompts
  - Code: 164
  - Math: 500
  - Safety: 520
  - Chat: 973 (subset of 1000)
  - Long-Context: 100

## Usage

These datasets were used as inputs to all 105 configurations in the DriftBench experiments:
- 5 models × 4 hardware platforms × 3 precision formats × 3 frameworks
- Each prompt evaluated across multiple infrastructure configurations
- Total: 236,985 prompt-response pairs (2,257 prompts × 105 configurations)

## Verification

To verify dataset integrity:

```bash
# Count prompts
wc -l *.jsonl

# View sample prompt
head -1 humaneval_prompts.jsonl | python3 -m json.tool
```

## Paper Reference

These datasets are documented in:
- **Main paper:** Section 2 (Methodology) → Workload Selection
- **Appendix:** Dataset Specifications (Appendix → Datasets)

All datasets are publicly available and properly cited in the paper.

Last updated: October 30, 2025
