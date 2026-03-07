# Models Directory

This directory should contain the required models for the production validation.

## Required Models

### 1. Llama-3.1-8B-Instruct
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b
```

### 2. LlamaGuard-3-8B
```bash
huggingface-cli download meta-llama/LlamaGuard-3-8B --local-dir models/llama-guard-3-8b
```

## Note

These models require HuggingFace account access to Meta's gated models. 

Expected structure after download:
```
models/
├── llama-3.1-8b/
│   ├── config.json
│   ├── tokenizer.json
│   └── *.safetensors
└── llama-guard-3-8b/
    ├── config.json
    ├── tokenizer.json
    └── *.safetensors
```

Total size: ~35GB (16GB for Llama-3.1-8B + 19GB for LlamaGuard-3-8B)
