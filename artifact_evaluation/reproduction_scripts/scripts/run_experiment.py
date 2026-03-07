#!/usr/bin/env python3
"""
Main experiment runner for DriftBench.

Executes inference on specified model/hardware/precision/framework configuration
with full determinism and provenance tracking.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add local scripts to path
sys.path.insert(0, str(Path(__file__).parent))


def set_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent).decode().strip()
        return sha
    except:
        return "unknown"


def get_metadata(framework: str, hardware: str) -> dict:
    """
    Collect metadata for provenance tracking.
    
    Args:
        framework: Inference framework name
        hardware: Hardware identifier
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "git_sha": get_git_sha(),
        "docker_digest": os.environ.get("DOCKER_DIGEST", "unknown"),
        "framework_version": f"{framework}==unknown",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seeds": {"model": 42, "numpy": 42, "torch": 42, "python": 42},
        "hardware": {
            "gpu_model": "unknown",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }
    
    # Get framework version
    try:
        if framework == "vllm":
            import vllm
            metadata["framework_version"] = f"vllm=={vllm.__version__}"
        elif framework == "sglang":
            import sglang
            metadata["framework_version"] = f"sglang=={sglang.__version__}"
    except:
        pass
    
    # Get GPU info
    if torch.cuda.is_available():
        metadata["hardware"]["gpu_model"] = torch.cuda.get_device_name(0)
        metadata["hardware"]["driver_version"] = torch.version.cuda
    
    return metadata


def run_inference_vllm(model_path: str, prompts: list[str], config: dict, seed: int = 42, 
                      temperature: float = 0.0, num_samples: int = 1) -> list[str]:
    """
    Run inference using vLLM.
    
    Args:
        model_path: Path to model
        prompts: List of prompts
        config: Configuration dict (precision, hardware, etc.)
        seed: Random seed
        temperature: Sampling temperature (0.0 = deterministic, >0 = stochastic)
        num_samples: Number of samples per prompt (for stochastic sampling)
        
    Returns:
        List of generated texts (or list of lists if num_samples > 1)
    """
    from vllm import LLM, SamplingParams
    
    set_seeds(seed)
    
    # Get hardware-specific config (optional - use defaults if vllm_config not available)
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
        from vllm_config import get_vllm_config
        vllm_config = get_vllm_config(config["hardware"], config["precision"])
    except (ImportError, FileNotFoundError):
        # Fallback to sensible defaults for production validation
        vllm_config = {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
        }
    
    # Determine tensor parallel size based on model size and hardware
    # 70B models typically need 2 GPUs on H100 (80GB each)
    # B200 has 183GB and can fit 70B FP16 on single GPU
    import os
    tensor_parallel_size = 1
    if "70b" in model_path.lower() or "70B" in model_path:
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpu_count = len(cuda_devices.split(",")) if cuda_devices else torch.cuda.device_count()
        
        # B200 has enough memory for 70B single GPU
        if config["hardware"] == "b200":
            tensor_parallel_size = 1
            print(f"Using single GPU for 70B model on B200 (183GB memory)")
        elif visible_gpu_count >= 2:
            tensor_parallel_size = 2
            print(f"Using tensor parallelism (TP={tensor_parallel_size}) for 70B model")
        else:
            raise RuntimeError(f"70B model requires 2 GPUs on {config['hardware']} but only {visible_gpu_count} visible")
    
    # Increase max_model_len for long_context workloads
    if config.get("workload") == "long_context":
        vllm_config["max_model_len"] = 32768  # 32K for long context
        print(f"Increased max_model_len to {vllm_config['max_model_len']} for long_context workload")
    
    print(f"Initializing vLLM with config: {vllm_config}, TP={tensor_parallel_size}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16" if config["precision"] == "fp16" else "auto",
        quantization=vllm_config.get("quantization"),
        kv_cache_dtype=vllm_config.get("kv_cache_dtype"),
        max_model_len=vllm_config.get("max_model_len", 8192),
        gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.90),
        seed=seed,
        enforce_eager=True,  # Disable CUDA graphs for determinism
        disable_log_stats=True,
    )
    
    # Configure sampling parameters (supports both deterministic and stochastic)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9 if temperature > 0 else 1.0,  # Use top_p for stochastic sampling
        max_tokens=512,
        n=num_samples,  # Number of samples per prompt
        seed=seed,
    )
    
    if temperature > 0:
        print(f"Running inference on {len(prompts)} prompts (temperature={temperature}, n={num_samples} samples)...")
    else:
        print(f"Running inference on {len(prompts)} prompts (deterministic)...")
    
    # Setup event loop for SGLang
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Return format depends on num_samples
    if num_samples == 1:
        # Backward compatible: single sample per prompt
        return [out.outputs[0].text for out in outputs]
    else:
        # Multiple samples: return list of lists
        return [[o.text for o in out.outputs] for out in outputs]


def run_inference_tensorrt_llm(model_path: str, prompts: list[str], config: dict, seed: int = 42) -> list[str]:
    """
    Run inference using TensorRT-LLM (via virtual environment).
    
    Note: TensorRT-LLM requires significant initialization time for model compilation.
    We use a subprocess with longer timeout and explicit MPI settings.
    
    Args:
        model_path: Path to model
        prompts: List of prompts
        config: Configuration dict
        seed: Random seed
        
    Returns:
        List of generated texts
    """
    import subprocess
    import json
    import tempfile
    
    precision = config["precision"]
    
    # Get hardware type and workload for workload-specific settings
    hardware = config.get("hardware", "h100")
    workload = config.get("workload", "unknown")
    
    # Determine tensor parallel size for TensorRT-LLM
    import os
    # Use config value if provided, otherwise auto-detect
    tensor_parallel_size = config.get('tensor_parallel_size', None)
    if tensor_parallel_size is None:
        tensor_parallel_size = 1
        model_lower = model_path.lower()
        
        # Enable TP=2 for 70B models or mixtral on H100
        needs_tp2 = "70b" in model_lower or ("mixtral" in model_lower and hardware == "h100")
        
        if needs_tp2:
            # 70B models and Mixtral on H100 need 2 GPUs
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            visible_gpu_count = len(cuda_devices.split(",")) if cuda_devices else 1
            if visible_gpu_count >= 2:
                tensor_parallel_size = 2
    
    # Create script to run in TensorRT-LLM environment
    script = f'''
import json
import sys
import os
import time

# CRITICAL: Set memory management BEFORE imports (FIX for OOM errors)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Hardware-specific environment variables
hardware = "{hardware}"

if hardware == "h200":
    # H200-specific environment variables (from H200 guide)
    # Fixes UCX/mpool warnings and multi-GPU issues on Hopper architecture
    os.environ['NCCL_P2P_LEVEL'] = 'SYS'
    os.environ['UCX_LOG_LEVEL'] = 'WARN'
    
    # Additional UCX settings for Hopper multi-GPU stability
    # Prevents memory pool issues on H200 with TensorRT-LLM
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'  # Disable memory type cache
    os.environ['UCX_TLS'] = 'tcp,cuda_copy,cuda_ipc'  # Explicit transport selection
    
    # Memory management for H200 (prevents OOM on large models)
    os.environ['NIM_LOW_MEMORY_MODE'] = '1'

# MPI settings for multi-GPU tensor parallelism
os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

# Ensure ninja is in PATH for FlashInfer JIT compilation
os.environ['PATH'] = '/workspace/trt_env/bin:' + os.environ.get('PATH', '')

# Enable engine caching to speed up subsequent runs
os.environ['TLLM_LOG_LEVEL'] = 'INFO'
os.environ['TLLM_ENGINE_CACHE'] = str(Path(__file__).parent.parent / 'trt_engines')

def main():
    from tensorrt_llm import LLM, SamplingParams
    import torch
    import random
    import numpy as np

    # Set seeds
    torch.manual_seed({seed})
    random.seed({seed})
    np.random.seed({seed})

    # Create LLM with tensor parallelism
    print("Initializing TensorRT-LLM (TP={tensor_parallel_size})...", file=sys.stderr)
    start = time.time()

    # TensorRT-LLM configuration with workload-specific settings
    workload = "{workload}"
    
    llm_kwargs = {{
        "model": "{model_path}",
        "dtype": "{precision}",
        "tensor_parallel_size": {tensor_parallel_size},
        "trust_remote_code": True,
    }}
    
    # Configure max_num_tokens for long_context workloads
    # Long context prompts are ~8500 tokens, need 10K+ capacity
    if workload == "long_context":
        llm_kwargs["max_num_tokens"] = 32768  # 32K to handle 8.5K prompts + 512 generation
        print("Long context: max_num_tokens=32768 (for 8.5K input + generation)", file=sys.stderr)
    else:
        # FP8 Mixtral needs reduced max_num_tokens to prevent OOM (KV cache memory)
        is_fp8 = "{precision}" == "fp8"
        is_mixtral = "mixtral" in "{model_path}".lower()
        
        if is_fp8 and is_mixtral:
            llm_kwargs["max_num_tokens"] = 2048  # Reduced for FP8 Mixtral to prevent OOM
            print("Standard workload (FP8 Mixtral): max_num_tokens=2048 (optimized)", file=sys.stderr)
        else:
            llm_kwargs["max_num_tokens"] = 8192  # Default for other workloads
            print("Standard workload: max_num_tokens=8192", file=sys.stderr)
    
    if "{hardware}" == "h200":
        print("H200 detected - using environment variables for stability (NCCL_P2P_LEVEL, UCX)", file=sys.stderr)
    
    llm = LLM(**llm_kwargs)

    print(f"Initialization took {{time.time() - start:.1f}}s", file=sys.stderr)

    # Create sampling params
    # For long_context workloads, allow longer generation (but prompts are long, not generation)
    max_gen_tokens = 512  # default output length
    if workload == "long_context":
        # Long context has long INPUT, but output can stay at 512
        # The 32K limit is for input+output combined context window
        print("Long context: using 512 token generation (32K input capacity)", file=sys.stderr)
    
    sampling_params = SamplingParams(
        max_tokens=max_gen_tokens,
        temperature=0.0,
        seed={seed}
    )

    # Generate (prompts embedded as Python list literal)
    prompts_list = {repr(prompts)}
    print(f"Generating for {{len(prompts_list)}} prompts...", file=sys.stderr)
    outputs = llm.generate(prompts_list, sampling_params=sampling_params)
    # Extract text from TensorRT-LLM RequestOutput objects
    results = [out.outputs[0].text if out.outputs else "" for out in outputs]
    # Print with marker to distinguish from logs
    print("__TRTLLM_OUTPUT_START__", file=sys.stderr)
    print(json.dumps(results))
    print("__TRTLLM_OUTPUT_END__", file=sys.stderr)

if __name__ == '__main__':
    main()
'''
    
    # Save script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    # Execute in TensorRT-LLM environment with longer timeout
    # TensorRT-LLM needs time to compile the model on first run
    # Set LD_LIBRARY_PATH to include system libraries for MPI and OpenMPI
    import os
    env = os.environ.copy()
    # Add system OpenMPI libraries (critical for multi-GPU on H200)
    mpi_paths = '/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib'
    env['LD_LIBRARY_PATH'] = f'/workspace/trt_env/lib:{mpi_paths}:' + env.get('LD_LIBRARY_PATH', '')
    
    # Timeout: 60 minutes for first-time compilation of large models
    # Engine compilation can take 35-45 minutes for FP8 models with TP=2
    # Subsequent runs will be faster (engines cached)
    timeout_seconds = 3600  # 60 minutes
    
    # Use mpirun explicitly when TP > 1 to avoid MPI spawning issues
    if tensor_parallel_size > 1:
        cmd = [
            'mpirun',
            '-n', str(tensor_parallel_size),
            '--allow-run-as-root',
            '--oversubscribe',
            '/workspace/trt_env/bin/python',
            script_path
        ]
    else:
        cmd = ['/workspace/trt_env/bin/python', script_path]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env
    )
    
    # Clean up
    os.unlink(script_path)
    
    if result.returncode != 0:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"TensorRT-LLM FAILURE", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"Hardware: {hardware}", file=sys.stderr)
        print(f"Model: {model_path}", file=sys.stderr)
        print(f"Tensor Parallel Size: {tensor_parallel_size}", file=sys.stderr)
        print(f"Precision: {precision}", file=sys.stderr)
        print(f"\nFull stderr output:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        
        # Check for specific error patterns (especially for H200)
        if "SIGSEGV" in result.stderr or "Segmentation fault" in result.stderr:
            if hardware == "h200":
                error_msg = (
                    f"TensorRT-LLM segmentation fault on H200. "
                    f"Model: {model_path.split('/')[-1]}, TP={tensor_parallel_size}. "
                    f"H200 fixes applied: NCCL_P2P_LEVEL=SYS, NIM_LOW_MEMORY_MODE=1. "
                    f"Error: {result.stderr[-500:]}"
                )
            else:
                error_msg = f"TensorRT-LLM segmentation fault: {result.stderr[-500:]}"
        else:
            error_msg = f"TensorRT-LLM inference failed with code {result.returncode}: {result.stderr[-1000:]}"
        
        raise RuntimeError(error_msg)
    
    # Extract JSON from stdout between markers
    stdout_lines = result.stdout.strip().split('\n')
    stderr_lines = result.stderr.strip().split('\n')
    
    # Try to find output between markers in stderr
    in_output = False
    json_lines = []
    for line in stderr_lines:
        if "__TRTLLM_OUTPUT_END__" in line:
            in_output = False
        if in_output:
            json_lines.append(line)
        if "__TRTLLM_OUTPUT_START__" in line:
            in_output = True
    
    # If markers not found, look for valid JSON in stdout
    if not json_lines:
        for line in stdout_lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    json.loads(line)  # Validate it's actual JSON
                    json_lines = [line]
                    break
                except json.JSONDecodeError:
                    continue
    
    if json_lines:
        json_output = ''.join(json_lines)
        try:
            return json.loads(json_output)
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse JSON: {json_output[:500]}", file=sys.stderr)
            print(f"DEBUG: Error: {e}", file=sys.stderr)
            raise
    
    print(f"DEBUG: No valid JSON found in output", file=sys.stderr)
    print(f"DEBUG: Stdout (first 1000 chars):\n{result.stdout[:1000]}", file=sys.stderr)
    print(f"DEBUG: Stderr (first 1000 chars):\n{result.stderr[:1000]}", file=sys.stderr)
    raise RuntimeError("No JSON output found from TensorRT-LLM")


def run_inference_sglang(model_path: str, prompts: list[str], config: dict, seed: int = 42) -> list[str]:
    """
    Run inference using SGLang offline Engine API with B200 Blackwell support.
    
    Implements proper B200 configuration from SGLang 0.5.3 documentation:
    - Uses sgl.Engine() for offline batch inference
    - Sets TORCH_CUDA_ARCH_LIST="10.0" for SM_100 compilation
    - Configures attention backends and quantization
    - Handles tensor parallelism for 70B models
    
    Args:
        model_path: Path to model
        prompts: List of prompts
        config: Configuration dict (hardware, precision, workload)
        seed: Random seed
        
    Returns:
        List of generated texts
    """
    import sglang as sgl


    
    set_seeds(seed)
    
    # B200 requires SM_100 architecture flag
    if config["hardware"] == "b200":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
        print("✓ Set TORCH_CUDA_ARCH_LIST=10.0 for B200 Blackwell support")
    
    # Determine tensor parallel size
    tensor_parallel_size = 1
    if "70b" in model_path.lower():
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpu_count = len(cuda_devices.split(",")) if cuda_devices else torch.cuda.device_count()
        
        if config["hardware"] == "b200":
            tensor_parallel_size = 1  # B200 has 192GB, can fit 70B
            print(f"Using single GPU for 70B model on B200 (192GB memory)")
        elif visible_gpu_count >= 2:
            tensor_parallel_size = 2
            print(f"Using tensor parallelism (TP={tensor_parallel_size}) for 70B model")
        else:
            raise RuntimeError(f"70B model requires 2 GPUs on {config['hardware']} but only {visible_gpu_count} visible")
    
    # Configure precision and quantization
    dtype = None
    quantization = None
    
    if config["precision"] == "fp16":
        dtype = "float16"
    elif config["precision"] == "bf16":
        dtype = "bfloat16"
    elif config["precision"] == "fp8":
        quantization = "fp8"
    elif config["precision"] == "fp4":
        # B200 NVFP4 support
        if config["hardware"] == "b200":
            quantization = "modelopt_fp4"
            print("✓ Using NVFP4 quantization for B200")
        else:
            raise ValueError(f"FP4 precision only supported on B200, not {config['hardware']}")
    
    # Select attention backend based on hardware
    # Per guide: TRTLLM MLA is Blackwell-optimized for decode operations
    attention_backend = "flashinfer"  # default
    if config["hardware"] == "b200" and config.get("workload") != "long_context":
        # Use TRTLLM MLA for B200 decode optimization (prefill falls back to FlashInfer)
        attention_backend = "trtllm_mla"
        print("✓ Using TRTLLM MLA attention backend for B200 decode optimization")
    
    # Configure max context length for workload
    max_model_len = 8192  # default
    if config.get("workload") == "long_context":
        max_model_len = 32768
        print(f"✓ Increased max_model_len to {max_model_len} for long_context workload")
    
    print(f"Initializing SGLang Engine with config:")
    print(f"  - TP: {tensor_parallel_size}")
    print(f"  - dtype: {dtype}")
    print(f"  - quantization: {quantization}")
    print(f"  - attention_backend: {attention_backend}")
    print(f"  - max_model_len: {max_model_len}")
    
    # Initialize SGLang Engine (offline batch API)
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=tensor_parallel_size,
        quantization=quantization,
        mem_fraction_static=0.85,
        max_running_requests=200,
        random_seed=seed,
        trust_remote_code=True,
        disable_cuda_graph=True,  # For determinism
    )
    
    # Sampling parameters (dict format for SGLang)
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 512,
    }
    
    print(f"Running inference on {len(prompts)} prompts...")
    # Setup event loop for SGLang
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract text from SGLang output format
    return [out["text"] for out in outputs]


def main():
    parser = argparse.ArgumentParser(description="Run DriftBench experiment")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--hardware", required=True, choices=["h100", "h200", "b200", "mi300x"])
    parser.add_argument("--precision", required=True, choices=["fp16", "bf16", "fp8", "fp4"])
    parser.add_argument("--framework", required=True, choices=["vllm", "tensorrt-llm", "sglang"])
    parser.add_argument("--prompt-file", required=True, help="Path to JSONL prompt file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--replicate-id", type=int, default=1, help="Replicate ID")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts (for testing)")
    
    # NEW: Revision experiment parameters (backward compatible - defaults maintain old behavior)
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Sampling temperature (0.0=deterministic, 0.7=stochastic for revision experiments)")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples per prompt (for stochastic sampling validation)")
    
    # NEW: Root cause analysis flag (FIX #11)
    parser.add_argument("--enable-root-cause-analysis", action="store_true",
                       help="Enable root cause mechanism analysis (saves additional diagnostics)")
    
    args = parser.parse_args()
    
    # Set deterministic environment
    set_seeds(args.seed)
    
    print(f"=== DriftBench Experiment ===")
    print(f"Model: {args.model}")
    print(f"Hardware: {args.hardware}")
    print(f"Precision: {args.precision}")
    print(f"Framework: {args.framework}")
    print(f"Replicate: {args.replicate_id}")
    print(f"Seed: {args.seed}")
    
    # NEW: Show stochastic sampling parameters if used
    if args.temperature > 0:
        print(f"⚠️  STOCHASTIC MODE: temperature={args.temperature}, num_samples={args.num_samples}")
        print(f"   (For MLSys revision FIX #5: Temperature>0 validation)")
    
    # Load prompts
    print(f"\nLoading prompts from {args.prompt_file}...")
    with open(args.prompt_file) as f:
        prompts_data = [json.loads(line) for line in f]
    
    if args.max_prompts:
        prompts_data = prompts_data[:args.max_prompts]
    
    print(f"Loaded {len(prompts_data)} prompts")
    
    prompts = [p["prompt"] for p in prompts_data]
    
    # Detect workload from first prompt
    workload = prompts_data[0].get("workload")
    if not workload or workload == "unknown":
        prompt_id = prompts_data[0]["prompt_id"]
        if "humaneval" in prompt_id:
            workload = "code"
        elif "gsm8k" in prompt_id:
            workload = "math"
        elif "advbench" in prompt_id:
            workload = "safety"
        elif "lmsys" in prompt_id:
            workload = "chat"
        elif "long" in prompt_id:
            workload = "long_context"
        else:
            workload = "unknown"
    
    # Get metadata
    config = {
        "model": os.path.basename(args.model),
        "hardware": args.hardware,
        "precision": args.precision,
        "framework": args.framework,
        "workload": workload,
    }
    
    metadata = get_metadata(args.framework, args.hardware)
    metadata["config"] = config
    
    # Run inference
    print(f"\nRunning inference with {args.framework} for workload: {workload}...")
    
    if args.framework == "vllm":
        outputs = run_inference_vllm(args.model, prompts, config, args.seed, 
                                     temperature=args.temperature, num_samples=args.num_samples)
    elif args.framework == "tensorrt-llm":
        outputs = run_inference_tensorrt_llm(args.model, prompts, config, args.seed)
    elif args.framework == "sglang":
        outputs = run_inference_sglang(args.model, prompts, config, args.seed)
    else:
        raise NotImplementedError(f"Framework {args.framework} not yet implemented")
    
    # Save outputs
    print(f"\nSaving outputs to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add sampling parameters to metadata for revision experiments
    if args.temperature > 0 or args.num_samples > 1:
        metadata["sampling"] = {
            "temperature": args.temperature,
            "num_samples": args.num_samples,
            "top_p": 0.9 if args.temperature > 0 else 1.0,
            "revision_experiment": "temperature_validation"  # Mark as revision experiment
        }
    
    # Add root cause analysis flag to metadata (FIX #11)
    if args.enable_root_cause_analysis:
        metadata["root_cause_analysis"] = {
            "enabled": True,
            "baseline_for_comparison": args.precision == "fp16",  # FP16 is typical baseline
            "diagnostic_mode": True
        }
    
    for prompt_data, output in zip(prompts_data, outputs):
        prompt_hash = prompt_data["prompt_hash"]
        
        # Detect workload from prompt_id if not provided
        workload = prompt_data.get("workload")
        if not workload or workload == "unknown":
            prompt_id = prompt_data["prompt_id"]
            if "humaneval" in prompt_id:
                workload = "code"
            elif "gsm8k" in prompt_id:
                workload = "math"
            elif "advbench" in prompt_id:
                workload = "safety"
            elif "lmsys" in prompt_id:
                workload = "chat"
            elif "long" in prompt_id:
                workload = "long_context"
            else:
                workload = "unknown"
        
        # Construct filename following naming convention
        filename = f"{config['model']}--{args.hardware}--{args.precision}--{args.framework}--{workload}--{prompt_hash}_{args.replicate_id:02d}.json"
        
        # Add workload to config for this specific output
        output_config = config.copy()
        output_config["workload"] = workload
        
        # Handle both single and multiple samples
        if isinstance(output, list):
            # Multiple samples (stochastic sampling)
            output_data = {
                "prompt_id": prompt_data["prompt_id"],
                "replicate_id": args.replicate_id,
                "prompt": prompt_data["prompt"],
                "generated_samples": output,  # List of samples
                "num_samples": len(output),
                "config": output_config,
                "metadata": metadata,
            }
        else:
            # Single sample (backward compatible)
            output_data = {
                "prompt_id": prompt_data["prompt_id"],
                "replicate_id": args.replicate_id,
                "prompt": prompt_data["prompt"],
                "generated_text": output,
                "config": output_config,
                "metadata": metadata,
            }
        
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)
        
        # Update MANIFEST (disabled - parquet files not needed for artifact evaluation)
        # try:
        #     update_manifest(filepath, metadata)
        # except Exception as e:
        #     print(f"Warning: Could not update MANIFEST: {e}")
    
    print(f"✓ Completed {len(outputs)} inferences")
    print(f"✓ Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
