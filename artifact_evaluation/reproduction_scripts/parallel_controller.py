#!/usr/bin/env python3
"""
Parallel experiment controller for Phase 2.

Automatically detects available GPUs and distributes experiments across them.
Supports running multiple experiments in parallel on different GPUs.
"""

import argparse
import pandas as pd
import subprocess
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
import torch
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU hardware verification disabled.")


class GPUPool:
    """Manages a pool of GPUs for parallel experiment execution."""
    
    def __init__(self, gpu_ids: list[int] = None, hardware_filter: str = None):
        """
        Initialize GPU pool.
        
        Args:
            gpu_ids: List of GPU IDs to use (None = auto-detect all)
            hardware_filter: Filter GPUs by hardware type (e.g., 'h100', 'b200')
        """
        if gpu_ids is None:
            # Auto-detect all available GPUs
            all_gpu_ids = list(range(torch.cuda.device_count()))
            
            # Filter by hardware type if specified
            if hardware_filter and PYNVML_AVAILABLE:
                self.gpu_ids = self._filter_gpus_by_hardware(all_gpu_ids, hardware_filter)
                if not self.gpu_ids:
                    print(f"⚠️  No GPUs matching hardware type '{hardware_filter}' found!")
                    print(f"   Available GPUs: {self._get_gpu_info(all_gpu_ids)}")
                    import sys
                    sys.exit(1)
            else:
                self.gpu_ids = all_gpu_ids
        else:
            self.gpu_ids = gpu_ids
        
        self.available = queue.Queue()
        for gpu_id in self.gpu_ids:
            self.available.put(gpu_id)
        
        # Print GPU info
        gpu_info = self._get_gpu_info(self.gpu_ids) if PYNVML_AVAILABLE else str(self.gpu_ids)
        print(f"✓ GPU Pool initialized with {len(self.gpu_ids)} GPUs: {gpu_info}")
    
    def _get_gpu_name(self, gpu_id: int) -> str:
        """Get the name of a GPU."""
        if not PYNVML_AVAILABLE:
            return f"GPU {gpu_id}"
        
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle)
            pynvml.nvmlShutdown()
            # Handle both string and bytes return types
            return name.decode('utf-8') if isinstance(name, bytes) else name
        except Exception as e:
            return f"GPU {gpu_id} (error: {e})"
    
    def _get_gpu_info(self, gpu_ids: list[int]) -> str:
        """Get formatted GPU information."""
        return ", ".join([f"GPU {gid} ({self._get_gpu_name(gid)})" for gid in gpu_ids])
    
    def _filter_gpus_by_hardware(self, gpu_ids: list[int], hardware_filter: str) -> list[int]:
        """Filter GPUs by hardware type."""
        hardware_map = {
            'h100': ['H100', 'h100'],
            'h200': ['H200', 'h200'],
            'b200': ['B200', 'b200'],
            'mi300x': ['MI300X', 'mi300x', 'MI300']
        }
        
        target_names = hardware_map.get(hardware_filter.lower(), [hardware_filter])
        filtered_ids = []
        
        for gpu_id in gpu_ids:
            gpu_name = self._get_gpu_name(gpu_id)
            if any(target in gpu_name for target in target_names):
                filtered_ids.append(gpu_id)
        
        return filtered_ids
    
    def acquire(self, timeout=None):
        """Acquire a GPU from the pool."""
        return self.available.get(timeout=timeout)
    
    def release(self, gpu_id: int):
        """Release a GPU back to the pool."""
        self.available.put(gpu_id)
    
    def size(self):
        """Number of GPUs in pool."""
        return len(self.gpu_ids)


class ExperimentWorker(threading.Thread):
    """Worker thread that executes experiments on a specific GPU."""
    
    def __init__(self, worker_id: int, gpu_pool: GPUPool, work_queue: queue.Queue, 
                 results_queue: queue.Queue, matrix_file: str, max_prompts: int = None,
                 lock: threading.Lock = None, temperature: float = None, num_samples: int = None):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.gpu_pool = gpu_pool
        self.work_queue = work_queue
        self.results_queue = results_queue
        self.matrix_file = matrix_file
        self.max_prompts = max_prompts
        self.running = True
        self.lock = lock  # Lock for CSV file writes
        # NEW: Stochastic sampling parameters (FIX #5, backward compatible)
        self.temperature = temperature  # CLI override if provided
        self.num_samples = num_samples  # CLI override if provided
    
    def run(self):
        """Main worker loop."""
        while self.running:
            try:
                # Get work item (now a batch, not individual experiment)
                batch_data = self.work_queue.get(timeout=1)
                if batch_data is None:  # Poison pill
                    break
                
                # Check if this is a batch or single experiment
                is_batch = 'experiments' in batch_data
                
                if is_batch:
                    # Process entire batch (model loads ONCE)
                    is_70b = '70b' in batch_data['model'].lower()
                    is_mixtral_h100 = 'mixtral' in batch_data['model'].lower() and batch_data.get('hardware', '').lower() == 'h100'
                    is_b200 = batch_data.get('hardware', '').lower() == 'b200'
                    
                    # B200 has 192GB memory, can handle 70B models on single GPU
                    # Mixtral on H100 needs 2 GPUs (TP=2) to avoid OOM
                    if (is_70b and not is_b200) or is_mixtral_h100:
                        # Acquire 2 GPUs for 70B models on other hardware (tensor parallelism)
                        gpu_ids = []
                        for _ in range(2):
                            gpu_ids.append(self.gpu_pool.acquire())
                        
                        try:
                            model_desc = "Mixtral with TP=2" if is_mixtral_h100 else "70B model"
                            print(f"\n[Worker {self.worker_id} | GPUs {gpu_ids}] Starting BATCH: {batch_data['model']}/{batch_data['framework']} ({model_desc}, {len(batch_data['experiments'])} experiments)")
                            success = self._run_batch(batch_data, gpu_ids)
                        finally:
                            for gpu_id in gpu_ids:
                                self.gpu_pool.release(gpu_id)
                            self.work_queue.task_done()
                    else:
                        # Single GPU for smaller models OR 70B on B200
                        gpu_id = self.gpu_pool.acquire()
                        
                        try:
                            model_desc = f"70B on B200" if is_70b and is_b200 else "standard"
                            print(f"\n[Worker {self.worker_id} | GPU {gpu_id}] Starting BATCH: {batch_data['model']}/{batch_data['framework']} ({len(batch_data['experiments'])} experiments, {model_desc})")
                            success = self._run_batch(batch_data, gpu_id)
                        finally:
                            self.gpu_pool.release(gpu_id)
                            self.work_queue.task_done()
                else:
                    # Fallback: individual experiment (old behavior)
                    is_70b = '70b' in batch_data['model'].lower()
                    is_mixtral_h100 = 'mixtral' in batch_data['model'].lower() and batch_data.get('hardware', '').lower() == 'h100'
                    is_b200 = batch_data.get('hardware', '').lower() == 'b200'
                    
                    # B200 has 192GB memory, can handle 70B models on single GPU
                    # Mixtral on H100 needs 2 GPUs (TP=2) to avoid OOM
                    if (is_70b and not is_b200) or is_mixtral_h100:
                        # Acquire 2 GPUs for 70B models on other hardware (tensor parallelism)
                        gpu_ids = []
                        for _ in range(2):
                            gpu_ids.append(self.gpu_pool.acquire())
                        
                        try:
                            model_desc = "Mixtral with TP=2" if is_mixtral_h100 else "70B model"
                            print(f"\n[Worker {self.worker_id} | GPUs {gpu_ids}] Starting config {batch_data['config_id']} ({model_desc})")
                            success = self._run_experiment(batch_data, gpu_ids)
                            self._save_result_immediately(batch_data['config_id'], success)
                            self.results_queue.put((batch_data['config_id'], success, batch_data))
                        finally:
                            for gpu_id in gpu_ids:
                                self.gpu_pool.release(gpu_id)
                            self.work_queue.task_done()
                    else:
                        gpu_id = self.gpu_pool.acquire()
                        
                        try:
                            model_desc = " (70B on B200)" if is_70b and is_b200 else ""
                            print(f"\n[Worker {self.worker_id} | GPU {gpu_id}] Starting config {batch_data['config_id']}{model_desc}")
                            success = self._run_experiment(batch_data, gpu_id)
                            self._save_result_immediately(batch_data['config_id'], success)
                            self.results_queue.put((batch_data['config_id'], success, batch_data))
                        finally:
                            self.gpu_pool.release(gpu_id)
                            self.work_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error: {e}")
                self.work_queue.task_done()
    
    def _save_result_immediately(self, config_id: str, success: bool):
        """
        Save experiment result to CSV immediately (for spot instance fault tolerance).
        
        Uses lock to prevent concurrent writes from multiple workers.
        """
        import sys
        
        if self.lock is None:
            print(f"[Worker {self.worker_id}] ERROR: No lock provided for CSV writes!", flush=True)
            return  # No lock provided, skip immediate save
        
        with self.lock:
            try:
                # Reload CSV to get latest state
                df = pd.read_csv(self.matrix_file)
                
                # Find and update row - ensure type compatibility
                config_id_int = int(config_id) if isinstance(config_id, str) else config_id
                mask = df['config_id'] == config_id_int
                
                num_matched = mask.sum()
                if num_matched == 0:
                    print(f"[Worker {self.worker_id}] ERROR: config_id {config_id_int} not found in CSV!", flush=True)
                    return
                
                if success:
                    df.loc[mask, 'status'] = 'completed'
                    df.loc[mask, 'notes'] = str(f"Completed {datetime.now().isoformat()}")
                else:
                    df.loc[mask, 'status'] = 'failed'
                    df.loc[mask, 'notes'] = str(f"Failed {datetime.now().isoformat()}")
                
                # Save immediately
                df.to_csv(self.matrix_file, index=False)
                
                status = "✓ completed" if success else "✗ failed"
                print(f"[Worker {self.worker_id}] Saved status: {config_id_int} → {status}", flush=True)
                sys.stdout.flush()
                
            except Exception as e:
                import traceback
                print(f"[Worker {self.worker_id}] ERROR: Could not save result to CSV: {e}", flush=True)
                print(f"[Worker {self.worker_id}] Traceback: {traceback.format_exc()}", flush=True)
                sys.stdout.flush()
    
    def _run_batch(self, batch: dict, gpu_id) -> bool:
        """
        Run a batch of experiments (same model/hardware/framework/precision, different workloads).
        
        Model loads ONCE and runs all workloads - massive time savings!
        """
        import json
        import tempfile
        
        # Create batch config
        base_dir = Path(__file__).parent
        batch_config = {
            'model_path': str(base_dir / "models" / batch['model']),
            'hardware': batch['hardware'],
            'precision': batch['precision'],
            'framework': batch['framework'],
            'workloads': batch['workloads'],
            'seed': 42,
            'replicate_id': 1,
            'output_base_dir': 'results'  # Save locally in reproduction_scripts/results/
        }
        
        # Save config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_config, f)
            config_file = f.name
        
        # Set CUDA_VISIBLE_DEVICES
        env = subprocess.os.environ.copy()
        if isinstance(gpu_id, list):
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_id)
            gpu_str = f"GPUs {gpu_id}"
        else:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            gpu_str = f"GPU {gpu_id}"
        
        try:
            start_time = time.time()
            
            # Run batch - use system Python (Docker has all packages pre-installed)
            python_bin = 'python'
            
            # Use local scripts directory  
            script_path = str(Path(__file__).parent / "scripts" / "run_experiment.py")
            
            # Run each experiment in the batch sequentially
            # Each experiment shares same model/hardware/framework/precision
            for exp in batch['experiments']:
                # Build workload paths
                workload_files = {
                    'safety': 'safety/advbench_prompts.jsonl',
                    'code': 'code/humaneval_prompts.jsonl', 
                    'math': 'math/gsm8k_prompts.jsonl',
                    'chat': 'chat/lmsys_prompts.jsonl',
                    'longform': 'longform/long_qa_prompts.jsonl'
                }
                
                model_path = str(base_dir / "models" / exp['model'])
                prompt_file = str(base_dir / "data" / workload_files[exp['workload']])
                output_dir = str(base_dir / "results" / exp['model'] / exp['hardware'] / exp['precision'] / exp['framework'] / exp['workload'])
                
                cmd = [
                    python_bin,
                    script_path,
                    '--model', model_path,
                    '--hardware', exp['hardware'],
                    '--precision', exp['precision'],
                    '--framework', exp['framework'],
                    '--prompt-file', prompt_file,
                    '--output-dir', output_dir
                ]
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env
                )
            
            elapsed = time.time() - start_time
            
            # Mark all experiments in batch as completed
            for exp in batch['experiments']:
                self._save_result_immediately(exp['config_id'], True)
                # Also put in results queue for live progress tracking
                self.results_queue.put((exp['config_id'], True, exp))
            
            print(f"[Worker {self.worker_id} | {gpu_str}] ✓ BATCH completed: {len(batch['experiments'])} experiments in {elapsed:.1f}s ({elapsed/len(batch['experiments']):.1f}s per exp)")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[Worker {self.worker_id} | {gpu_str}] ✗ BATCH failed: {e}")
            print(f"stderr: {e.stderr[-1000:]}")
            
            # Mark all experiments in batch as failed
            for exp in batch['experiments']:
                self._save_result_immediately(exp['config_id'], False)
                # Also put in results queue for live progress tracking
                self.results_queue.put((exp['config_id'], False, exp))
            
            return False
        except Exception as e:
            print(f"[Worker {self.worker_id} | {gpu_str}] ✗ BATCH error: {e}")
            
            # Mark all experiments in batch as failed
            for exp in batch['experiments']:
                self._save_result_immediately(exp['config_id'], False)
                # Also put in results queue for live progress tracking
                self.results_queue.put((exp['config_id'], False, exp))
            
            return False
        finally:
            # Clean up temp config file
            import os
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def _run_experiment(self, row: dict, gpu_id) -> bool:
        """
        Execute a single experiment on specified GPU(s).
        
        Args:
            row: Experiment configuration
            gpu_id: Either int (single GPU) or list[int] (multiple GPUs for 70B)
        """
        
        # Map workload to dataset file
        workload_files = {
            'code': 'code/humaneval_prompts.jsonl',
            'math': 'math/gsm8k_prompts.jsonl',
            'safety': 'safety/advbench_prompts.jsonl',
            'chat': 'chat/lmsys_prompts.jsonl',
            'long_context': 'long_context/long_qa_prompts.jsonl'
        }
        
        # Use local models and data folders
        base_dir = Path(__file__).parent
        model_path = str(base_dir / "models" / row['model'])
        prompt_file = str(base_dir / "data" / workload_files[row['workload']])
        
        # Use local results folder
        output_dir = str(base_dir / "results" / row['model'] / row['hardware'] / row['precision'] / row['framework'] / row['workload']) + "/"
        
        # Build command - use system Python (Docker has all packages pre-installed)
        python_bin = 'python'
        
        # Use local run_experiment.py script
        script_path = str(Path(__file__).parent / "scripts" / "run_experiment.py")
        
        cmd = [
            python_bin, script_path,
            '--model', model_path,
            '--hardware', row['hardware'],
            '--precision', row['precision'],
            '--framework', row['framework'],
            '--prompt-file', prompt_file,
            '--output-dir', output_dir,
            '--seed', '42',
            '--replicate-id', '1'
        ]
        
        if self.max_prompts:
            cmd.extend(['--max-prompts', str(self.max_prompts)])
        
        # NEW: Add stochastic sampling parameters if provided (FIX #5, backward compatible)
        # Priority: CLI override > CSV value > default (0.0/1)
        temperature = self.temperature if self.temperature is not None else row.get('temperature', 0.0)
        num_samples = self.num_samples if self.num_samples is not None else row.get('num_samples', 1)
        
        # Only add flags if non-default values (backward compatible)
        if temperature != 0.0:
            cmd.extend(['--temperature', str(temperature)])
        if num_samples != 1:
            cmd.extend(['--num-samples', str(num_samples)])
        
        # Set CUDA_VISIBLE_DEVICES - handle both single GPU and multiple GPUs
        env = subprocess.os.environ.copy()
        if isinstance(gpu_id, list):
            # Multiple GPUs for 70B models
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_id)
            gpu_str = f"GPUs {gpu_id}"
        else:
            # Single GPU
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            gpu_str = f"GPU {gpu_id}"
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                env=env
            )
            elapsed = time.time() - start_time
            
            print(f"[Worker {self.worker_id} | {gpu_str}] ✓ Config {row['config_id']} completed ({elapsed:.1f}s)")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[Worker {self.worker_id} | {gpu_str}] ✗ Config {row['config_id']} failed: {e}")
            print(f"stderr: {e.stderr[-500:]}")
            return False
        except Exception as e:
            print(f"[Worker {self.worker_id} | {gpu_str}] ✗ Config {row['config_id']} error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="DriftBench Phase 2 parallel experiment controller")
    parser.add_argument("--matrix", default="experiment_matrix.csv", 
                       help="Path to experiment matrix CSV")
    parser.add_argument("--hardware", default=None, 
                       help="Filter by hardware (h100, h200, b200, mi300x)")
    parser.add_argument("--priority", default=None, 
                       help="Filter by priority (P0, P1)")
    parser.add_argument("--model", default=None,
                       help="Filter by model")
    parser.add_argument("--framework", default=None,
                       help="Filter by framework (vllm, tensorrt-llm, sglang)")
    parser.add_argument("--precision", default=None,
                       help="Filter by precision (fp16, fp8, fp4)")
    parser.add_argument("--max-experiments", type=int, default=None, 
                       help="Limit number of experiments")
    parser.add_argument("--max-prompts", type=int, default=None, 
                       help="Limit prompts per experiment (for testing)")
    parser.add_argument("--gpus", default=None,
                       help="GPU IDs to use (comma-separated, e.g., '0,1,2' or 'all')")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of worker threads (default: num GPUs)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print what would be executed without running")
    parser.add_argument("--resume", action="store_true", 
                       help="Skip completed experiments")
    
    # NEW: Stochastic sampling parameters (FIX #5, backward compatible)
    parser.add_argument("--temperature", type=float, default=None,
                       help="Override temperature for all experiments (default: use CSV value or 0.0)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Override num_samples for all experiments (default: use CSV value or 1)")
    
    args = parser.parse_args()
    
    # Setup GPU pool
    if args.gpus and args.gpus != 'all':
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_ids = None  # Auto-detect
    
    # Initialize GPU pool with hardware filter for verification
    gpu_pool = GPUPool(gpu_ids, hardware_filter=args.hardware)
    
    # Determine number of workers (will refine after checking workload)
    num_workers = args.workers if args.workers else None
    
    # Load experiment matrix
    print(f"\nLoading experiment matrix from {args.matrix}")
    df = pd.read_csv(args.matrix)
    
    # Apply filters
    if args.hardware:
        df = df[df['hardware'] == args.hardware]
        print(f"Filtered to hardware: {args.hardware}")
    
    if args.priority:
        df = df[df['priority'] == args.priority]
        print(f"Filtered to priority: {args.priority}")
    
    if args.model:
        df = df[df['model'] == args.model]
        print(f"Filtered to model: {args.model}")
    
    if args.framework:
        df = df[df['framework'] == args.framework]
        print(f"Filtered to framework: {args.framework}")
    
    if args.precision:
        df = df[df['precision'] == args.precision]
        print(f"Filtered to precision: {args.precision}")
    
    # Filter pending if resume
    if args.resume:
        df = df[df['status'] == 'pending']
        print(f"Resuming: {len(df)} pending experiments")
    
    # Limit experiments
    if args.max_experiments:
        df = df.head(args.max_experiments)
    
    # SMART BATCHING: Sort experiments to minimize model reloads
    # Group by (model, hardware, framework, precision) - workload varies freely
    # This allows running all workloads on same loaded model before switching
    print("\n🔄 Optimizing experiment order (smart batching)...")
    df = df.sort_values(by=['model', 'hardware', 'framework', 'precision', 'workload'])
    
    # Calculate reload estimates
    reload_groups = df.groupby(['model', 'hardware', 'framework', 'precision']).size()
    num_reloads = len(reload_groups)
    print(f"   Batched into {num_reloads} model-load groups (vs {len(df)} naive reloads)")
    print(f"   Estimated time saved: ~{(len(df) - num_reloads) * 2:.0f} minutes")
    
    # Smart worker count: Optimize based on model sizes
    if num_workers is None:
        has_70b = df['model'].str.contains('70b', case=False).any()
        has_small = (~df['model'].str.contains('70b', case=False)).any()
        
        if has_70b and not has_small:
            # Only 70B models: each needs 2 GPUs
            num_workers = gpu_pool.size() // 2
            print(f"\n💡 Detected only 70B models → Using {num_workers} workers (2 GPUs per 70B model)")
        elif has_small and not has_70b:
            # Only small models: each needs 1 GPU
            num_workers = gpu_pool.size()
            print(f"\n💡 Detected only small models → Using {num_workers} workers (1 GPU per model)")
        else:
            # Mixed: Use all GPUs, let dynamic allocation handle it
            num_workers = gpu_pool.size()
            print(f"\n💡 Detected mixed workload → Using {num_workers} workers (dynamic GPU allocation)")
    
    print(f"\n{'='*70}")
    print(f"Parallel Execution Plan:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Model reloads needed: {num_reloads} (optimized)")
    print(f"  GPUs: {gpu_pool.size()}")
    print(f"  Workers: {num_workers}")
    if df['model'].str.contains('70b', case=False).any():
        print(f"  70B models: 2 GPUs each (tensor parallelism)")
        print(f"  Parallelism: Up to {gpu_pool.size() // 2} concurrent 70B batches")
    else:
        print(f"  Parallelism: Up to {min(num_workers, gpu_pool.size())} concurrent experiments")
    print(f"{'='*70}")
    
    if args.dry_run:
        print("\nDRY RUN - Would execute (in optimized order):")
        for idx, row in df.iterrows():
            print(f"  {row['config_id']}: {row['model']}/{row['hardware']}/{row['precision']}/{row['framework']}/{row['workload']}")
        return
    
    # Create work queue and CSV write lock
    work_queue = queue.Queue()
    results_queue = queue.Queue()
    csv_lock = threading.Lock()  # Protect concurrent CSV writes
    
    # Create batches: group experiments by (model, hardware, framework, precision)
    # Each batch will load model ONCE and run all workloads
    print("\n📦 Creating execution batches...")
    batches = []
    for (model, hardware, framework, precision), group in df.groupby(['model', 'hardware', 'framework', 'precision']):
        batch = {
            'model': model,
            'hardware': hardware,
            'framework': framework,
            'precision': precision,
            'experiments': group.to_dict('records'),  # All experiments in this batch
            'workloads': group['workload'].unique().tolist()
        }
        batches.append(batch)
        print(f"   Batch: {model}/{hardware}/{framework}/{precision} → {len(group)} experiments ({', '.join(batch['workloads'])})")
    
    print(f"\n✓ Created {len(batches)} batches (1 model load per batch)")
    print(f"   Total experiments: {len(df)}")
    print(f"   Model loads: {len(batches)} (saved {len(df) - len(batches)} redundant loads!)")
    
    # Add batches to queue
    for batch in batches:
        work_queue.put(batch)
    
    # Start workers
    workers = []
    for i in range(num_workers):
        worker = ExperimentWorker(
            worker_id=i,
            gpu_pool=gpu_pool,
            work_queue=work_queue,
            results_queue=results_queue,
            matrix_file=args.matrix,
            max_prompts=args.max_prompts,
            lock=csv_lock,  # Pass lock for immediate saves
            temperature=args.temperature,  # NEW: FIX #5 support
            num_samples=args.num_samples   # NEW: FIX #5 support
        )
        worker.start()
        workers.append(worker)
    
    print(f"\n✓ Started {num_workers} workers")
    print(f"✓ Processing {len(df)} experiments...\n")
    
    # Monitor progress with live updates
    total_experiments = len(df)
    completed = 0
    failed = 0
    start_time = time.time()
    
    print(f"\nRunning experiments... please wait\n")
    
    # Progress monitoring loop
    while completed + failed < total_experiments:
        # Collect completed results
        while not results_queue.empty():
            config_id, success, row_data = results_queue.get()
            if success:
                completed += 1
            else:
                failed += 1

            # Print updated progress
            total_done = completed + failed
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            eta = (total_experiments - total_done) / rate if rate > 0 else 0

            print(f"\r[{total_done}/{total_experiments}] ✓ {completed} completed | ✗ {failed} failed | "
                  f"⏱️ {elapsed:.0f}s elapsed | ETA: {eta:.0f}s", end='', flush=True)

        time.sleep(1)  # Update every second

    # Stop workers
    for worker in workers:
        worker.running = False

    for _ in range(num_workers):
        work_queue.put(None)  # Poison pill (in case worker is blocked on get)

    for worker in workers:
        worker.join(timeout=5)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Completed: {completed}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Total: {completed + failed}/{total_experiments}")
    print(f"✨ Success rate: {completed/(completed+failed)*100:.1f}%" if (completed+failed) > 0 else "N/A")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"🖥️  GPUs used: {gpu_pool.size()}")
    print(f"⚡ Workers: {num_workers}")
    print(f"⚡ Avg time per experiment: {total_time/(completed+failed):.1f}s" if (completed+failed) > 0 else "N/A")


if __name__ == "__main__":
    main()
