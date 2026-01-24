import logging
import time
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("Importing original Engram implementation")
    from engram_demo_v1 import (
        Engram as EngramOriginal,
        TransformerBlock as TransformerBlockOriginal,
        EngramConfig as EngramConfigOriginal,
        BackBoneConfig as BackBoneConfigOriginal,
    )
    logger.info("Original Engram imported successfully")
except ImportError as e:
    logger.error("Failed to import original Engram: %s", e)
    sys.exit(1)

try:
    logger.info("Importing optimized Engram implementation")
    from engram_optimized import (
        Engram as EngramOptimized,
        TransformerBlock as TransformerBlockOptimized,
        EngramConfig as EngramConfigOptimized,
        BackBoneConfig as BackBoneConfigOptimized,
    )
    logger.info("Optimized Engram imported successfully")
except ImportError as e:
    logger.error("Failed to import optimized Engram: %s", e)
    sys.exit(1)

@dataclass
class BenchmarkResult:
    model_type: str
    layer_id: int
    batch_size: int
    seq_len: int
    forward_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate_pct: float = 0.0
    iterations: int = 100
    device: str = "cuda"

class MemoryTracker:
    """Track GPU memory usage"""
    def __init__(self, device):
        self.device = device
        self.enabled = device.type == 'cuda'
        logger.info("MemoryTracker initialized for device=%s, enabled=%s", 
                   device, self.enabled)
    
    def reset(self):
        if self.enabled:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.debug("GPU memory cache cleared and stats reset")
    
    def get_current_usage(self):
        if self.enabled:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            logger.debug("Current memory - allocated: %.2f MB, reserved: %.2f MB",
                        allocated, reserved)
            return allocated, reserved
        return 0.0, 0.0
    
    def get_peak_usage(self):
        if self.enabled:
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
            logger.debug("Peak memory - allocated: %.2f MB, reserved: %.2f MB",
                        peak_allocated, peak_reserved)
            return peak_allocated, peak_reserved
        return 0.0, 0.0

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    logger.info("Setting random seed to %d", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_test_data(batch_size, seq_len, vocab_size, device):
    """Generate synthetic test data"""
    logger.debug("Creating test data: batch_size=%d, seq_len=%d, vocab_size=%d",
                batch_size, seq_len, vocab_size)
    set_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids

def create_hidden_states(batch_size, seq_len, hidden_size, hc_mult, device):
    """Generate synthetic hidden states"""
    logger.debug("Creating hidden states: batch_size=%d, seq_len=%d, hidden_size=%d, hc_mult=%d",
                batch_size, seq_len, hidden_size, hc_mult)
    set_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hc_mult, hidden_size, device=device)
    return hidden_states

def warmup_model(model, input_ids, hidden_states, warmup_iterations=10):
    """Warmup model before benchmarking"""
    logger.info("Starting model warmup with %d iterations", warmup_iterations)
    model.eval()
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = model(input_ids, hidden_states)
            if i % 5 == 0:
                logger.debug("Warmup iteration %d/%d", i+1, warmup_iterations)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info("Model warmup complete")

def benchmark_forward_pass(model, input_ids, hidden_states, iterations=100):
    """Benchmark forward pass latency"""
    logger.info("Starting benchmark with %d iterations", iterations)
    
    model.eval()
    
    warmup_model(model, input_ids, hidden_states, warmup_iterations=10)
    
    times = []
    
    logger.info("Running timed iterations")
    with torch.no_grad():
        for i in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_ids, hidden_states)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start_time) * 1000
            times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                logger.debug("Completed %d/%d iterations, current avg: %.2fms",
                           i+1, iterations, np.mean(times))
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    
    logger.info("Benchmark statistics:")
    logger.info("  Mean: %.2f ms", mean_time)
    logger.info("  Std: %.2f ms", std_time)
    logger.info("  Median: %.2f ms", median_time)
    logger.info("  Min: %.2f ms", min_time)
    logger.info("  Max: %.2f ms", max_time)
    logger.info("  P95: %.2f ms", p95_time)
    logger.info("  P99: %.2f ms", p99_time)
    
    return {
        'mean': mean_time,
        'std': std_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'p95': p95_time,
        'p99': p99_time,
        'all_times': times
    }

def benchmark_model(model, model_type, layer_id, batch_size, seq_len, 
                   backbone_config, device, memory_tracker, iterations=100):
    """Benchmark a single model configuration"""
    logger.info("="*80)
    logger.info("Benchmarking %s model", model_type)
    logger.info("  Layer ID: %d", layer_id)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Sequence length: %d", seq_len)
    logger.info("="*80)
    
    memory_tracker.reset()
    
    input_ids = create_test_data(batch_size, seq_len, backbone_config.vocab_size, device)
    hidden_states = create_hidden_states(
        batch_size, seq_len,
        backbone_config.hidden_size,
        backbone_config.hc_mult,
        device
    )
    
    logger.info("Moving model to device: %s", device)
    model = model.to(device)
    model.eval()
    
    logger.info("Measuring initial memory footprint")
    initial_alloc, initial_reserved = memory_tracker.get_current_usage()
    logger.info("Initial memory - allocated: %.2f MB, reserved: %.2f MB",
               initial_alloc, initial_reserved)
    
    logger.info("Starting forward pass benchmark")
    timing_stats = benchmark_forward_pass(model, input_ids, hidden_states, iterations)
    
    logger.info("Measuring peak memory usage")
    peak_alloc, peak_reserved = memory_tracker.get_peak_usage()
    logger.info("Peak memory - allocated: %.2f MB, reserved: %.2f MB",
               peak_alloc, peak_reserved)
    
    cache_hits = 0
    cache_misses = 0
    cache_hit_rate = 0.0
    
    if model_type == "optimized":
        logger.info("Extracting cache statistics from optimized model")
        if hasattr(model, 'engram') and model.engram is not None:
            if hasattr(model.engram.multi_head_embedding, 'cache'):
                cache = model.engram.multi_head_embedding.cache
                cache_hits = cache.hits
                cache_misses = cache.misses
                total_requests = cache_hits + cache_misses
                cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
                
                logger.info("Cache statistics:")
                logger.info("  Hits: %d", cache_hits)
                logger.info("  Misses: %d", cache_misses)
                logger.info("  Total requests: %d", total_requests)
                logger.info("  Hit rate: %.2f%%", cache_hit_rate)
    
    result = BenchmarkResult(
        model_type=model_type,
        layer_id=layer_id,
        batch_size=batch_size,
        seq_len=seq_len,
        forward_time_ms=timing_stats['mean'],
        memory_allocated_mb=peak_alloc,
        memory_reserved_mb=peak_reserved,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_hit_rate_pct=cache_hit_rate,
        iterations=iterations,
        device=str(device)
    )
    
    logger.info("Benchmark complete for %s model", model_type)
    logger.info("  Mean latency: %.2f ms", result.forward_time_ms)
    logger.info("  Memory allocated: %.2f MB", result.memory_allocated_mb)
    
    return result, timing_stats

def run_benchmark_suite(
    layer_ids: List[int],
    batch_sizes: List[int],
    seq_lens: List[int],
    iterations: int = 100
):
    """Run comprehensive benchmark suite"""
    logger.info("#"*80)
    logger.info("STARTING BENCHMARK SUITE")
    logger.info("#"*80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA capability: %s", torch.cuda.get_device_capability(0))
        logger.info("Total GPU memory: %.2f GB", 
                   torch.cuda.get_device_properties(0).total_memory / 1024**3)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    
    memory_tracker = MemoryTracker(device)
    
    backbone_config = BackBoneConfigOriginal()
    
    logger.info("Benchmark configuration:")
    logger.info("  Layer IDs: %s", layer_ids)
    logger.info("  Batch sizes: %s", batch_sizes)
    logger.info("  Sequence lengths: %s", seq_lens)
    logger.info("  Iterations per test: %d", iterations)
    
    all_results = []
    all_timing_stats = []
    
    total_tests = len(layer_ids) * len(batch_sizes) * len(seq_lens) * 2
    current_test = 0
    
    for layer_id in layer_ids:
        logger.info("")
        logger.info("Testing layer %d", layer_id)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                current_test += 1
                logger.info("")
                logger.info("Test %d/%d: batch_size=%d, seq_len=%d",
                           current_test, total_tests, batch_size, seq_len)
                
                try:
                    logger.info("Benchmarking ORIGINAL model")
                    model_orig = TransformerBlockOriginal(layer_id=layer_id)
                    result_orig, timing_orig = benchmark_model(
                        model_orig, "original", layer_id, batch_size, seq_len,
                        backbone_config, device, memory_tracker, iterations
                    )
                    all_results.append(result_orig)
                    all_timing_stats.append({
                        'model_type': 'original',
                        'layer_id': layer_id,
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'stats': timing_orig
                    })
                    
                    del model_orig
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    current_test += 1
                    logger.info("")
                    logger.info("Test %d/%d: batch_size=%d, seq_len=%d",
                               current_test, total_tests, batch_size, seq_len)
                    
                    logger.info("Benchmarking OPTIMIZED model")
                    model_opt = TransformerBlockOptimized(layer_id=layer_id)
                    result_opt, timing_opt = benchmark_model(
                        model_opt, "optimized", layer_id, batch_size, seq_len,
                        backbone_config, device, memory_tracker, iterations
                    )
                    all_results.append(result_opt)
                    all_timing_stats.append({
                        'model_type': 'optimized',
                        'layer_id': layer_id,
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'stats': timing_opt
                    })
                    
                    speedup = result_orig.forward_time_ms / result_opt.forward_time_ms
                    memory_reduction_pct = (
                        (result_orig.memory_allocated_mb - result_opt.memory_allocated_mb) /
                        result_orig.memory_allocated_mb * 100
                    )
                    
                    logger.info("")
                    logger.info("COMPARISON for layer=%d, bs=%d, seqlen=%d:",
                               layer_id, batch_size, seq_len)
                    logger.info("  Speedup: %.2fx", speedup)
                    logger.info("  Original latency: %.2f ms", result_orig.forward_time_ms)
                    logger.info("  Optimized latency: %.2f ms", result_opt.forward_time_ms)
                    logger.info("  Memory reduction: %.2f%%", memory_reduction_pct)
                    logger.info("  Cache hit rate: %.2f%%", result_opt.cache_hit_rate_pct)
                    
                    del model_opt
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error("Exception during benchmark: %s", e)
                    logger.exception("Full traceback:")
    
    logger.info("")
    logger.info("#"*80)
    logger.info("BENCHMARK SUITE COMPLETE")
    logger.info("#"*80)
    
    return all_results, all_timing_stats

def save_results(results: List[BenchmarkResult], timing_stats: List[Dict], output_dir: str = "results"):
    """Save benchmark results to JSON and CSV"""
    logger.info("Saving results to directory: %s", output_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info("Created output directory: %s", output_path.absolute())
    
    results_json = [asdict(r) for r in results]
    json_path = output_path / "benchmark_results.json"
    
    logger.info("Writing results to JSON: %s", json_path)
    with open(json_path, 'w') as f:
        json.dump({
            'results': results_json,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    logger.info("JSON results saved: %d entries", len(results_json))
    
    df = pd.DataFrame(results_json)
    csv_path = output_path / "benchmark_results.csv"
    
    logger.info("Writing results to CSV: %s", csv_path)
    df.to_csv(csv_path, index=False)
    logger.info("CSV results saved: %d rows, %d columns", len(df), len(df.columns))
    
    timing_json_path = output_path / "timing_statistics.json"
    logger.info("Writing detailed timing stats to: %s", timing_json_path)
    
    timing_serializable = []
    for stat in timing_stats:
        stat_copy = stat.copy()
        if 'all_times' in stat_copy['stats']:
            stat_copy['stats']['all_times'] = [float(t) for t in stat_copy['stats']['all_times']]
        timing_serializable.append(stat_copy)
    
    with open(timing_json_path, 'w') as f:
        json.dump(timing_serializable, f, indent=2)
    logger.info("Timing statistics saved")
    
    return df

def generate_plots(df: pd.DataFrame, output_dir: str = "results"):
    """Generate comparison plots"""
    logger.info("Generating visualization plots")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    df_pivot = df.pivot_table(
        values='forward_time_ms',
        index=['layer_id', 'batch_size', 'seq_len'],
        columns='model_type'
    ).reset_index()
    
    df_pivot['speedup'] = df_pivot['original'] / df_pivot['optimized']
    
    logger.info("Creating speedup comparison plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = [f"L{row['layer_id']}_B{row['batch_size']}_S{row['seq_len']}" 
                for _, row in df_pivot.iterrows()]
    x_pos = np.arange(len(x_labels))
    
    width = 0.35
    ax.bar(x_pos - width/2, df_pivot['original'], width, label='Original', alpha=0.8)
    ax.bar(x_pos + width/2, df_pivot['optimized'], width, label='Optimized', alpha=0.8)
    
    ax.set_xlabel('Configuration (Layer_BatchSize_SeqLen)')
    ax.set_ylabel('Forward Pass Time (ms)')
    ax.set_title('Original vs Optimized Engram Performance')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = output_path / "speedup_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info("Speedup plot saved: %s", plot_path)
    plt.close()
    
    logger.info("Creating speedup factor plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if s > 1 else 'red' for s in df_pivot['speedup']]
    bars = ax.bar(x_pos, df_pivot['speedup'], color=colors, alpha=0.7)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No speedup')
    ax.set_xlabel('Configuration (Layer_BatchSize_SeqLen)')
    ax.set_ylabel('Speedup Factor (Original / Optimized)')
    ax.set_title('Speedup Factor Across Configurations')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, speedup) in enumerate(zip(bars, df_pivot['speedup'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=8)
    
    plot_path = output_path / "speedup_factor.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info("Speedup factor plot saved: %s", plot_path)
    plt.close()
    
    logger.info("Creating memory comparison plot")
    df_memory = df.pivot_table(
        values='memory_allocated_mb',
        index=['layer_id', 'batch_size', 'seq_len'],
        columns='model_type'
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x_pos - width/2, df_memory['original'], width, label='Original', alpha=0.8)
    ax.bar(x_pos + width/2, df_memory['optimized'], width, label='Optimized', alpha=0.8)
    
    ax.set_xlabel('Configuration (Layer_BatchSize_SeqLen)')
    ax.set_ylabel('Peak Memory Allocated (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plot_path = output_path / "memory_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info("Memory comparison plot saved: %s", plot_path)
    plt.close()
    
    df_opt = df[df['model_type'] == 'optimized']
    if not df_opt.empty and df_opt['cache_hit_rate_pct'].sum() > 0:
        logger.info("Creating cache hit rate plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cache_x_labels = [f"L{row['layer_id']}_B{row['batch_size']}_S{row['seq_len']}" 
                         for _, row in df_opt.iterrows()]
        cache_x_pos = np.arange(len(cache_x_labels))
        
        colors_cache = ['green' if rate > 50 else 'orange' if rate > 25 else 'red' 
                        for rate in df_opt['cache_hit_rate_pct']]
        bars = ax.bar(cache_x_pos, df_opt['cache_hit_rate_pct'], color=colors_cache, alpha=0.7)
        
        ax.set_xlabel('Configuration (Layer_BatchSize_SeqLen)')
        ax.set_ylabel('Cache Hit Rate (%)')
        ax.set_title('LRU Cache Hit Rate (Optimized Model)')
        ax.set_xticks(cache_x_pos)
        ax.set_xticklabels(cache_x_labels, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, df_opt['cache_hit_rate_pct']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        
        plot_path = output_path / "cache_hit_rate.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info("Cache hit rate plot saved: %s", plot_path)
        plt.close()
    
    logger.info("All plots generated successfully")

def print_summary(df: pd.DataFrame):
    """Print summary statistics"""
    logger.info("")
    logger.info("#"*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("#"*80)
    
    df_pivot = df.pivot_table(
        values='forward_time_ms',
        index=['layer_id', 'batch_size', 'seq_len'],
        columns='model_type'
    ).reset_index()
    
    df_pivot['speedup'] = df_pivot['original'] / df_pivot['optimized']
    
    logger.info("")
    logger.info("Speedup Statistics:")
    logger.info("  Mean speedup: %.2fx", df_pivot['speedup'].mean())
    logger.info("  Median speedup: %.2fx", df_pivot['speedup'].median())
    logger.info("  Min speedup: %.2fx", df_pivot['speedup'].min())
    logger.info("  Max speedup: %.2fx", df_pivot['speedup'].max())
    logger.info("  Std speedup: %.2fx", df_pivot['speedup'].std())
    
    logger.info("")
    logger.info("Latency Statistics (ms):")
    
    df_orig = df[df['model_type'] == 'original']
    df_opt = df[df['model_type'] == 'optimized']
    
    logger.info("  Original - Mean: %.2f, Median: %.2f, Std: %.2f",
               df_orig['forward_time_ms'].mean(),
               df_orig['forward_time_ms'].median(),
               df_orig['forward_time_ms'].std())
    
    logger.info("  Optimized - Mean: %.2f, Median: %.2f, Std: %.2f",
               df_opt['forward_time_ms'].mean(),
               df_opt['forward_time_ms'].median(),
               df_opt['forward_time_ms'].std())
    
    if not df_opt.empty and df_opt['memory_allocated_mb'].sum() > 0:
        logger.info("")
        logger.info("Memory Statistics (MB):")
        
        memory_reduction = (
            (df_orig['memory_allocated_mb'].mean() - df_opt['memory_allocated_mb'].mean()) /
            df_orig['memory_allocated_mb'].mean() * 100
        )
        
        logger.info("  Original - Mean: %.2f, Median: %.2f",
                   df_orig['memory_allocated_mb'].mean(),
                   df_orig['memory_allocated_mb'].median())
        
        logger.info("  Optimized - Mean: %.2f, Median: %.2f",
                   df_opt['memory_allocated_mb'].mean(),
                   df_opt['memory_allocated_mb'].median())
        
        logger.info("  Average memory reduction: %.2f%%", memory_reduction)
    
    if not df_opt.empty and df_opt['cache_hit_rate_pct'].sum() > 0:
        logger.info("")
        logger.info("Cache Statistics:")
        logger.info("  Mean hit rate: %.2f%%", df_opt['cache_hit_rate_pct'].mean())
        logger.info("  Median hit rate: %.2f%%", df_opt['cache_hit_rate_pct'].median())
        logger.info("  Min hit rate: %.2f%%", df_opt['cache_hit_rate_pct'].min())
        logger.info("  Max hit rate: %.2f%%", df_opt['cache_hit_rate_pct'].max())
    
    logger.info("")
    logger.info("#"*80)

if __name__ == '__main__':
    logger.info("Starting benchmark script")
    
    cfg = EngramConfigOriginal()
    
    layer_ids = cfg.layer_ids
    batch_sizes = [2, 4, 8]
    seq_lens = [128, 256, 512]
    iterations = 100
    
    logger.info("Benchmark parameters:")
    logger.info("  Layer IDs: %s", layer_ids)
    logger.info("  Batch sizes: %s", batch_sizes)
    logger.info("  Sequence lengths: %s", seq_lens)
    logger.info("  Iterations: %d", iterations)
    
    results, timing_stats = run_benchmark_suite(
        layer_ids=layer_ids,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        iterations=iterations
    )
    
    logger.info("Saving results and generating visualizations")
    df = save_results(results, timing_stats)
    
    generate_plots(df)
    
    print_summary(df)
    
    logger.info("Benchmark complete. Results saved to 'results/' directory")