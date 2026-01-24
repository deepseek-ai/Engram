"""
Correctness testing: Compare original vs optimized Engram outputs
"""

import logging
import time
import sys
import torch
import numpy as np
from transformers import AutoTokenizer

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

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    logger.info("Setting random seed to %d for reproducibility", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set successfully")

def create_test_data(batch_size, seq_len, vocab_size, device):
    """Generate synthetic test data"""
    logger.info("Creating test data: batch_size=%d, seq_len=%d, vocab_size=%d",
               batch_size, seq_len, vocab_size)
    
    set_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logger.debug("Generated input_ids with shape=%s, dtype=%s", 
                input_ids.shape, input_ids.dtype)
    
    return input_ids

def create_hidden_states(batch_size, seq_len, hidden_size, hc_mult, device):
    """Generate synthetic hidden states"""
    logger.info("Creating hidden states: batch_size=%d, seq_len=%d, hidden_size=%d, hc_mult=%d",
               batch_size, seq_len, hidden_size, hc_mult)
    
    set_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hc_mult, hidden_size, device=device)
    logger.debug("Generated hidden_states with shape=%s, dtype=%s",
                hidden_states.shape, hidden_states.dtype)
    
    return hidden_states

def copy_weights(model_src, model_dst):
    """Copy weights from source model to destination model"""
    logger.info("Copying weights from source to destination model")
    
    src_state = model_src.state_dict()
    dst_state = model_dst.state_dict()
    
    logger.debug("Source model has %d parameters", len(src_state))
    logger.debug("Destination model has %d parameters", len(dst_state))
    
    matched_keys = 0
    missing_keys = []
    unexpected_keys = []
    
    for key in dst_state.keys():
        if key in src_state:
            if dst_state[key].shape == src_state[key].shape:
                dst_state[key].copy_(src_state[key])
                matched_keys += 1
                logger.debug("Copied parameter: %s with shape=%s", key, src_state[key].shape)
            else:
                logger.warning("Shape mismatch for key=%s: src=%s, dst=%s",
                             key, src_state[key].shape, dst_state[key].shape)
        else:
            missing_keys.append(key)
    
    for key in src_state.keys():
        if key not in dst_state:
            unexpected_keys.append(key)
    
    logger.info("Weight copying complete: matched=%d, missing=%d, unexpected=%d",
               matched_keys, len(missing_keys), len(unexpected_keys))
    
    if missing_keys:
        logger.warning("Missing keys in destination model: %s", missing_keys[:5])
    if unexpected_keys:
        logger.warning("Unexpected keys in source model: %s", unexpected_keys[:5])
    
    model_dst.load_state_dict(dst_state)
    logger.info("Destination model state_dict loaded")

def compare_outputs(output_orig, output_opt, rtol=1e-4, atol=1e-5):
    """Compare two tensors and return detailed statistics"""
    logger.info("Comparing outputs with rtol=%.2e, atol=%.2e", rtol, atol)
    
    logger.debug("Original output - shape=%s, dtype=%s, device=%s",
                output_orig.shape, output_orig.dtype, output_orig.device)
    logger.debug("Optimized output - shape=%s, dtype=%s, device=%s",
                output_opt.shape, output_opt.dtype, output_opt.device)
    
    if output_orig.shape != output_opt.shape:
        logger.error("Shape mismatch: original=%s, optimized=%s",
                    output_orig.shape, output_opt.shape)
        return False
    
    is_close = torch.allclose(output_orig, output_opt, rtol=rtol, atol=atol)
    
    diff = torch.abs(output_orig - output_opt)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    
    rel_diff = diff / (torch.abs(output_orig) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    logger.info("Absolute difference statistics:")
    logger.info("  Max: %.6e", max_diff)
    logger.info("  Mean: %.6e", mean_diff)
    logger.info("  Median: %.6e", median_diff)
    
    logger.info("Relative difference statistics:")
    logger.info("  Max: %.6e", max_rel_diff)
    logger.info("  Mean: %.6e", mean_rel_diff)
    
    if is_close:
        logger.info("Outputs match within tolerance: PASS")
    else:
        logger.error("Outputs do NOT match within tolerance: FAIL")
        
        mismatch_mask = ~torch.isclose(output_orig, output_opt, rtol=rtol, atol=atol)
        num_mismatches = mismatch_mask.sum().item()
        total_elements = output_orig.numel()
        mismatch_pct = (num_mismatches / total_elements) * 100
        
        logger.error("Number of mismatched elements: %d / %d (%.2f%%)",
                    num_mismatches, total_elements, mismatch_pct)
        
        if num_mismatches > 0 and num_mismatches <= 10:
            mismatch_indices = torch.nonzero(mismatch_mask)
            logger.error("First few mismatched positions:")
            for idx in mismatch_indices[:5]:
                idx_tuple = tuple(idx.tolist())
                orig_val = output_orig[idx_tuple].item()
                opt_val = output_opt[idx_tuple].item()
                logger.error("  Position %s: original=%.6e, optimized=%.6e",
                           idx_tuple, orig_val, opt_val)
    
    return is_close

def test_engram_module(layer_id, batch_size, seq_len, device):
    """Test single Engram module correctness"""
    logger.info("=" * 80)
    logger.info("Testing Engram module for layer_id=%d", layer_id)
    logger.info("=" * 80)
    
    set_seed(42)
    
    logger.info("Initializing original Engram module")
    engram_orig = EngramOriginal(layer_id=layer_id)
    logger.info("Original Engram parameters: %d",
               sum(p.numel() for p in engram_orig.parameters()))
    
    logger.info("Initializing optimized Engram module")
    engram_opt = EngramOptimized(layer_id=layer_id)
    logger.info("Optimized Engram parameters: %d",
               sum(p.numel() for p in engram_opt.parameters()))
    
    logger.info("Copying weights from original to optimized")
    copy_weights(engram_orig, engram_opt)
    
    if device.type == 'cuda':
        logger.info("Moving models to GPU")
        engram_orig = engram_orig.to(device)
        engram_opt = engram_opt.to(device)
    
    engram_orig.eval()
    engram_opt.eval()
    logger.info("Models set to evaluation mode")
    
    cfg_orig = EngramConfigOriginal()
    backbone_orig = BackBoneConfigOriginal()
    
    actual_vocab_size = len(engram_orig.hash_mapping.compressed_tokenizer.tokenizer)
    input_ids = create_test_data(batch_size, seq_len, actual_vocab_size, device)
    hidden_states = create_hidden_states(
        batch_size, seq_len, 
        backbone_orig.hidden_size, 
        backbone_orig.hc_mult, 
        device
    )
    
    logger.info("Running forward pass on original Engram")
    start_time = time.time()
    with torch.no_grad():
        output_orig = engram_orig(hidden_states, input_ids)
    orig_time = (time.time() - start_time) * 1000
    logger.info("Original Engram forward pass took %.2fms", orig_time)
    
    logger.info("Running forward pass on optimized Engram")
    start_time = time.time()
    with torch.no_grad():
        output_opt = engram_opt(hidden_states, input_ids)
    opt_time = (time.time() - start_time) * 1000
    logger.info("Optimized Engram forward pass took %.2fms", opt_time)
    
    speedup = orig_time / opt_time if opt_time > 0 else 0
    logger.info("Speedup: %.2fx (original=%.2fms, optimized=%.2fms)",
               speedup, orig_time, opt_time)
    
    logger.info("Comparing outputs")
    passed = compare_outputs(output_orig, output_opt)
    
    if passed:
        logger.info("Test PASSED for layer_id=%d", layer_id)
    else:
        logger.error("Test FAILED for layer_id=%d", layer_id)
    
    logger.info("Logging cache statistics")
    if hasattr(engram_opt.multi_head_embedding, 'log_cache_stats'):
        cache_stats = engram_opt.multi_head_embedding.log_cache_stats()
        logger.info("Cache hit rate: %.2f%%", cache_stats['hit_rate_pct'])
    
    return passed, orig_time, opt_time

def test_transformer_block(layer_id, batch_size, seq_len, device):
    """Test full TransformerBlock correctness"""
    logger.info("=" * 80)
    logger.info("Testing TransformerBlock for layer_id=%d", layer_id)
    logger.info("=" * 80)
    
    set_seed(42)
    
    logger.info("Initializing original TransformerBlock")
    block_orig = TransformerBlockOriginal(layer_id=layer_id)
    
    logger.info("Initializing optimized TransformerBlock")
    block_opt = TransformerBlockOptimized(layer_id=layer_id)
    
    logger.info("Copying weights")
    copy_weights(block_orig, block_opt)
    
    if device.type == 'cuda':
        logger.info("Moving blocks to GPU")
        block_orig = block_orig.to(device)
        block_opt = block_opt.to(device)
    
    block_orig.eval()
    block_opt.eval()
    
    backbone_orig = BackBoneConfigOriginal()
    
    actual_vocab_size = len(block_orig.engram.hash_mapping.compressed_tokenizer.tokenizer) if block_orig.engram else backbone_orig.vocab_size
    input_ids = create_test_data(batch_size, seq_len, actual_vocab_size, device)
    hidden_states = create_hidden_states(
        batch_size, seq_len,
        backbone_orig.hidden_size,
        backbone_orig.hc_mult,
        device
    )
    
    logger.info("Running forward pass on original TransformerBlock")
    start_time = time.time()
    with torch.no_grad():
        output_orig = block_orig(input_ids, hidden_states)
    orig_time = (time.time() - start_time) * 1000
    logger.info("Original TransformerBlock forward pass took %.2fms", orig_time)
    
    logger.info("Running forward pass on optimized TransformerBlock")
    start_time = time.time()
    with torch.no_grad():
        output_opt = block_opt(input_ids, hidden_states)
    opt_time = (time.time() - start_time) * 1000
    logger.info("Optimized TransformerBlock forward pass took %.2fms", opt_time)
    
    speedup = orig_time / opt_time if opt_time > 0 else 0
    logger.info("Speedup: %.2fx", speedup)
    
    passed = compare_outputs(output_orig, output_opt)
    
    if passed:
        logger.info("Test PASSED for TransformerBlock layer_id=%d", layer_id)
    else:
        logger.error("Test FAILED for TransformerBlock layer_id=%d", layer_id)
    
    return passed, orig_time, opt_time

def run_all_tests():
    """Run all correctness tests"""
    logger.info("#" * 80)
    logger.info("STARTING ENGRAM CORRECTNESS TESTS")
    logger.info("#" * 80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA memory allocated: %.2f MB", 
                   torch.cuda.memory_allocated() / 1024**2)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (tests will be slower)")
    
    cfg = EngramConfigOriginal()
    test_batch_size = 4
    test_seq_len = 128
    
    logger.info("Test configuration:")
    logger.info("  Batch size: %d", test_batch_size)
    logger.info("  Sequence length: %d", test_seq_len)
    logger.info("  Engram layers: %s", cfg.layer_ids)
    
    all_passed = True
    results = []
    
    for layer_id in cfg.layer_ids:
        logger.info("")
        logger.info("Testing layer %d", layer_id)
        
        try:
            passed_engram, orig_time_engram, opt_time_engram = test_engram_module(
                layer_id, test_batch_size, test_seq_len, device
            )
            
            passed_block, orig_time_block, opt_time_block = test_transformer_block(
                layer_id, test_batch_size, test_seq_len, device
            )
            
            layer_passed = passed_engram and passed_block
            all_passed = all_passed and layer_passed
            
            results.append({
                'layer_id': layer_id,
                'engram_passed': passed_engram,
                'block_passed': passed_block,
                'engram_speedup': orig_time_engram / opt_time_engram if opt_time_engram > 0 else 0,
                'block_speedup': orig_time_block / opt_time_block if opt_time_block > 0 else 0,
            })
            
        except Exception as e:
            logger.error("Exception during test for layer %d: %s", layer_id, e)
            logger.exception("Full traceback:")
            all_passed = False
            results.append({
                'layer_id': layer_id,
                'engram_passed': False,
                'block_passed': False,
                'error': str(e)
            })
    
    logger.info("")
    logger.info("#" * 80)
    logger.info("TEST SUMMARY")
    logger.info("#" * 80)
    
    for result in results:
        layer_id = result['layer_id']
        if 'error' in result:
            logger.error("Layer %d: ERROR - %s", layer_id, result['error'])
        else:
            engram_status = "PASS" if result['engram_passed'] else "FAIL"
            block_status = "PASS" if result['block_passed'] else "FAIL"
            logger.info("Layer %d: Engram=%s (%.2fx), Block=%s (%.2fx)",
                       layer_id, engram_status, result['engram_speedup'],
                       block_status, result['block_speedup'])
    
    logger.info("")
    if all_passed:
        logger.info("ALL TESTS PASSED")
        logger.info("Optimized implementation is numerically correct")
        
        avg_engram_speedup = np.mean([r['engram_speedup'] for r in results if 'engram_speedup' in r])
        avg_block_speedup = np.mean([r['block_speedup'] for r in results if 'block_speedup' in r])
        logger.info("Average Engram speedup: %.2fx", avg_engram_speedup)
        logger.info("Average Block speedup: %.2fx", avg_block_speedup)
    else:
        logger.error("SOME TESTS FAILED")
        logger.error("Optimized implementation has correctness issues")
    
    logger.info("#" * 80)
    
    return all_passed

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)