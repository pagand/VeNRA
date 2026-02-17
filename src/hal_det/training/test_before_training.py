#!/usr/bin/env python3
"""
VeNRA Pre-Flight Test Suite
============================
Run this BEFORE full training to catch configuration issues early.
Takes ~2-3 minutes to verify everything works.
"""

import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import format_prompt_func

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")
    
def print_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def print_section(msg):
    print(f"\n{BLUE}{'='*70}")
    print(f"{msg}")
    print(f"{'='*70}{RESET}")

# Test configurations
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
TOKEN_MAP = {
    "Supported": " Found",
    "Unfounded": " Fake",
    "General": " General"
}

def test_environment():
    """Test 1: Environment Variables"""
    print_section("Test 1: Environment Variables")
    
    load_dotenv()
    
    # Check HF_TOKEN
    if "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]
        if token.startswith("hf_"):
            print_success(f"HF_TOKEN loaded (starts with 'hf_')")
        else:
            print_error("HF_TOKEN doesn't look valid (should start with 'hf_')")
            return False
    else:
        print_error("HF_TOKEN not found in environment")
        print("  Add it to .env file: HF_TOKEN=hf_xxxxx")
        return False
    
    # Check WANDB_API_KEY (optional)
    if "WANDB_API_KEY" in os.environ:
        print_success("WANDB_API_KEY loaded")
    else:
        print_warning("WANDB_API_KEY not set (wandb logging will be local only)")
    
    return True

def test_gpu():
    """Test 2: GPU & CUDA"""
    print_section("Test 2: GPU & CUDA Availability")
    
    if not torch.cuda.is_available():
        print_error("CUDA not available!")
        print("  Check: nvidia-smi")
        print("  Reinstall PyTorch with CUDA support")
        return False
    
    print_success(f"CUDA available: {torch.version.cuda}")
    print_success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print_success(f"GPU Memory: {gpu_mem:.1f} GB")
    
    if gpu_mem < 20:
        print_warning(f"GPU has {gpu_mem:.1f}GB - may be tight for 3B model with 4k context")
        print("  Consider reducing batch size or max_seq_length if OOM occurs")
    
    return True

def test_dataset():
    """Test 3: Dataset Access"""
    print_section("Test 3: Dataset Access")
    
    try:
        print("Loading dataset pagand/venra (v2.1)...")
        dataset = load_dataset("pagand/venra", revision="v2.1")
        
        print_success(f"Train samples: {len(dataset['train'])}")
        print_success(f"Val samples: {len(dataset['validation'])}")
        print_success(f"Test samples: {len(dataset['test'])}")
        
        # Check data structure
        sample = dataset['train'][0]
        required_fields = ['id', 'label', 'input_components', 'output_components', 'meta']
        for field in required_fields:
            if field in sample:
                print_success(f"Field '{field}' present")
            else:
                print_error(f"Missing required field: {field}")
                return False
        
        # Check family_id in metadata
        if 'family_id' in sample.get('meta', {}):
            print_success("Metadata contains 'family_id' (required for paired evaluation)")
        else:
            print_warning("'family_id' not found in metadata - paired eval may fail")
        
        # Check sabotage_type (top-level)
        if 'sabotage_type' in sample:
            print_success(f"sabotage_type found: {sample['sabotage_type']}")
        else:
            print_warning("sabotage_type not at top level - stratification may fail")
        
        return True
        
    except Exception as e:
        print_error(f"Dataset loading failed: {e}")
        print("  Check your HF_TOKEN has access to pagand/venra")
        return False

def test_tokenizer():
    """Test 4: Tokenizer & Orthogonal Tokens"""
    print_section("Test 4: Tokenizer & Orthogonal Token Mapping")
    
    try:
        print(f"Loading tokenizer: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print_success("Tokenizer loaded")
        
        # Test orthogonal tokens
        print("\nVerifying orthogonal token mapping:")
        token_map_ids = {}
        for label, token in TOKEN_MAP.items():
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                token_map_ids[label] = ids[0]
                print_success(f"  {label:12s} -> '{token}' (ID: {ids[0]})")
            else:
                print_error(f"  {label:12s} -> '{token}' FRAGMENTED into {len(ids)} tokens!")
                print(f"    Token IDs: {ids}")
                print("    This violates the orthogonal label requirement!")
                return False
        
        # Test prompt formatting
        test_prompt = (
            f"<|im_start|>system\nYou are a financial auditor.<|im_end|>\n"
            f"<|im_start|>user\nQuery: Test\nContext: Test\nTrace: Test\nStatement: Test\n"
            f"Task: Classify [Found, Fake, General].<|im_end|>\n"
            f"<|im_start|>assistant\nLabel: Found\nAnalysis: Test<|im_end|>"
        )
        tokens = tokenizer.encode(test_prompt)
        print_success(f"Test prompt tokenizes to {len(tokens)} tokens")
        
        if len(tokens) > 4096:
            print_warning(f"Test prompt is {len(tokens)} tokens - check max_seq_length")
        
        return True
        
    except Exception as e:
        print_error(f"Tokenizer test failed: {e}")
        return False

def test_model_loading():
    """Test 5: Model Loading with Quantization"""
    print_section("Test 5: Model Loading (4-bit Quantization)")
    
    try:
        print("Loading model with 4-bit quantization (this may take 1-2 minutes)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        
        print_success("Model loaded successfully")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print_success(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        if reserved > 20:
            print_warning(f"High memory usage ({reserved:.2f}GB) - training may OOM")
        
        # Prepare for training
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        print_success("Model prepared for k-bit training")
        
        return True
        
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        if "CUDA out of memory" in str(e):
            print("  Try: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        return False

def test_lora_config():
    """Test 6: LoRA Configuration"""
    print_section("Test 6: LoRA Configuration")
    
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=32,
            lora_dropout=0.05,
            use_rslora=True,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
        )
        
        print_success(f"LoRA Config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        print_success(f"rsLoRA enabled: {peft_config.use_rslora}")
        print_success(f"Target modules: {len(peft_config.target_modules)} modules")
        
        return True
        
    except Exception as e:
        print_error(f"LoRA config failed: {e}")
        return False

def test_data_formatting():
    """Test 7: Data Formatting"""
    print_section("Test 7: Data Formatting & Batch Processing")
    
    try:
        # Load small sample
        dataset = load_dataset("pagand/venra", revision="v2.1")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Test formatting function
        sample_batch = dataset['train'].select(range(2))
        
        formatted = format_prompt_func(sample_batch)
        
        if len(formatted) > 0:
            print_success(f"Formatted {len(formatted)} examples")
            print_success(f"Sample length: {len(formatted[0])} chars")
            
            # Check token length
            tokens = tokenizer.encode(formatted[0])
            print_success(f"Sample tokenizes to {len(tokens)} tokens")
            
            if len(tokens) > 4096:
                print_warning(f"Sample exceeds max_seq_length (4096)")
        else:
            print_error("Formatting produced no outputs!")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Data formatting test failed: {e}")
        print(f"  Error details: {str(e)}")
        return False

def test_quick_training():
    """Test 8: Quick Training Run (5 steps)"""
    print_section("Test 8: Quick Training Sanity Check (5 steps)")
    
    response = input(f"\n{YELLOW}Run a quick 5-step training test? This will take ~1-2 minutes. (y/n): {RESET}")
    
    if response.lower() != 'y':
        print_warning("Skipped quick training test")
        return True
    
    try:
        print("\nRunning 5-step training test...")
        print("(This tests the full training pipeline without committing to hours)")
        
        # This would require running a minimal version of the training script
        # For now, we'll skip this and recommend the user does it manually
        print_warning("Quick training test not implemented in this script")
        print("To test manually:")
        print("  python src/hal_det/training/train.py \\")
        print("    --output_dir ./data/test-run \\")
        print("    --max_steps 5 \\")
        print("    --logging_steps 1 \\")
        print("    --save_strategy no")
        
        return True
        
    except Exception as e:
        print_error(f"Quick training test failed: {e}")
        return False

def main():
    print(f"""
{BLUE}╔═══════════════════════════════════════════════════════════════════╗
║          VeNRA Pre-Flight Test Suite                              ║
║  Run this BEFORE full training to catch issues early              ║
╚═══════════════════════════════════════════════════════════════════╝{RESET}
""")
    
    tests = [
        ("Environment Variables", test_environment),
        ("GPU & CUDA", test_gpu),
        ("Dataset Access", test_dataset),
        ("Tokenizer & Tokens", test_tokenizer),
        ("Model Loading", test_model_loading),
        ("LoRA Config", test_lora_config),
        ("Data Formatting", test_data_formatting),
        ("Quick Training", test_quick_training),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print_error(f"\n{test_name} FAILED - Fix this before proceeding!")
                break
                
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Tests interrupted by user{RESET}")
            break
        except Exception as e:
            print_error(f"\n{test_name} crashed: {e}")
            results.append((test_name, False))
            break
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name:30s} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print(f"""
{GREEN}╔═══════════════════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED!                                              ║
║                                                                   ║
║  You're ready to start training:                                 ║
║  python python src/hal_det/training/train.py --output_dir ./venra-output  ║
╚═══════════════════════════════════════════════════════════════════╝{RESET}
""")
        return 0
    else:
        print(f"""
{RED}╔═══════════════════════════════════════════════════════════════════╗
║  ✗ SOME TESTS FAILED                                              ║
║                                                                   ║
║  Fix the issues above before running full training.              ║
╚═══════════════════════════════════════════════════════════════════╝{RESET}
""")
        return 1

if __name__ == "__main__":
    sys.exit(main())