#!/usr/bin/env python3
"""
Days 4-5 - Phase 3.3.1: Prepare PRI Dataset
Feature engineering for Portability Risk Index model.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Base directory: where this script lives
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

def load_flip_rates():
    """Load flip rate metrics from Phase 3.1.3"""
    path = DATA_DIR / 'flip_rates.csv'
    
    if not path.exists():
        print(f"❌ Flip rates not found at {path}")
        print("Please ensure data/flip_rates.csv exists")
        sys.exit(1)
    
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} flip rate measurements")
    return df

def create_features(df):
    """
    Phase 3.3.1: Feature engineering for PRI
    Create one-hot encoded features for categorical variables
    """
    print("\n" + "="*80)
    print("Phase 3.3.1: Feature Engineering for PRI")
    print("="*80)
    
    # Parse experiment_id if not already split
    if 'model' not in df.columns or 'hardware' not in df.columns:
        print("Parsing experiment IDs...")
        parts = df['experiment_id'].str.split('--', expand=True)
        df['model'] = parts[0]
        df['hardware'] = parts[1]
        df['precision'] = parts[2]
        df['framework'] = parts[3]
    
    print("\nFeature dimensions:")
    print(f"  Models: {df['model'].nunique()} unique values")
    print(f"  Hardware: {df['hardware'].nunique()} unique values")
    print(f"  Precision: {df['precision'].nunique()} unique values")
    print(f"  Frameworks: {df['framework'].nunique()} unique values")
    print(f"  Workloads: {df['workload'].nunique()} unique values")
    
    # One-hot encoding
    print("\nCreating one-hot encoded features...")
    features = pd.get_dummies(
        df[['model', 'hardware', 'precision', 'framework', 'workload']],
        columns=['model', 'hardware', 'precision', 'framework', 'workload'],
        prefix=['model', 'hardware', 'precision', 'framework', 'workload']
    )
    
    print(f"✓ Created {len(features.columns)} features")
    print(f"  Feature names: {list(features.columns)[:10]}... (showing first 10)")
    
    return features

def create_enhanced_features(df):
    """
    Enhanced feature engineering with numerical representations
    Enables better generalization to unseen configurations
    """
    print("\n📊 Creating enhanced features...")
    
    enhanced = pd.DataFrame()
    
    # Hardware features (compute capability, memory)
    hardware_map = {
        'h100': {'compute_capability': 9.0, 'memory_gb': 80, 'tensor_core_gen': 4},
        'h200': {'compute_capability': 9.0, 'memory_gb': 141, 'tensor_core_gen': 4},
        'b200': {'compute_capability': 10.0, 'memory_gb': 192, 'tensor_core_gen': 5},
        'mi300x': {'compute_capability': 9.4, 'memory_gb': 192, 'tensor_core_gen': 0}  # AMD
    }
    
    # Precision features (mantissa bits, exponent bits)
    precision_map = {
        'fp16': {'mantissa_bits': 10, 'exponent_bits': 5, 'total_bits': 16},
        'bf16': {'mantissa_bits': 7, 'exponent_bits': 8, 'total_bits': 16},
        'fp8': {'mantissa_bits': 3, 'exponent_bits': 4, 'total_bits': 8},
        'fp4': {'mantissa_bits': 2, 'exponent_bits': 2, 'total_bits': 4}
    }
    
    # Model features (parameters, layers)
    model_map = {
        'llama-3.1-70b': {'params_b': 70, 'num_layers': 80, 'is_moe': 0},
        'llama-3.1-8b': {'params_b': 8, 'num_layers': 32, 'is_moe': 0},
        'mistral-7b': {'params_b': 7, 'num_layers': 32, 'is_moe': 0},
        'mixtral-8x7b': {'params_b': 47, 'num_layers': 32, 'is_moe': 1},
        'qwen-7b': {'params_b': 7, 'num_layers': 28, 'is_moe': 0}
    }
    
    # Apply mappings
    for col, mapping in [
        ('hardware', hardware_map),
        ('precision', precision_map),
        ('model', model_map)
    ]:
        if col in df.columns:
            for key in next(iter(mapping.values())).keys():
                enhanced[f'{col}_{key}'] = df[col].map(lambda x: mapping.get(x, {}).get(key, 0))
    
    print(f"✓ Created {len(enhanced.columns)} enhanced features")
    
    return enhanced

def prepare_target(df):
    """Prepare target variable (flip_rate)"""
    target = df['flip_rate'].copy()
    
    print("\nTarget variable (flip_rate) statistics:")
    print(f"  Mean: {target.mean():.2f}%")
    print(f"  Median: {target.median():.2f}%")
    print(f"  Std: {target.std():.2f}%")
    print(f"  Min: {target.min():.2f}%")
    print(f"  Max: {target.max():.2f}%")
    print(f"  Non-zero: {(target > 0).sum()} / {len(target)} ({(target > 0).sum()/len(target)*100:.1f}%)")
    
    return target

def save_dataset(features, target, enhanced_features=None):
    """Save prepared dataset for training"""
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Save standard features
    features_path = os.path.join(output_dir, 'pri_features.csv')
    features.to_csv(features_path, index=False)
    print(f"\n✓ Saved features to {features_path}")
    
    # Save enhanced features if available
    if enhanced_features is not None:
        enhanced_path = os.path.join(output_dir, 'pri_features_enhanced.csv')
        enhanced_features.to_csv(enhanced_path, index=False)
        print(f"✓ Saved enhanced features to {enhanced_path}")
    
    # Save target
    target_path = os.path.join(output_dir, 'pri_target.csv')
    target.to_csv(target_path, index=False, header=['flip_rate'])
    print(f"✓ Saved target to {target_path}")
    
    # Save feature names for later use
    feature_names_path = os.path.join(output_dir, 'pri_feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(features.columns))
    print(f"✓ Saved feature names to {feature_names_path}")

def main():
    print("="*80)
    print("DAYS 4-5: PRI DATASET PREPARATION")
    print("Phase 3.3.1: Feature Engineering")
    print("="*80)
    
    # Load data
    df = load_flip_rates()
    
    # Create features
    features = create_features(df)
    enhanced_features = create_enhanced_features(df)
    
    # Prepare target
    target = prepare_target(df)
    
    # Save dataset
    save_dataset(features, target, enhanced_features)
    
    # Summary
    print("\n" + "="*80)
    print("PRI DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"✓ Features: {features.shape}")
    print(f"✓ Enhanced features: {enhanced_features.shape}")
    print(f"✓ Target: {target.shape}")
    print("\n📁 Output files:")
    print("  - pri_features.csv")
    print("  - pri_features_enhanced.csv")
    print("  - pri_target.csv")
    print("  - pri_feature_names.txt")
    
    print("\n📋 NEXT STEP:")
    print("  → python3 tools/train_pri.py")
    
    print("="*80)

if __name__ == '__main__':
    main()
