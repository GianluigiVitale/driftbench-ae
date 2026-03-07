"""
Re-run generalization validation with the enhanced PRI model.
Tests model performance on held-out dimensions (hardware, precision, framework, model).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json

# Base directory: where this script lives
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

print("="*80)
print("GENERALIZATION VALIDATION - ENHANCED PRI MODEL")
print("="*80)
print()

# Load data
base_features = pd.read_csv(RESULTS_DIR / 'pri_features.csv')
targets = pd.read_csv(RESULTS_DIR / 'pri_target.csv')['flip_rate'].values
flip_rates = pd.read_csv(DATA_DIR / 'flip_rates.csv')

print(f"✓ Loaded {len(base_features)} configurations")
print()

# Add interaction features (same as training)
def add_interactions(base_df):
    """Add interaction features matching training"""
    features = base_df.copy()
    
    prec_cols = [c for c in base_df.columns if c.startswith('precision_')]
    fw_cols = [c for c in base_df.columns if c.startswith('framework_')]
    wl_cols = [c for c in base_df.columns if c.startswith('workload_')]
    hw_cols = [c for c in base_df.columns if c.startswith('hardware_')]
    
    # Precision × workload
    for prec in prec_cols:
        for wl in wl_cols:
            features[f'{prec}_{wl}'] = base_df[prec] * base_df[wl]
    
    # Framework × workload
    for fw in fw_cols:
        for wl in wl_cols:
            features[f'{fw}_{wl}'] = base_df[fw] * base_df[wl]
    
    # Hardware × precision
    for hw in hw_cols:
        for prec in prec_cols:
            features[f'{hw}_{prec}'] = base_df[hw] * base_df[prec]
    
    return features

features = add_interactions(base_features)
print(f"✓ Created {features.shape[1]} features (with interactions)")
print()

# Define holdout scenarios
holdout_scenarios = [
    {
        'name': 'Hardware',
        'train_condition': lambda df: df['hardware'] != 'b200',
        'test_condition': lambda df: df['hardware'] == 'b200',
        'description': 'Train on H100/H200/MI300X, test on B200'
    },
    {
        'name': 'Precision',
        'train_condition': lambda df: df['precision'] != 'fp4',
        'test_condition': lambda df: df['precision'] == 'fp4',
        'description': 'Train on FP16/FP8, test on FP4'
    },
    {
        'name': 'Framework',
        'train_condition': lambda df: df['framework'] != 'sglang',
        'test_condition': lambda df: df['framework'] == 'sglang',
        'description': 'Train on vLLM/TensorRT, test on SGLang'
    },
    {
        'name': 'Model',
        'train_condition': lambda df: df['model'] != 'qwen-7b',
        'test_condition': lambda df: df['model'] == 'qwen-7b',
        'description': 'Train on Llama/Mistral/Mixtral, test on Qwen'
    }
]

results = []

print("="*80)
print("TESTING GENERALIZATION TO HELD-OUT DIMENSIONS")
print("="*80)
print()

for scenario in holdout_scenarios:
    print(f"{scenario['name']} Holdout Test")
    print(f"  {scenario['description']}")
    
    # Split data
    train_mask = scenario['train_condition'](flip_rates).values
    test_mask = scenario['test_condition'](flip_rates).values
    
    X_train = features[train_mask]
    y_train = targets[train_mask]
    X_test = features[test_mask]
    y_test = targets[test_mask]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model (same hyperparams as best model)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_samples_split=3, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.2f}pp")
    print()
    
    results.append({
        'holdout_type': scenario['name'].lower(),
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
        'test_mae': round(test_mae, 2),
        'n_train': len(X_train),
        'n_test': len(X_test)
    })

# Save results
results_df = pd.DataFrame(results)
output_path = RESULTS_DIR / 'generalization_results.csv'
results_df.to_csv(output_path, index=False)

print("="*80)
print("GENERALIZATION SUMMARY")
print("="*80)
print()
print(f"{'Dimension':<12} {'Train R²':>10} {'Test R²':>10} {'Test MAE':>10}")
print("-" * 50)
for _, row in results_df.iterrows():
    print(f"{row['holdout_type'].capitalize():<12} {row['train_r2']:>10.4f} {row['test_r2']:>10.4f} {row['test_mae']:>9.2f}pp")

print()
print(f"Mean Test R²: {results_df['test_r2'].mean():.4f}")
print()
print(f"✓ Results saved to: {output_path}")
print()

# Also save as JSON for figure generation
json_path = RESULTS_DIR / 'generalization_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ JSON saved to: {json_path}")

print()
print("="*80)
print("✅ GENERALIZATION VALIDATION COMPLETE")
print("="*80)
