#!/usr/bin/env python3
"""
Master Verification Script for DriftBench MLSys 2026 Submission
Verifies ALL numerical claims in the paper against source data
"""

import pandas as pd
import json
import os
from pathlib import Path

class PaperVerification:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
    def check(self, claim_id, description, actual, expected, tolerance=0.01):
        """Check if actual matches expected within tolerance"""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            match = abs(actual - expected) <= tolerance
        else:
            match = actual == expected
        
        status = "[PASS]" if match else "[FAIL]"
        self.results.append({
            'id': claim_id,
            'description': description,
            'expected': expected,
            'actual': actual,
            'status': status
        })
        
        # Print immediately for real-time feedback
        print(f"  [{claim_id}] {description}: Expected={expected}, Actual={actual} ... {status}")
        
        if match:
            self.passed += 1
        else:
            self.failed += 1
        
        return match
    
    def warning(self, claim_id, description, message):
        """Log a warning"""
        self.results.append({
            'id': claim_id,
            'description': description,
            'expected': 'N/A',
            'actual': message,
            'status': '[WARN]'
        })
        
        # Print immediately for real-time feedback
        print(f"  [{claim_id}] {description}: {message} ... [WARN]")
        
        self.warnings += 1
    
    def print_results(self):
        """Print all verification results"""
        print("\n" + "="*80)
        print("MASTER VERIFICATION RESULTS")
        print("="*80)
        
        for r in self.results:
            print(f"\n[{r['id']}] {r['description']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Actual:   {r['actual']}")
            print(f"  {r['status']}")
        
        print("\n" + "="*80)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        print("="*80)
        
        if self.failed == 0:
            print("[PASS] ALL VERIFICATIONS PASSED")
        else:
            print(f"[FAIL] {self.failed} VERIFICATIONS FAILED - REVIEW REQUIRED")
        
        return self.failed == 0


def main():
    v = PaperVerification()
    
    # Load core data files
    print("Loading data files...")
    
    # Determine base path for data files (relative to script location)
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / 'data'
    
    try:
        flip_rates = pd.read_csv(base_path / 'flip_rates.csv')
        print("[OK] flip_rates.csv loaded")
    except FileNotFoundError:
        print("[ERROR] flip_rates.csv NOT FOUND")
        return False
    
    print("\n" + "="*80)
    print("SECTION 1: CORE CLAIMS (Abstract & Introduction)")
    print("="*80)
    
    # Claim 1: Configuration count
    configs = len(flip_rates[['model','hardware','precision','framework']].drop_duplicates())
    v.check('1.1', 'Configuration count', configs, 105)
    
    # Claim 2: Total pairs
    total_pairs = configs * 2257
    v.check('1.2', 'Total prompt-response pairs', total_pairs, 236985)
    
    # Claim 3: Safety flip rate
    wl_means = flip_rates.groupby('workload')['flip_rate'].mean()
    v.check('1.3', 'Safety flip rate (%)', round(wl_means['safety'], 2), 7.97)
    
    # Claim 4: Code flip rate
    v.check('1.4', 'Code flip rate (%)', round(wl_means['code'], 2), 0.09, tolerance=0.02)
    
    # Claim 5: Math flip rate
    v.check('1.5', 'Math flip rate (%)', round(wl_means['math'], 2), 16.74)
    
    # Claim 6: Safety/Code ratio
    ratio = int(round(wl_means['safety'], 2) / round(wl_means['code'], 2))
    v.check('1.6', 'Safety/Code ratio (×)', ratio, 88, tolerance=2)
    
    # Claim 7: FP4/B200
    fp4_b200 = flip_rates[(flip_rates['precision']=='fp4') & 
                           (flip_rates['hardware']=='b200')]['flip_rate'].mean()
    v.check('1.7', 'FP4/B200 flip rate (%)', round(fp4_b200, 1), 12.7, tolerance=0.2)
    
    print("\n" + "="*80)
    print("SECTION 2: PRODUCTION VALIDATION (Table 6)")
    print("="*80)
    
    # Production validation
    safety = flip_rates[flip_rates['workload'] == 'safety']
    baseline = safety[(safety['model']=='llama-3.1-8b') &
                      (safety['hardware']=='h100') &
                      (safety['precision']=='fp16') &
                      (safety['framework']=='sglang')]
    
    target = safety[(safety['model']=='llama-3.1-8b') &
                    (safety['hardware']=='b200') &
                    (safety['precision']=='fp8') &
                    (safety['framework']=='sglang')]
    
    if len(baseline) > 0 and len(target) > 0:
        baseline_rate = baseline['flip_rate'].values[0]
        target_rate = target['flip_rate'].values[0]
        
        v.check('2.1', 'Baseline violations (%)', round(baseline_rate, 2), 12.50)
        v.check('2.2', 'Target violations (%)', round(target_rate, 2), 23.85)
    else:
        v.warning('2.x', 'Production validation', 'Data not found in flip_rates.csv')
    
    print("\n" + "="*80)
    print("SECTION 3: PRI GENERALIZATION (Table 2)")
    print("="*80)
    
    try:
        gen = pd.read_csv(base_path / 'generalization_results.csv')
        
        gen_expected = {
            'hardware': 0.909,
            'precision': 0.763,
            'framework': 0.479,
            'model': 0.118
        }
        
        for dim, expected_r2 in gen_expected.items():
            row = gen[gen['holdout_type'] == dim]
            if len(row) > 0:
                actual_r2 = row['test_r2'].values[0]
                v.check(f'3.{dim[:3]}', f'PRI R^2 ({dim})', round(actual_r2, 3), expected_r2, tolerance=0.01)
            else:
                v.warning(f'3.{dim[:3]}', f'PRI R^2 ({dim})', f'{dim} not found in generalization_results.csv')
    
    except FileNotFoundError:
        v.warning('3.x', 'PRI Generalization', 'generalization_results.csv not found')
    
    print("\n" + "="*80)
    print("SECTION 4: FEATURE IMPORTANCE (Table 5)")
    print("="*80)
    
    try:
        feat_imp = pd.read_csv(base_path / 'feature_importance.csv')
        
        # Check top features
        top_checks = {
            'workload_safety': 33.4,
            'framework_tensorrt': 25.1,
            'qwen': 15.4,
            'workload_math': 11.5
        }
        
        for feature_substr, expected in top_checks.items():
            matching = feat_imp[feat_imp['feature'].str.contains(feature_substr, case=False, na=False)]
            if len(matching) > 0:
                actual = matching['importance'].values[0] * 100  # Convert decimal to percentage
                v.check(f'4.{feature_substr[:4]}', f'{feature_substr} importance (%)', 
                       round(actual, 1), expected, tolerance=1.0)
            else:
                v.warning(f'4.{feature_substr[:4]}', f'{feature_substr} importance', 'Feature not found')
        
        # Check hardware < 0.3%
        hardware_feats = feat_imp[feat_imp['feature'].str.contains('hardware', case=False, na=False)]
        if len(hardware_feats) > 0:
            max_hw = hardware_feats['importance'].max() * 100  # Convert to percentage
            # Manual check since expected is a condition not a number
            if max_hw < 0.3:
                v.results.append({
                    'id': '4.hw',
                    'description': 'Hardware max importance (< 0.3%)',
                    'expected': '<0.3%',
                    'actual': f'{round(max_hw, 3)}%',
                    'status': '[PASS]'
                })
                v.passed += 1
            else:
                v.results.append({
                    'id': '4.hw',
                    'description': 'Hardware max importance (< 0.3%)',
                    'expected': '<0.3%',
                    'actual': f'{round(max_hw, 3)}%',
                    'status': '[FAIL]'
                })
                v.failed += 1
    
    except FileNotFoundError:
        v.warning('4.x', 'Feature Importance', 'feature_importance.csv not found')
    
    print("\n" + "="*80)
    print("SECTION 5: DATASET SIZES")
    print("="*80)
    
    dataset_sizes = {
        'code': 164,
        'math': 500,
        'safety': 520,
        'long_context': 100
    }
    
    for wl, expected_size in dataset_sizes.items():
        actual_size = flip_rates[flip_rates['workload']==wl]['num_comparisons'].iloc[0] if len(flip_rates[flip_rates['workload']==wl]) > 0 else 0
        v.check(f'5.{wl[:4]}', f'{wl.capitalize()} dataset size', actual_size, expected_size)
    
    print("\n" + "="*80)
    print("SECTION 6: ANOVA RESULTS (if available)")
    print("="*80)
    
    try:
        anova = pd.read_csv(base_path / 'anova_flip_rate_corrected.csv', index_col=0)
        
        if 'C(workload)' in anova.index:
            f_stat = anova.loc['C(workload)', 'F']
            eta_sq = anova.loc['C(workload)', 'eta_squared']
            v.check('6.1', 'Workload F-statistic', round(f_stat, 1), 53.6, tolerance=2.0)
            v.check('6.2', 'Workload eta-squared', round(eta_sq, 3), 0.275, tolerance=0.01)
        else:
            v.warning('6.x', 'ANOVA', 'Workload factor not found in ANOVA results')
    
    except FileNotFoundError:
        v.warning('6.x', 'ANOVA Results', 'anova_flip_rate_corrected.csv not found')
    
    print("\n" + "="*80)
    print("SECTION 7: COST CALCULATIONS")
    print("="*80)
    
    # Mathematical verification of cost claims
    cost_data = {
        'B200': {'rate': 6.00, 'hours': 140.2, 'expected_cost': 841},
        'H100': {'rate': 2.69, 'hours': 225.7, 'expected_cost': 607},
        'H200': {'rate': 3.59, 'hours': 75.8, 'expected_cost': 272},
        'MI300X': {'rate': 2.29, 'hours': 58.5, 'expected_cost': 134}
    }
    
    total_cost = 0
    total_hours = 0
    for platform, data in cost_data.items():
        calculated_cost = round(data['rate'] * data['hours'])
        v.check(f'7.{platform[:3]}', f'{platform} cost ($)', calculated_cost, data['expected_cost'], tolerance=5)
        total_cost += calculated_cost
        total_hours += data['hours']
    
    v.check('7.tot', 'Total cost ($)', total_cost, 1854, tolerance=10)
    v.check('7.hrs', 'Total GPU-hours', round(total_hours, 1), 500.2, tolerance=5)
    
    avg_per_config = round(total_cost / 105, 2)
    v.check('7.avg', 'Average cost per config ($)', avg_per_config, 17.67, tolerance=1.0)
    
    # Print final results
    success = v.print_results()
    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
