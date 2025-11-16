import sys
sys.path.append('src')

from tuning.synthetic_expert import SyntheticExpert, PerceptualMetrics
from models.nn_v0 import DatasetLoader
from inference.predictor import HapticPredictor
import json
from pathlib import Path

def test_synthetic_expert():
    """
    Test synthetic expert tuning on dataset samples.
    Generates Phase 2 training data automatically.
    """
    
    print("\n" + "="*60)
    print("SYNTHETIC EXPERT TUNING - Phase 2 Data Generation")
    print("="*60 + "\n")
    
    # Initialize
    predictor = HapticPredictor("models/checkpoints/nn_v0_best.pt")
    expert = SyntheticExpert(expert_id="synthetic_001")
    dataset = DatasetLoader("data/processed/phase1_test.h5")
    
    print(expert.get_tuning_style_description())
    
    # Generate tunings
    n_samples = 500
    print(f"\nGenerating {n_samples} synthetic expert tunings...")
    
    import numpy as np
    indices = np.random.choice(dataset.n_samples, n_samples, replace=False)
    
    tunings = []
    
    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_samples}", end='\r')
        
        # Extract features
        feature_vec = extract_feature_vec(dataset, idx)
        
        # Get baseline prediction
        baseline = predictor.predict(feature_vec)
        
        # Apply expert tuning
        gold = expert.tune(baseline, feature_vec)
        
        # Compute delta
        delta = compute_delta(baseline, gold)
        
        # Save tuning
        tuning = {
            'scenario_id': f"scenario_{idx}",
            'expert_id': expert.expert_id,
            'feature_vec': feature_vec,
            'baseline_cues': baseline,
            'gold_cues': gold,
            'delta_cues': delta
        }
        tunings.append(tuning)
    
    print(f"  Progress: {n_samples}/{n_samples} ✓")
    
    # Analyze tuning patterns
    print("\nAnalyzing tuning patterns...")
    analysis = PerceptualMetrics.analyze_tuning_patterns(tunings)
    
    print(f"\nTuning Statistics:")
    print(f"  Avg Impact Rise Change: {analysis['avg_impact_rise_change']:.1f}%")
    print(f"  Avg Weight Change: {analysis['avg_weight_change']:.1f}%")
    
    # Save to JSON
    output_dir = Path("data/phase2_tunings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "synthetic_expert_001_tunings.json"
    with open(output_file, 'w') as f:
        json.dump(tunings, f, indent=2)
    
    print(f"\n✓ Saved {len(tunings)} tunings to: {output_file}")
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE TUNINGS")
    print("="*60)
    
    for i in range(min(3, len(tunings))):
        t = tunings[i]
        print(f"\nExample {i+1}:")
        print(f"  Force: {t['feature_vec']['normal_force_N']:.2f}N")
        print(f"  Baseline impact.rise: {t['baseline_cues']['impact']['rise_ms']:.2f}ms")
        print(f"  Gold impact.rise: {t['gold_cues']['impact']['rise_ms']:.2f}ms")
        print(f"  Delta: {t['delta_cues']['impact']['rise_ms']:.2f}ms")
    
    print("\n" + "="*60)
    print("READY FOR PHASE 2 TRAINING!")
    print("="*60)
    print("\nNext steps:")
    print("1. Convert JSON tunings to HDF5 dataset")
    print("2. Train NN_v1 on delta predictions")
    print("3. Validate combined NN_v0 + NN_v1 system")

def extract_feature_vec(dataset, idx):
    """Helper to extract feature vec from dataset"""
    phase_map = {0: "PHASE_NO_CONTACT", 1: "PHASE_IMPACT", 
                 2: "PHASE_HOLD", 3: "PHASE_SLIP", 4: "PHASE_RELEASE"}
    
    phase_val = int(dataset.X[idx, 0] * 4)
    
    return {
        'phase': phase_map.get(phase_val, "PHASE_NO_CONTACT"),
        'phase_confidence_01': float(dataset.X[idx, 1]),
        'normal_force_N': float(dataset.X[idx, 2]),
        'shear_force_N': float(dataset.X[idx, 3]),
        'slip_speed_mms': float(dataset.X[idx, 4]),
        'material_features': {
            'hardness_01': float(dataset.X[idx, 5]),
            'mu_01': float(dataset.X[idx, 6]),
            'roughness_rms_um': float(dataset.X[idx, 7])
        },
        'uncertainty_pct': float(dataset.X[idx, 8])
    }

def compute_delta(baseline, gold):
    """Compute delta = gold - baseline"""
    return {
        'impact': {
            'A': gold['impact']['A'] - baseline['impact']['A'],
            'rise_ms': gold['impact']['rise_ms'] - baseline['impact']['rise_ms'],
            'fall_ms': gold['impact']['fall_ms'] - baseline['impact']['fall_ms'],
            'hf_weight': gold['impact']['hf_weight'] - baseline['impact']['hf_weight']
        },
        'weight': {
            'A': gold['weight']['A'] - baseline['weight']['A'],
            'rate_ms': gold['weight']['rate_ms'] - baseline['weight']['rate_ms']
        },
        'shear': {
            'A': gold['shear']['A'] - baseline['shear']['A'],
            'band_Hz': gold['shear']['band_Hz']
        },
        'texture': {
            'A': gold['texture']['A'] - baseline['texture']['A'],
            'color': gold['texture']['color']
        }
    }

if __name__ == "__main__":
    test_synthetic_expert()