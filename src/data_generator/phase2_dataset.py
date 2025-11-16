"""
Phase 2 dataset manager: Handles delta training data.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from typing import List, Dict

class Phase2DatasetManager:
    """
    Manages Phase 2 delta training datasets.
    Similar to Phase 1, but stores deltas instead of absolute values.
    """
    
    def __init__(self, output_dir: str = "data/phase2_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_json_to_hdf5(self, json_path: str, dataset_name: str):
        """
        Convert Phase 2 JSON tunings to HDF5 format for training.
        
        Args:
            json_path: Path to JSON file with tunings
            dataset_name: Output dataset name
        """
        print(f"Loading tunings from: {json_path}")
        
        with open(json_path, 'r') as f:
            tunings = json.load(f)
        
        print(f"Found {len(tunings)} tunings")
        
        # Extract X (features)
        X = []
        for t in tunings:
            fv = t['feature_vec']
            features = [
                self._encode_phase(fv['phase']) / 4.0,
                fv['phase_confidence_01'],
                fv['normal_force_N'],
                fv['shear_force_N'],
                fv['slip_speed_mms'],
                fv['material_features']['hardness_01'],
                fv['material_features']['mu_01'],
                fv['material_features']['roughness_rms_um'],
                fv['uncertainty_pct']
            ]
            X.append(features)
        
        X = np.array(X, dtype=np.float32)
        
        # Extract Y (deltas)
        delta_impact_A = [t['delta_cues']['impact']['A'] for t in tunings]
        delta_impact_rise = [t['delta_cues']['impact']['rise_ms'] for t in tunings]
        delta_impact_fall = [t['delta_cues']['impact']['fall_ms'] for t in tunings]
        delta_impact_hf = [t['delta_cues']['impact']['hf_weight'] for t in tunings]
        
        delta_weight_A = [t['delta_cues']['weight']['A'] for t in tunings]
        delta_weight_rate = [t['delta_cues']['weight']['rate_ms'] for t in tunings]
        
        delta_shear_A = [t['delta_cues']['shear']['A'] for t in tunings]
        delta_shear_band = [t['delta_cues']['shear']['band_Hz'][0] if isinstance(t['delta_cues']['shear']['band_Hz'], list) else t['delta_cues']['shear']['band_Hz'] for t in tunings]
        
        delta_texture_A = [t['delta_cues']['texture']['A'] for t in tunings]
        delta_texture_color = [self._encode_color(t['delta_cues']['texture']['color']) for t in tunings]
        
        # Save to HDF5
        output_path = self.output_dir / f"{dataset_name}.h5"
        
        print(f"\nSaving to HDF5: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save X
            grp_x = f.create_group('X')
            for i, name in enumerate(['phase', 'phase_confidence', 'normal_force_N', 
                                     'shear_force_N', 'slip_speed_mms', 'hardness',
                                     'mu', 'roughness', 'uncertainty']):
                grp_x.create_dataset(name, data=X[:, i])
            
            # Save deltas (Y)
            grp_y = f.create_group('Y_delta')
            
            grp_impact = grp_y.create_group('impact')
            grp_impact.create_dataset('A', data=np.array(delta_impact_A, dtype=np.float32))
            grp_impact.create_dataset('rise_ms', data=np.array(delta_impact_rise, dtype=np.float32))
            grp_impact.create_dataset('fall_ms', data=np.array(delta_impact_fall, dtype=np.float32))
            grp_impact.create_dataset('hf_weight', data=np.array(delta_impact_hf, dtype=np.float32))
            
            grp_weight = grp_y.create_group('weight')
            grp_weight.create_dataset('A', data=np.array(delta_weight_A, dtype=np.float32))
            grp_weight.create_dataset('rate_ms', data=np.array(delta_weight_rate, dtype=np.float32))
            
            grp_shear = grp_y.create_group('shear')
            grp_shear.create_dataset('A', data=np.array(delta_shear_A, dtype=np.float32))
            grp_shear.create_dataset('band_Hz', data=np.array(delta_shear_band, dtype=np.float32))
            
            grp_texture = grp_y.create_group('texture')
            grp_texture.create_dataset('A', data=np.array(delta_texture_A, dtype=np.float32))
            grp_texture.create_dataset('color', data=np.array(delta_texture_color, dtype=np.int32))
            
            # Metadata
            f.attrs['n_samples'] = len(tunings)
            f.attrs['dataset_type'] = 'phase2_deltas'
            f.attrs['expert_id'] = tunings[0]['expert_id'] if tunings else 'unknown'
        
        print(f"✓ Saved {len(tunings)} delta samples")
        
        # Print statistics
        self._print_delta_stats(tunings)
        
        return output_path
    
    def _encode_phase(self, phase_str: str) -> int:
        phase_map = {
            "PHASE_NO_CONTACT": 0,
            "PHASE_IMPACT": 1,
            "PHASE_HOLD": 2,
            "PHASE_SLIP": 3,
            "PHASE_RELEASE": 4
        }
        return phase_map.get(phase_str, 0)
    
    def _encode_color(self, color_str: str) -> int:
        color_map = {
            "COLOR_WHITE": 0,
            "COLOR_PINK": 1,
            "COLOR_BROWN": 2
        }
        return color_map.get(color_str, 1)
    
    def _print_delta_stats(self, tunings: List[Dict]):
        """Print statistics about the deltas"""
        import numpy as np
        
        impact_A_deltas = [t['delta_cues']['impact']['A'] for t in tunings]
        impact_fall_deltas = [t['delta_cues']['impact']['fall_ms'] for t in tunings]
        weight_A_deltas = [t['delta_cues']['weight']['A'] for t in tunings]
        
        print(f"\n{'='*60}")
        print(f"DELTA STATISTICS")
        print(f"{'='*60}")
        print(f"Impact A deltas:")
        print(f"  Mean: {np.mean(impact_A_deltas):.4f}")
        print(f"  Std:  {np.std(impact_A_deltas):.4f}")
        print(f"  Range: [{np.min(impact_A_deltas):.4f}, {np.max(impact_A_deltas):.4f}]")
        
        print(f"\nImpact fall_ms deltas:")
        print(f"  Mean: {np.mean(impact_fall_deltas):.4f} ms")
        print(f"  Std:  {np.std(impact_fall_deltas):.4f} ms")
        
        print(f"\nWeight A deltas:")
        print(f"  Mean: {np.mean(weight_A_deltas):.4f}")
        print(f"  Std:  {np.std(weight_A_deltas):.4f}")
        print(f"  Range: [{np.min(weight_A_deltas):.4f}, {np.max(weight_A_deltas):.4f}]")
        print(f"{'='*60}\n")


class Phase2DatasetLoader:
    """Load Phase 2 delta datasets for training"""
    
    def __init__(self, dataset_path: str):
        import h5py
        self.dataset_path = dataset_path
        
        with h5py.File(dataset_path, 'r') as f:
            # Load X (same as Phase 1)
            self.X = np.stack([
                f['X/phase'][:],
                f['X/phase_confidence'][:],
                f['X/normal_force_N'][:],
                f['X/shear_force_N'][:],
                f['X/slip_speed_mms'][:],
                f['X/hardness'][:],
                f['X/mu'][:],
                f['X/roughness'][:],
                f['X/uncertainty'][:]
            ], axis=1).astype(np.float32)
            
            # Load Y_delta (deltas instead of absolute values)
            self.Y_delta = {
                'impact': {
                    'A': f['Y_delta/impact/A'][:].astype(np.float32),
                    'rise_ms': f['Y_delta/impact/rise_ms'][:].astype(np.float32),
                    'fall_ms': f['Y_delta/impact/fall_ms'][:].astype(np.float32),
                    'hf_weight': f['Y_delta/impact/hf_weight'][:].astype(np.float32)
                },
                'weight': {
                    'A': f['Y_delta/weight/A'][:].astype(np.float32),
                    'rate_ms': f['Y_delta/weight/rate_ms'][:].astype(np.float32)
                },
                'shear': {
                    'A': f['Y_delta/shear/A'][:].astype(np.float32),
                    'band_Hz': f['Y_delta/shear/band_Hz'][:].astype(np.float32)
                },
                'texture': {
                    'A': f['Y_delta/texture/A'][:].astype(np.float32),
                    'color': f['Y_delta/texture/color'][:].astype(np.int64)
                }
            }
            
            self.n_samples = len(self.X)
    
    def get_train_val_split(self, val_split: float = 0.2):
        """Split into train/val sets"""
        n_val = int(self.n_samples * val_split)
        indices = np.random.permutation(self.n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        return train_indices, val_indices
    
    def get_batch(self, indices):
        """Get a batch of data"""
        import torch
        
        X_batch = torch.from_numpy(self.X[indices])
        
        Y_delta_batch = {
            'impact': {
                'A': torch.from_numpy(self.Y_delta['impact']['A'][indices]),
                'rise_ms': torch.from_numpy(self.Y_delta['impact']['rise_ms'][indices]),
                'fall_ms': torch.from_numpy(self.Y_delta['impact']['fall_ms'][indices]),
                'hf_weight': torch.from_numpy(self.Y_delta['impact']['hf_weight'][indices])
            },
            'weight': {
                'A': torch.from_numpy(self.Y_delta['weight']['A'][indices]),
                'rate_ms': torch.from_numpy(self.Y_delta['weight']['rate_ms'][indices])
            },
            'shear': {
                'A': torch.from_numpy(self.Y_delta['shear']['A'][indices]),
                'band_Hz': torch.from_numpy(self.Y_delta['shear']['band_Hz'][indices])
            },
            'texture': {
                'A': torch.from_numpy(self.Y_delta['texture']['A'][indices]),
                'color': torch.from_numpy(self.Y_delta['texture']['color'][indices])
            }
        }
        
        return X_batch, Y_delta_batch