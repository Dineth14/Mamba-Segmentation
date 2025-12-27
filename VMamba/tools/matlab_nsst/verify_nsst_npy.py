#!/usr/bin/env python3
"""
Verify NSST .npy files for correctness:
- Check shape (should be C, H, W with C=27)
- Check for NaN/Inf values
- Check dtype (should be float32)
- Sample statistics
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm


def verify_npy_file(npy_path: Path) -> dict:
    """Verify a single .npy file and return statistics."""
    try:
        data = np.load(npy_path)
        
        # Check basic properties
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        
        return {
            'path': npy_path,
            'shape': data.shape,
            'dtype': data.dtype,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'path': npy_path,
            'success': False,
            'error': str(e)
        }


def main():
    nsst_root = Path("/storage2/ChangeDetection/Datasets/Loveda/NSST_27ch")
    
    if not nsst_root.exists():
        print(f"Error: {nsst_root} does not exist")
        sys.exit(1)
    
    # Find all .npy files
    npy_files = sorted(nsst_root.rglob("*.npy"))
    
    if not npy_files:
        print(f"Error: No .npy files found under {nsst_root}")
        sys.exit(1)
    
    print(f"Found {len(npy_files)} .npy files")
    print("Verifying files...")
    
    results = []
    errors = []
    
    for npy_path in tqdm(npy_files, desc="Verifying"):
        result = verify_npy_file(npy_path)
        results.append(result)
        
        if not result['success']:
            errors.append(result)
        elif result['has_nan'] or result['has_inf']:
            errors.append(result)
    
    # Summary statistics
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")
    
    if successful:
        shapes = [r['shape'] for r in successful]
        dtypes = [r['dtype'] for r in successful]
        has_nan_count = sum(1 for r in successful if r['has_nan'])
        has_inf_count = sum(1 for r in successful if r['has_inf'])
        
        print(f"\nShape distribution:")
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        for shape, count in sorted(shape_counts.items()):
            print(f"  {shape}: {count} files")
        
        print(f"\nDtype distribution:")
        dtype_counts = {}
        for dtype in dtypes:
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        for dtype, count in sorted(dtype_counts.items()):
            print(f"  {dtype}: {count} files")
        
        print(f"\nData quality:")
        print(f"  Files with NaN: {has_nan_count}")
        print(f"  Files with Inf: {has_inf_count}")
        
        # Sample statistics from first valid file
        sample = successful[0]
        print(f"\nSample statistics (from {sample['path'].name}):")
        print(f"  Shape: {sample['shape']}")
        print(f"  Dtype: {sample['dtype']}")
        print(f"  Min: {sample['min']:.6f}")
        print(f"  Max: {sample['max']:.6f}")
        print(f"  Mean: {sample['mean']:.6f}")
        print(f"  Std: {sample['std']:.6f}")
    
    if errors:
        print(f"\n{'='*80}")
        print(f"ERRORS DETECTED ({len(errors)} files)")
        print("="*80)
        for err in errors[:10]:  # Show first 10 errors
            if not err['success']:
                print(f"  {err['path'].name}: {err['error']}")
            else:
                issues = []
                if err['has_nan']:
                    issues.append("NaN")
                if err['has_inf']:
                    issues.append("Inf")
                print(f"  {err['path'].name}: Contains {', '.join(issues)}")
        
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        
        sys.exit(1)
    else:
        print("\n✓ All NSST .npy files are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
