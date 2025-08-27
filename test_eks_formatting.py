#!/usr/bin/env python
"""
Test script to verify EKS data formatting with correct bodyparts configuration
"""

import numpy as np
import keypoint_moseq as kpms

# Load the EKS data
print("=== Loading EKS data ===")
coordinates, confidences, bodyparts = kpms.load_keypoints(
    ['sub-460_strain-B6_2024-11-12T12_30_00_eks.csv'], 
    format='eks'
)

print(f"Loaded data for {len(coordinates)} recording(s)")
for recording_name in coordinates:
    print(f"  {recording_name}:")
    print(f"    Shape: {coordinates[recording_name].shape}")
    print(f"    Bodyparts: {len(bodyparts)}")

# Create config with the correct bodyparts from EKS data
config = lambda: {
    'bodyparts': bodyparts,  # Use the bodyparts directly from the loaded data
    'use_bodyparts': bodyparts,  # Use all bodyparts for now
    'anterior_bodyparts': ['nose', 'left_ear', 'right_ear', 'neck'],
    'posterior_bodyparts': ['tail_base', 'mid_backend3'],
    'conf_pseudocount': 1e-3,
    'video_dir': '.',
    'added_noise_level': 0.1,
    'verbose': True
}

# Skip outlier detection for now
print("\n=== Skipping outlier detection ===")
print("Proceeding directly to data formatting...")

# Format data for modeling
print("\n=== Formatting data for modeling ===")
try:
    result = kpms.format_data(coordinates, confidences, **config())
    print("✓ Data formatting successful!")
    
    if isinstance(result, tuple) and len(result) == 2:
        data, metadata = result
        print(f"  Formatted data keys: {list(data.keys())}")
        print(f"  Metadata type: {type(metadata)}")
        if hasattr(metadata, 'keys'):
            print(f"  Metadata keys: {list(metadata.keys())}")
        else:
            print(f"  Metadata: {metadata}")
    else:
        data = result
        metadata = None
        print(f"  Formatted data keys: {list(data.keys())}")
        print("  No metadata returned")
    
    # Print some data statistics
    if 'Y' in data:
        print(f"\n  Observations (Y) shape: {data['Y'].shape}")
    if 'conf' in data:
        print(f"  Confidences shape: {data['conf'].shape}")
    if 'mask' in data:
        print(f"  Mask shape: {data['mask'].shape}")
        print(f"  Valid frames: {data['mask'].sum()} / {data['mask'].size}")
        
except Exception as e:
    print(f"✗ Error during formatting: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completed ===")