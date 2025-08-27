#!/usr/bin/env python
"""Test script for the EKS data loader."""

import numpy as np
import keypoint_moseq as kpms

# Test loading the EKS CSV file
print("Testing EKS loader...")
print("-" * 50)

# Load the sample EKS file
filepath = "sub-460_strain-B6_2024-11-12T12_30_00_eks.csv"
coordinates, confidences, bodyparts = kpms.load_keypoints(filepath, "eks")

# Display results
for recording_name in coordinates:
    print(f"\nRecording: {recording_name}")
    print(f"  Coordinates shape: {coordinates[recording_name].shape}")
    print(f"  Confidences shape: {confidences[recording_name].shape}")
    print(f"  Number of frames: {coordinates[recording_name].shape[0]}")
    print(f"  Number of bodyparts: {coordinates[recording_name].shape[1]}")
    print(f"  Dimensions: {coordinates[recording_name].shape[2]}")
    
    # Display bodyparts
    print(f"\n  Bodyparts ({len(bodyparts)} total):")
    for i, bp in enumerate(bodyparts):
        if i < 5:  # Show first 5 bodyparts
            print(f"    {i}: {bp}")
        elif i == 5:
            print(f"    ...")
        elif i >= len(bodyparts) - 2:  # Show last 2 bodyparts
            print(f"    {i}: {bp}")
    
    # Check data validity
    coords = coordinates[recording_name]
    confs = confidences[recording_name]
    
    print(f"\n  Data statistics:")
    print(f"    Coordinates range: [{np.min(coords):.2f}, {np.max(coords):.2f}]")
    print(f"    Confidences range: [{np.min(confs):.4f}, {np.max(confs):.4f}]")
    print(f"    NaN values in coordinates: {np.sum(np.isnan(coords))}")
    print(f"    NaN values in confidences: {np.sum(np.isnan(confs))}")
    
    # Show sample data for first bodypart
    print(f"\n  Sample data for first bodypart '{bodyparts[0]}':")
    print(f"    First 3 frames (x, y):")
    for i in range(min(3, coords.shape[0])):
        x, y = coords[i, 0, :]
        conf = confs[i, 0]
        print(f"      Frame {i}: ({x:.2f}, {y:.2f}), confidence: {conf:.4f}")

print("\n" + "=" * 50)
print("EKS loader test completed successfully!")