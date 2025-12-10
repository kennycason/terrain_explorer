"""Analyze the .npy heightmap file"""
import numpy as np

# Load the raw data
data = np.load('raw_map_20251210_134411.npy')

print("="*50)
print("NPY FILE ANALYSIS")
print("="*50)
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min value: {data.min()}")
print(f"Max value: {data.max()}")
print(f"Mean value: {data.mean():.4f}")
print(f"Std dev: {data.std():.4f}")
print()

# Check for NaN or Inf values
print(f"Contains NaN: {np.isnan(data).any()}")
print(f"Contains Inf: {np.isinf(data).any()}")
print()

# Sample some values
print("Sample values (center region):")
h, w = data.shape[:2]
center_y, center_x = h//2, w//2
if len(data.shape) == 2:
    print(f"  Center: {data[center_y, center_x]}")
    print(f"  Top-left corner: {data[0, 0]}")
    print(f"  Bottom-right corner: {data[-1, -1]}")
elif len(data.shape) == 3:
    print(f"  Center: {data[center_y, center_x]}")
    print(f"  Channels: {data.shape[2]}")

print()
print("Value distribution:")
percentiles = [0, 10, 25, 50, 75, 90, 100]
for p in percentiles:
    val = np.percentile(data, p)
    print(f"  {p}th percentile: {val:.4f}")

