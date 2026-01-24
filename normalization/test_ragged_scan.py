"""
import numpy as np

# Load with pickle allowed just to inspect it
data = np.load("normalization/teeth3ds_dataset.npy", allow_pickle=True)

print(f"Overall Shape: {data.shape}") # Expecting (N, 4096, 3)
print(f"Data Type: {data.dtype}")     # Expecting float32, likely 'object'

# Check if it is a 'Ragged Array'
if data.dtype == 'object':
    print("\nDETECTED RAGGED ARRAY!")
    print("Checking for shape mismatches...")
    for i, scan in enumerate(data):
        if scan.shape != (4096, 3):
            print(f"  -> Index {i} has weird shape: {scan.shape}")
            break
"""
import numpy as np
import os

# Define possible locations for the file
possible_paths = [
    "teeth3ds_dataset.npy",                # If file is in the same folder as this script
    "../teeth3ds_dataset.npy",             # If file is in the project root
    "normalization/teeth3ds_dataset.npy"   # If running from root looking into folder
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("❌ ERROR: Could not find 'teeth3ds_dataset.npy'.")
    print(f"   I looked in: {possible_paths}")
    print("   Please make sure the file exists and move it to this folder.")
    exit()

print(f"✅ Found file at: {dataset_path}")

# Check file size
file_size = os.path.getsize(dataset_path)
print(f"   File size: {file_size / 1024:.2f} KB")

if file_size < 1024:
    print("\n⚠️  WARNING: File is suspiciously small (< 1KB).")
    print("   A valid dataset should be several MBs.")
    print("   It is likely empty or corrupted.")

print("   Loading with allow_pickle=True...")

# --- LOAD AND INSPECT ---
try:
    data = np.load(dataset_path, allow_pickle=True)
    
    # 1. Check if it's an object array (Ragged)
    if data.dtype == 'object':
        print("\n⚠️  DETECTED RAGGED ARRAY (Object Array)!")
        print("   This means not all scans have the same shape.")
        print("   Scanning for the bad file...")
        
        found_bad = False
        for i, scan in enumerate(data):
            # Check for None or wrong shape
            if scan is None:
                 print(f"   -> ❌ Index {i} is None (Corrupted file?)")
                 found_bad = True
            elif hasattr(scan, 'shape') and scan.shape != (4096, 3):
                print(f"   -> ❌ Index {i} has wrong shape: {scan.shape} (Expected 4096, 3)")
                found_bad = True
            elif not hasattr(scan, 'shape'):
                print(f"   -> ❌ Index {i} is not a numpy array. It is type: {type(scan)}")
                found_bad = True
                
        if not found_bad:
            print("   (Strange... dtype is object but all shapes seem correct. Did you mix float32 and float64?)")
            
    # 2. Check if it's a perfect matrix
    else:
        print(f"\n✅ Data is a perfect uniform matrix.")
        print(f"   Shape: {data.shape}")
        print(f"   Type:  {data.dtype}")
        print("   You strictly don't need pickle, but allow_pickle=True won't hurt.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR while loading: {e}")