import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

# --- CONFIGURATION ---
# Point this to the root folder where your data is stored.
# It will recursively find files in subfolders (e.g., Upper/015WXFRN/...).
SOURCE_FOLDER = "./data_part_1" 

# This is the "Textbook" file we are creating.
OUTPUT_FILENAME = "teeth3ds_dataset.npy"

# The fixed number of points every patient must have.
NUM_POINTS = 4096

def get_normalized_pcd(file_path):
    """
    1. Loads the 3D mesh.
    2. Centers it at (0,0,0).
    3. Rotates it to face "forward" (PCA).
    4. Turns it into a cloud of 4,096 points.
    """
    try:
        # force='mesh' merges multi-part files into one single dental arch
        mesh = trimesh.load(file_path, force='mesh')
        
        # --- A. Center the Mesh ---
        mesh.vertices -= mesh.center_mass
        
        # --- B. PCA Alignment (Fix Rotation) ---
        # Calculate the "principal axes" (length, width, height)
        inertia = mesh.moment_inertia
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Create a rotation matrix to align these axes with X, Y, Z
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = eigenvectors
        mesh.apply_transform(np.linalg.inv(transform_matrix))
        
        # --- C. Sampling (Mesh -> Points) ---
        # 1. Sample 10,000 points randomly first (fast)
        points_uniform = mesh.sample(10000)
        
        # 2. Use Farthest Point Sampling (FPS) to pick the BEST 4,096 points
        # FPS ensures we capture the sharp cusps of teeth, not just flat gums.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_uniform)
        pcd_final = pcd.farthest_point_down_sample(NUM_POINTS)
        
        return np.asarray(pcd_final.points)
        
    except Exception as e:
        # If a file is corrupt, print a warning but keep going
        # print(f"Skipping {file_path}: {e}")
        return None

def process_batch():
    all_points = []
    file_list = []
    
    # 1. FIND FILES
    print(f"Scanning '{SOURCE_FOLDER}' for .obj and .stl files...")
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            # Look for both .obj (Teeth3DS) and .stl (Your future data)
            if file.lower().endswith(('.obj', '.stl')):
                full_path = os.path.join(root, file)
                file_list.append(full_path)
    
    print(f"Found {len(file_list)} files. Starting processing...")
    
    # 2. PROCESS FILES
    # tqdm shows a progress bar so you know how long it will take
    for file_path in tqdm(file_list):
        pcd_array = get_normalized_pcd(file_path)
        
        if pcd_array is not None:
            # Double check we actually got the right shape
            if pcd_array.shape == (NUM_POINTS, 3):
                all_points.append(pcd_array)

    # 3. SAVE RESULT
    if len(all_points) > 0:
        final_data = np.array(all_points)
        np.save(OUTPUT_FILENAME, final_data)
        
        print("-" * 30)
        print("SUCCESS!")
        print(f"Saved dataset to: {OUTPUT_FILENAME}")
        print(f"Total Patients: {final_data.shape[0]}") # e.g., 2000
        print(f"Points per Patient: {final_data.shape[1]}") # 4096
    else:
        print("Error: No valid 3D files were processed.")

if __name__ == "__main__":
    process_batch()