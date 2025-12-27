import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

# --- CONFIGURATION ---
# The script will start here and dig into all subfolders (Upper, Lower, CaseIDs, etc.)
SOURCE_FOLDER = "./data_part_1" 
OUTPUT_FILENAME = "teeth3ds_dataset.npy"
NUM_POINTS = 4096

def get_normalized_pcd(file_path):
    """
    Loads .obj, normalizes rotation (PCA), and samples 4096 points.
    """
    try:
        # force='mesh' is crucial for .obj files to ensure they load as a single object
        # rather than a "Scene" of disjoint parts.
        mesh = trimesh.load(file_path, force='mesh')
        
        # --- 1. PCA Pose Normalization ---
        # Center the mesh
        mesh.vertices -= mesh.center_mass
        
        # Align with Principal Axes (X, Y, Z)
        inertia = mesh.moment_inertia
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = eigenvectors
        mesh.apply_transform(np.linalg.inv(transform_matrix))
        
        # --- 2. Sampling ---
        # Sample 10k first to give FPS a good candidate pool
        points_uniform = mesh.sample(10000)
        
        # Convert to Open3D for efficient Farthest Point Sampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_uniform)
        pcd_final = pcd.farthest_point_down_sample(NUM_POINTS)
        
        return np.asarray(pcd_final.points)
        
    except Exception as e:
        # This catches empty files or corrupt geometry
        # print(f"Warning: Could not process {file_path} - {e}")
        return None

def process_batch():
    all_points = []
    file_list = []
    
    print(f"Scanning '{SOURCE_FOLDER}' recursively...")
    
    # os.walk automatically dives into 'Upper', then '015WXFRN', etc.
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            # We strictly look for .obj files and ignore .json
            if file.lower().endswith('.obj'):
                full_path = os.path.join(root, file)
                file_list.append(full_path)
    
    print(f"Found {len(file_list)} .obj files. Starting processing...")
    
    # Process files with a progress bar
    for file_path in tqdm(file_list):
        pcd_array = get_normalized_pcd(file_path)
        
        if pcd_array is not None:
            # Sanity check: ensure we actually got 4096 points
            if pcd_array.shape == (NUM_POINTS, 3):
                all_points.append(pcd_array)

    # Convert list to one big numpy array
    final_data = np.array(all_points)
    
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Scans Processed: {len(final_data)}")
    print(f"Output Shape: {final_data.shape}") # Should be (N, 4096, 3)
    
    # Save the file
    np.save(OUTPUT_FILENAME, final_data)
    print(f"Saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_batch()