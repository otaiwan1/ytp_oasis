import os
import numpy as np
import trimesh
import open3d as o3d
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_FOLDER = "../collecting-data/stlFiles" 
OUTPUT_FILENAME = "teeth3ds_dataset.npy"
INDEX_FILENAME = "teeth3ds_filenames.json"
NUM_POINTS = 2048
# Increase this if you want safer sample before FPS
INITIAL_SAMPLE = 50000 

# --- GPU KERNEL ---
def farthest_point_sample_gpu(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3] (Tensor on GPU)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud, [npoint, 3] (Tensor on GPU)
    """
    N, C = xyz.shape
    device = xyz.device
    
    centroids = torch.zeros((npoint, C), device=device)
    # Using float32 max value for safety
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()
    
    for i in range(npoint):
        centroids[i] = xyz[farthest]
        centroid = xyz[farthest, :].view(1, 3)
        # Calculate distance
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distance mask
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance).item()
        
    return centroids

# --- CPU WORKER ---
def preprocess_mesh_cpu(file_path):
    """
    Pure CPU Function:
    1. Load Mesh from Disk (I/O)
    2. Center & Align (Geometry Math)
    3. Uniform Sample (Geometry Math)
    
    Returns:
        np.array (10000, 3) or None
    """
    try:
        # 1. Load
        # force='mesh' prevents creating Scenes for single objects
        mesh = trimesh.load(file_path, force='mesh')
        
        # 2. Center
        mesh.vertices -= mesh.center_mass
        
        # 3. PCA Alignment
        inertia = mesh.moment_inertia
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = eigenvectors
        mesh.apply_transform(np.linalg.inv(transform_matrix))
        
        # 4. Initial Uniform Sampling (Heavy CPU OP)
        points_uniform = mesh.sample(INITIAL_SAMPLE)
        
        return points_uniform.astype(np.float32)
        
    except Exception as e:
        # print(f"Error {file_path}: {e}")
        return None

import gc

def process_batch():
    # 1. FIND FILES
    file_list = []
    print(f"Scanning '{SOURCE_FOLDER}' for .obj and .stl files...")
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.lower().endswith(('.obj', '.stl')):
                file_list.append(os.path.join(root, file))
    
    # Sort file_list to ensure determinism
    file_list.sort()
    
    total_files = len(file_list)
    print(f"Found {total_files} files.")
    
    # 2. PROCESS IN CHUNKS (Prevents Memory/Tmp Overflow)
    # Use 15 cores as requested
    num_workers = 15
    CHUNK_SIZE = 100  # Process 100 files at a time
    print(f"Starting Multiprocessing with {num_workers} CPU cores (Batch Size: {CHUNK_SIZE})...")
    
    all_data_chunks = []
    all_filenames = []
    
    # Split file_list into chunks
    file_chunks = [file_list[i:i + CHUNK_SIZE] for i in range(0, total_files, CHUNK_SIZE)]
    
    for chunk_idx, current_files in enumerate(file_chunks):
        print(f"\nProcessing Batch {chunk_idx+1}/{len(file_chunks)} ({len(current_files)} files)...")
        chunk_points = []
        chunk_filenames = []
        
        # Use Context Manager for clean process shutdown/startup per batch
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Keep track of which future belongs to which file path
            future_to_file = {executor.submit(preprocess_mesh_cpu, fp): fp for fp in current_files}
            
            for future in tqdm(as_completed(future_to_file), total=len(current_files), desc=f"Batch {chunk_idx+1}"):
                result = future.result()
                file_path = future_to_file[future]
                
                if result is not None:
                    # 3. GPU PROCESSING (On the fly)
                    if torch.cuda.is_available():
                        try:
                            pts_tensor = torch.from_numpy(result).float().cuda()
                            pts_out = farthest_point_sample_gpu(pts_tensor, NUM_POINTS)
                            chunk_points.append(pts_out.cpu().numpy())
                            chunk_filenames.append(file_path)
                        except Exception as e:
                            print(f"GPU Error: {e}")
                            # Fallback if GPU fails mid-batch
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(result)
                            pcd_final = pcd.farthest_point_down_sample(NUM_POINTS)
                            chunk_points.append(np.asarray(pcd_final.points))
                            chunk_filenames.append(file_path)
                    else:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(result)
                        pcd_final = pcd.farthest_point_down_sample(NUM_POINTS)
                        chunk_points.append(np.asarray(pcd_final.points))
                        chunk_filenames.append(file_path)
        
        # Consolidation: Convert list to numpy array immediately to save RAM
        if len(chunk_points) > 0:
            chunk_arr = np.array(chunk_points, dtype=np.float32)
            all_data_chunks.append(chunk_arr)
            all_filenames.extend(chunk_filenames)
            
        # Clean up memory explicitly
        del chunk_points
        del chunk_filenames
        del future_to_file
        gc.collect()

    # 4. SAVE RESULT
    if len(all_data_chunks) > 0:
        print("\nConcatenating all batches...")
        final_data = np.concatenate(all_data_chunks, axis=0)
        np.save(OUTPUT_FILENAME, final_data)
        
        # Save Filenames Index
        import json
        with open(INDEX_FILENAME, "w") as f:
            json.dump(all_filenames, f)
        
        print("-" * 30)
        print("SUCCESS!")
        print(f"Saved dataset to: {OUTPUT_FILENAME}")
        print(f"Saved index map to: {INDEX_FILENAME}")
        print(f"Total Patients: {final_data.shape[0]}") 
        print(f"Points per Patient: {final_data.shape[1]}")
    else:
        print("Error: No valid 3D files were processed.")

if __name__ == "__main__":
    # Ensure correct start method if needed, though default usually works for this setup
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    process_batch()
