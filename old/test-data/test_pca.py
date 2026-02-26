import numpy as np
import trimesh
import open3d as o3d

def normalize_and_sample(file_path, num_points=4096):
    """
    Takes a path to a .stl/.ply file and returns a normalized (N, 3) numpy array.
    """
    
    # --- 1. LOAD THE MESH ---
    # We use trimesh for robust loading (handles file formats better) 
    print(f"Loading: {file_path}")
    mesh = trimesh.load(file_path)
    
    # --- 2. POSE NORMALIZATION (PCA ALIGNMENT) ---
    # Deep learning models are sensitive to rotation. If Patient A is scanned 
    # at 0 degrees and Patient B at 90 degrees, the AI thinks they are different.
    # We use Principal Component Analysis (PCA) to align them[cite: 168, 170].
    
    # A. Center the mesh at the origin (0,0,0)
    mesh.vertices -= mesh.center_mass
    
    # B. Compute the Principal Axes (Eigenvectors of covariance matrix) [cite: 171, 172]
    # This finds the "long axis" (length) and "short axis" (width) of the teeth.
    inertia = mesh.moment_inertia
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    
    # C. Rotate mesh so its principal axes align with X, Y, Z
    # We create a 4x4 transformation matrix from the eigenvectors
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = eigenvectors
    mesh.apply_transform(np.linalg.inv(transform_matrix))
    
    # --- 3. SAMPLING (CONVERT TO POINT CLOUD) ---
    # Neural networks need a fixed input size (e.g., 2048 points).
    # We convert the continuous mesh into a discrete point cloud.
    
    # Option A: Uniform Sampling (Fast, good for general shape)
    points_uniform = mesh.sample(5000) # Sample more first
    
    # Option B: Farthest Point Sampling (FPS) - The "Optimal Path" [cite: 166, 332]
    # We pass data to Open3D because it has a fast C++ implementation of FPS.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_uniform)
    
    # This selects points that are farthest apart, preserving coverage of the whole arch
    pcd_final = pcd.farthest_point_down_sample(num_points)
    
    # Convert back to numpy array
    final_point_cloud = np.asarray(pcd_final.points)
    
    return final_point_cloud

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Replace with a real file path from your dataset
    input_file = "test_case.stl" 
    
    try:
        # Run the pipeline
        point_cloud = normalize_and_sample(input_file)
        
        print(f"Success! Generated Point Cloud Shape: {point_cloud.shape}")
        
        # VISUALIZATION (To verify it worked)
        # We visualize the final points to ensure they look like a dental arch
        pcd_viz = o3d.geometry.PointCloud()
        pcd_viz.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Add a coordinate frame to see if it's aligned (Red=X, Green=Y, Blue=Z)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)
        
        print("Opening visualizer... (Close window to finish)")
        o3d.visualization.draw_geometries([pcd_viz, coord_frame])
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a sample .stl file in the folder!")