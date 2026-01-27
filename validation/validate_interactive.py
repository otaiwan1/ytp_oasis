import open3d as o3d

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path

# --- 1. SETUP PATHS ---
# Get the folder where THIS script lives (Project_Root/validation)
current_folder = Path(__file__).parent.resolve()
# Get the Project Root (Project_Root/)
project_root = current_folder.parent

# Define the neighbor folders based on your structure
TRAIN_DIR = project_root / "train"
DATA_DIR = project_root / "normalization"

# Add 'train' folder to Python's search path so we can import the model class
sys.path.append(str(TRAIN_DIR))

# --- 2. IMPORT MODEL ---
try:
    # NOTE: Ensure your training script is named 'train_oasis.py' as per your tree
    from train_oasis import SimCLREncoder
    print(f"Successfully imported model class from: {TRAIN_DIR}")
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import from 'train_oasis.py'")
    print(f"I looked in: {TRAIN_DIR}")
    print(f"Python Error: {e}")
    print("Check: Is the file in the 'train' folder named 'train_oasis.py'?")
    sys.exit()

# --- 3. CONFIGURATION ---
# Using the filenames from your tree structure
MODEL_PATH = TRAIN_DIR / "oasis_simclr_edgeconv.pth"
DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
INDEX_PATH = DATA_DIR / "teeth3ds_filenames.json"  # New Index File
REPORT_SAVE_PATH = current_folder / "oasis_validation_report.csv"

NUM_TEST_CASES = 10
TOP_K = 5

def load_model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit()
    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found at {MODEL_PATH}")
        sys.exit()

    print("Loading data...")
    data = np.load(DATASET_PATH)
    
    # Load Index Map if available
    filenames = []
    if INDEX_PATH.exists():
        import json
        with open(INDEX_PATH, 'r') as f:
            filenames = json.load(f)
        print(f"Loaded {len(filenames)} filename mappings.")
    else:
        print("Warning: No filename index found. You will only see IDs.")

    print("Loading model...")
    model = SimCLREncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Finish load_model_and_data()!")
    
    return model, data, filenames, device

def get_vectors(model, data, device):
    """Convert point clouds to searchable vectors"""
    print("Indexing database...")
    vectors = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            tensor = torch.tensor(batch, dtype=torch.float32).transpose(2, 1).to(device)
            reps, _ = model(tensor) 
            vectors.append(reps.cpu().numpy())
            
    return np.vstack(vectors)

import trimesh

def visualize_pair(query_data, match_data, title, query_name="Query", match_name="Match"):
    """
    Visualizes two 3D objects.
    If 'query_data' is a string, it loads the mesh file (STL/OBJ).
    If 'query_data' is numpy array, it creates a point cloud.
    """
    geometries = []

    # --- Helper to load Mesh or PCD ---
    def load_geometry(data, color, shift_x=0):
        geom = None
        if isinstance(data, str) and os.path.exists(data):
            # Load Mesh (High Quality)
            try:
                mesh = trimesh.load(data, force='mesh')
                # Trimesh -> Open3D
                geom = o3d.geometry.TriangleMesh()
                geom.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                geom.triangles = o3d.utility.Vector3iVector(mesh.faces)
                geom.compute_vertex_normals()
                geom.paint_uniform_color(color)
            except Exception as e:
                print(f"Error loading mesh {data}: {e}")
        
        # Fallback or if data is Point Cloud
        if geom is None and isinstance(data, np.ndarray):
            geom = o3d.geometry.PointCloud()
            geom.points = o3d.utility.Vector3dVector(data)
            geom.paint_uniform_color(color)
            
        if geom:
            # Shift position
            if shift_x != 0:
                geom.translate((shift_x, 0, 0))
        return geom

    # 1. Load Query (Red)
    query_geom = load_geometry(query_data, [1, 0.7, 0.7], shift_x=0) # Light Red
    if query_geom: geometries.append(query_geom)
    
    # 2. Load Match (Green)
    # Estimate width for spacing
    width = 50.0 # Default fallback width
    if isinstance(query_data, np.ndarray):
        width = np.max(query_data[:, 0]) - np.min(query_data[:, 0])
        
    match_geom = load_geometry(match_data, [0.7, 1, 0.7], shift_x=width * 1.5) # Light Green
    if match_geom: geometries.append(match_geom)
    
    print(f"Visualizing: {title}")
    # Show window with file names
    print(f"  Left (Red): {query_name}")
    print(f"  Right (Green): {match_name}")
    o3d.visualization.draw_geometries(geometries, window_name=title)

def run_validation():
    model, data, filenames, device = load_model_and_data()
    all_vectors = get_vectors(model, data, device)
    
    # Normalize for Cosine Similarity
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    normalized_vectors = all_vectors / norms
    
    results_log = []
    # Ensure we don't try to test more cases than we have
    num_to_test = min(NUM_TEST_CASES, len(data))
    test_indices = random.sample(range(len(data)), num_to_test)
    
    print(f"\nStarting Validation on {num_to_test} random cases...")
    print("Grading Scale: 0 (Bad) to 3 (Perfect)")
    
    for i, query_idx in enumerate(test_indices):
        print(f"\n--- CASE {i+1}/{num_to_test} ---")
        
        # Search
        query_vec = normalized_vectors[query_idx:query_idx+1]
        similarities = np.dot(normalized_vectors, query_vec.T).flatten()
        
        # Get Top K (excluding self at rank 0, usually index 0 is self because dist=1.0)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter out self (query_idx) from results
        top_indices = [idx for idx in sorted_indices if idx != query_idx][:TOP_K]
        
        for rank, match_idx in enumerate(top_indices):
            score = similarities[match_idx]
            
            # Prepare Data for Visualization
            # If we have filenames, try to load the original STL
            query_obj = data[query_idx]
            match_obj = data[match_idx]
            q_name = f"ID_{query_idx}"
            m_name = f"ID_{match_idx}"
            
            if len(filenames) > 0:
                # Use raw file path if available
                query_obj = filenames[query_idx] 
                match_obj = filenames[match_idx]
                q_name = os.path.basename(query_obj)
                m_name = os.path.basename(match_obj)

            # Visualize
            title = f"Rank {rank+1} | Score: {score:.4f}"
            visualize_pair(query_obj, match_obj, title, q_name, m_name)
            
            # Input Grade
            while True:
                try:
                    val = input(f"Rate Match {rank+1} (0-3): ")
                    grade = int(val)
                    if 0 <= grade <= 3: break
                except ValueError: pass
            
            results_log.append({
                "Query_ID": query_idx,
                "Match_ID": match_idx,
                "Dentist_Grade": grade,
                "Query_File": q_name,
                "Match_File": m_name
            })
            
    # Save Report
    df = pd.DataFrame(results_log)
    df.to_csv(REPORT_SAVE_PATH, index=False)
    print(f"\nSaved report to: {REPORT_SAVE_PATH}")
    
    if len(df) > 0:
        precision = len(df[df['Dentist_Grade']>=2])/len(df)
        print(f"Precision (>2): {precision:.1%}")

if __name__ == "__main__":
    run_validation()