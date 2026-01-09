import torch
import numpy as np
import open3d as o3d
import pandas as pd
import random
import os
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
MODEL_PATH = TRAIN_DIR / "oasis_model_v1_cuda.pth"
DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
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

    print("Loading model...")
    model = SimCLREncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    return model, data, device

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

def visualize_pair(query_points, match_points, title):
    pcd_query = o3d.geometry.PointCloud()
    pcd_query.points = o3d.utility.Vector3dVector(query_points)
    pcd_query.paint_uniform_color([1, 0, 0]) # Red
    
    pcd_match = o3d.geometry.PointCloud()
    pcd_match.points = o3d.utility.Vector3dVector(match_points)
    pcd_match.paint_uniform_color([0, 1, 0]) # Green
    
    # Shift the match to the right
    width = np.max(query_points[:, 0]) - np.min(query_points[:, 0])
    pcd_match.translate((width * 1.5, 0, 0))
    
    print(f"Visualizing: {title}")
    o3d.visualization.draw_geometries([pcd_query, pcd_match], window_name=title)

def run_validation():
    model, data, device = load_model_and_data()
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
        
        # Get Top K (excluding self at rank 0)
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[1 : TOP_K+1] 
        
        for rank, match_idx in enumerate(top_indices):
            # Visualize
            title = f"Rank {rank+1} | Score: {similarities[match_idx]:.2f}"
            visualize_pair(data[query_idx], data[match_idx], title)
            
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
                "Dentist_Grade": grade
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