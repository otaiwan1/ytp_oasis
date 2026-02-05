import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import copy
import os
import threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import random
import sys
import trimesh
from pathlib import Path

# --- 1. SETUP PATHS ---
current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent

TRAIN_DIR = project_root / "train"
DATA_DIR = project_root / "normalization"

sys.path.append(str(TRAIN_DIR))

# --- 2. IMPORT MODEL ---
try:
    from train_oasis import SimCLREncoder
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit()

# --- 3. CONFIGURATION ---
# Base model path (original, never modified)
BASE_MODEL_PATH = TRAIN_DIR / "oasis_simclr_edgeconv.pth"
# Directory for fine-tuned model history
MODEL_HISTORY_DIR = current_folder / "model_history"
LATEST_MODEL_INFO = MODEL_HISTORY_DIR / "latest.txt"
# Directory for feedback history
FEEDBACK_HISTORY_DIR = current_folder / "feedback_history"

DATASET_PATH = DATA_DIR / "teeth3ds_dataset.npy"
INDEX_PATH = DATA_DIR / "teeth3ds_filenames.json"
REPORT_SAVE_PATH = current_folder / "feedback_report.csv"
VECTOR_CACHE_PATH = current_folder / "feedback_vectors_cache.npy"

# Reduced test cases for quick feedback
NUM_TEST_CASES = 7
TOP_K = 3


def get_latest_model_path():
    """
    Get the path to the latest fine-tuned model.
    If no fine-tuned model exists, return the base model path.
    """
    if LATEST_MODEL_INFO.exists():
        with open(LATEST_MODEL_INFO, 'r') as f:
            latest_path = Path(f.read().strip())
            if latest_path.exists():
                print(f"📦 Using LATEST fine-tuned model: {latest_path.name}")
                return latest_path
    print(f"📦 Using BASE model (no fine-tuned model found)")
    return BASE_MODEL_PATH


def save_feedback_with_history(results):
    """
    Save feedback results to both current file and history.
    """
    from datetime import datetime
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # 1. Save to current feedback_report.csv (append)
    if REPORT_SAVE_PATH.exists():
        existing = pd.read_csv(REPORT_SAVE_PATH)
        df_combined = pd.concat([existing, df], ignore_index=True)
    else:
        df_combined = df
    df_combined.to_csv(REPORT_SAVE_PATH, index=False)
    print(f"✓ Updated: {REPORT_SAVE_PATH}")
    
    # 2. Save to history with timestamp
    FEEDBACK_HISTORY_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = FEEDBACK_HISTORY_DIR / f"feedback_{timestamp}.csv"
    df.to_csv(history_path, index=False)
    print(f"✓ Saved to history: {history_path.name}")


# Get the model path to use
MODEL_PATH = get_latest_model_path()

class FeedbackApp:
    def __init__(self, cases, data_loader, on_complete):
        self.cases = cases
        self.data_loader = data_loader
        self.on_complete = on_complete
        self.current_idx = 0
        self.results = {}
        
        self.loader_cache = {}

        # Initialize App
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Oasis Feedback Tool", 1280, 720)
        
        # --- 3D Scene ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene_widget.scene.set_background([0.7, 0.7, 0.7, 1.0])
        self.scene_widget.scene.scene.enable_sun_light(True)
        self.scene_widget.scene.scene.set_sun_light([-0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 100000)
        
        # --- UI Panel ---
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # Status
        self.lbl_info = gui.Label("Initializing...")
        self.panel.add_child(self.lbl_info)
        self.panel.add_child(gui.Label(""))

        # Slider Setup
        self.panel.add_child(gui.Label("Similarity Score (1-10)"))
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(1, 10)
        self.slider.int_value = 5 # Default
        self.slider.set_on_value_changed(self.on_slider_change)
        self.panel.add_child(self.slider)
        
        # Score Display
        self.lbl_score = gui.Label("Score: 5 (Neutral)")
        self.lbl_score.text_color = gui.Color(0, 0, 0)
        self.panel.add_child(self.lbl_score)
        
        self.panel.add_child(gui.Label(""))
        
        # Navigation Buttons
        hlayout = gui.Horiz(0.25 * em)
        self.btn_prev = gui.Button("<< Prev")
        self.btn_prev.set_on_clicked(self.on_prev)
        self.btn_next = gui.Button("Next >>")
        self.btn_next.set_on_clicked(self.on_next)
        hlayout.add_child(self.btn_prev)
        hlayout.add_child(self.btn_next)
        self.panel.add_child(hlayout)

        # Layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        
        # Start
        self.load_current_case()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        width = 400 # Increased width
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - width, r.height)
        self.panel.frame = gui.Rect(r.width - width, r.y, width, r.height)
        
    def on_slider_change(self, val_float):
        val = int(val_float)
        if val <= 4:
            self.lbl_score.text = f"Score: {val} (Dissimilar)"
            self.lbl_score.text_color = gui.Color(1.0, 0.0, 0.0) # Red
        elif val >= 7:
            self.lbl_score.text = f"Score: {val} (Similar)"
            self.lbl_score.text_color = gui.Color(0.0, 0.6, 0.0) # Green
        else:
            self.lbl_score.text = f"Score: {val} (Neutral)"
            self.lbl_score.text_color = gui.Color(0.0, 0.0, 0.0) # Black

    def load_geometry_cached(self, obj):
        # 1. Handle Caching for Files (Strings)
        if isinstance(obj, str):
            if obj not in self.loader_cache:
                self.loader_cache[obj] = self.data_loader(obj)
            
            # Return a DEEP COPY to allow transformations without affecting cache
            cached = self.loader_cache[obj]
            if cached:
                return copy.deepcopy(cached)
            return None
            
        # 2. Handle Arrays (Fast enough, usually)
        return self.data_loader(obj)

    def load_current_case(self):
        if self.current_idx >= len(self.cases):
            self.window.close()
            final_list = [self.results[i] for i in range(len(self.cases)) if i in self.results]
            self.on_complete(final_list)
            return

        # 1. Update UI to "Loading" state IMMEDIATELY
        self.lbl_info.text = f"Loading Case {self.current_idx + 1}..."
        self.btn_next.enabled = False
        self.btn_prev.enabled = False
        
        # 2. Start Background Loader
        threading.Thread(target=self._load_case_background, args=(self.current_idx,)).start()

    def _load_case_background(self, idx):
        # runs in thread
        case = self.cases[idx]
        q_geom = self.load_geometry_cached(case['q_obj'])
        m_geom = self.load_geometry_cached(case['m_obj'])
        
        # Post back to Main Thread
        gui.Application.instance.post_to_main_thread(
            self.window, 
            lambda: self._on_case_loaded(idx, q_geom, m_geom)
        )

    def _on_case_loaded(self, idx, q_geom, m_geom):
        # runs on main thread
        
        # Restore saved score if exists, else 5
        saved_score = 5
        if idx in self.results:
            saved_score = self.results[idx]['Dentist_Grade']
            
        self.slider.int_value = saved_score
        self.on_slider_change(saved_score)
        
        # Clear scene
        self.scene_widget.scene.clear_geometry()
        
        mat = rendering.MaterialRecord()
        mat.base_color = [0.95, 0.95, 0.95, 1.0]
        mat.shader = "defaultLit"
        
        # Rotation -90 around X
        rot_t = np.eye(4)
        rot_t[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        # Query
        if q_geom:
            q_geom.transform(rot_t)
            self.scene_widget.scene.add_geometry("query", q_geom, mat)
            
        # Match
        if m_geom:
            m_geom.transform(rot_t)
            t = np.eye(4)
            t[1, 3] = -50.0 
            m_geom.transform(t)
            self.scene_widget.scene.add_geometry("match", m_geom, mat)
            
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        
        # Update UI Text Final
        self.lbl_info.text = f"Case {idx + 1}/{len(self.cases)}\nTop: Query\nBottom: Match"
        self.btn_next.enabled = True
        self.btn_prev.enabled = True

    def on_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_case()

    def on_next(self):
        # Save Result
        score = int(self.slider.int_value)
        case = self.cases[self.current_idx]
        
        self.results[self.current_idx] = {
            "Query_ID": case['q_id'],
            "Match_ID": case['m_id'],
            "Dentist_Grade": score,
            "Query_File": case['q_name'],
            "Match_File": case['m_name']
        }
        
        self.current_idx += 1
        self.load_current_case()
    
    def run(self):
        self.app.run()

# --- HELPERS ---
def load_data():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    data = np.load(DATASET_PATH)
    filenames = []
    if INDEX_PATH.exists():
        import json
        with open(INDEX_PATH,'r') as f: filenames=json.load(f)
        
    model = SimCLREncoder().to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, data, filenames, device

def geom_loader(data_obj):
    if isinstance(data_obj, str) and os.path.exists(data_obj):
        try:
            mesh = trimesh.load(data_obj, force='mesh')
            g = o3d.geometry.TriangleMesh()
            g.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            g.triangles = o3d.utility.Vector3iVector(mesh.faces)
            g.compute_vertex_normals()
            return g
        except: pass
    if isinstance(data_obj, np.ndarray):
        g = o3d.geometry.PointCloud()
        g.points = o3d.utility.Vector3dVector(data_obj)
        return g
    return None

def run():
    model, data, filenames, device = load_data()
    
    # Simple Random Sampling
    indices = random.sample(range(len(data)), min(NUM_TEST_CASES, len(data)))
    cases = []
    
    # Just skip vector calc for now to test UI if cache missing, 
    # but re-calc is fast enough for small batches. 
    # Let's do simple neighbor search on the fly to avoid complexity
    
    print("Computing vectors for random subset...")
    subset = torch.tensor(data[indices], dtype=torch.float32).transpose(2,1).to(device)
    with torch.no_grad():
        sub_vecs, _ = model(subset)
    sub_vecs = sub_vecs.cpu().numpy()
    sub_vecs = sub_vecs / np.linalg.norm(sub_vecs, axis=1, keepdims=True)
    
    # For each in subset, find a match in the full dataset? 
    # That's slow. Let's just create pairs from the random sample for UI test
    # Or load cache.
    
    # LOAD CACHE (Previous logic was better, restoring simplified version)
    full_vecs = None
    if VECTOR_CACHE_PATH.exists():
        full_vecs = np.load(VECTOR_CACHE_PATH)
    else:
        # Quick compute for ALL (needed for meaningful matches)
        print("Computing all vectors...")
        vecs = []
        bs = 32
        with torch.no_grad():
            for i in range(0, len(data), bs):
                b = torch.tensor(data[i:i+bs], dtype=torch.float32).transpose(2,1).to(device)
                v, _ = model(b)
                vecs.append(v.cpu().numpy())
        full_vecs = np.vstack(vecs)
        np.save(VECTOR_CACHE_PATH, full_vecs)
        
    full_vecs = full_vecs / np.linalg.norm(full_vecs, axis=1, keepdims=True)
    
    # Find Matches
    for q_idx in indices:
        q_vec = full_vecs[q_idx]
        sims = np.dot(full_vecs, q_vec)
        # Top 3 excluding self
        top_ids = np.argsort(sims)[::-1]
        top_ids = [x for x in top_ids if x != q_idx][:TOP_K]
        
        for m_idx in top_ids:
            q_name = filenames[q_idx] if filenames else f"ID_{q_idx}"
            m_name = filenames[m_idx] if filenames else f"ID_{m_idx}"
            
            cases.append({
                "q_id": q_idx, "m_id": m_idx,
                "q_obj": q_name, "m_obj": m_name, # Pass name if string, else array?
                # Actually geom_loader handles filenames strings. 
                # If filenames is empty, we need to pass data array
                "q_name": os.path.basename(str(q_name)),
                "m_name": os.path.basename(str(m_name))
            })
            # Fix obj reference logic
            cases[-1]['q_obj'] = q_name if filenames else data[q_idx]
            cases[-1]['m_obj'] = m_name if filenames else data[m_idx]

    app = FeedbackApp(cases, geom_loader, save_feedback_with_history)
    app.run()

if __name__ == "__main__":
    run()
