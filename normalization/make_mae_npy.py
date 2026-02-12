import os
import sys
import argparse
import subprocess
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# ==========================================
# 1. 環境與路徑設定
# ==========================================
# 強制設定 OpenMP 使用單執行緒，避免 C++ 函式庫打架 (非常重要！)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# GPU 設定 (Worker 模式會用到)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

current_folder = Path(__file__).parent.resolve()
project_root = current_folder.parent
SOURCE_FOLDER = project_root / "collecting-data" / "stlFiles"
OUTPUT_FILENAME = current_folder / "teeth3ds_mae_dataset.npy"
INDEX_FILENAME = current_folder / "teeth3ds_mae_filenames.json"
TEMP_DIR = current_folder / "temp_npy"

# ==========================================
# 2. WORKER 模式 (處理單一檔案)
# ==========================================
def run_worker(file_path_str, save_path_str):
    """
    這是被 subprocess 呼叫的函數。
    它是一個獨立的 Python 空間，就算這裡 SegFault 也不會影響主程式。
    """
    import trimesh
    import open3d as o3d
    import torch
    
    # 參數
    NUM_POINTS = 4096
    INITIAL_SAMPLE = 50000 
    
    try:
        # A. 讀取 Mesh
        mesh = trimesh.load(file_path_str, force='mesh')
        if mesh.is_empty: sys.exit(1) # Error
        
        mesh.vertices -= mesh.center_mass
        
        try:
            inertia = mesh.moment_inertia
            eigenvalues, eigenvectors = np.linalg.eigh(inertia)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = eigenvectors
            if np.all(np.isfinite(transform_matrix)):
                mesh.apply_transform(np.linalg.inv(transform_matrix))
        except: pass
        
        points_uniform = mesh.sample(INITIAL_SAMPLE)
        if np.isnan(points_uniform).any(): sys.exit(1)

        max_dist = np.max(np.linalg.norm(points_uniform, axis=1))
        if max_dist > 0:
            points_uniform /= max_dist
        
        # B. Normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_uniform)
        
        if len(pcd.points) < 10: sys.exit(1)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        normals = np.asarray(pcd.normals)
        if normals.shape[0] == 0 or np.isnan(normals).any(): sys.exit(1)
        
        normals = np.nan_to_num(normals)
        features = np.hstack([points_uniform, normals]).astype(np.float32)
        
        # C. GPU FPS
        final_data = None
        if torch.cuda.is_available():
            try:
                # 定義 FPS (簡化版，內嵌以免 import 問題)
                def fps_gpu(pts, npoint):
                    N, C = pts.shape
                    dev = pts.device
                    xyz = pts[:, :3]
                    centroids = torch.zeros((npoint, C), device=dev)
                    distance = torch.ones(N, device=dev) * 1e10
                    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=dev).item()
                    for i in range(npoint):
                        centroids[i] = pts[farthest]
                        c_xyz = xyz[farthest, :].view(1, 3)
                        dist = torch.sum((xyz - c_xyz) ** 2, -1)
                        mask = dist < distance
                        distance[mask] = dist[mask]
                        farthest = torch.argmax(distance).item()
                    return centroids

                with torch.no_grad():
                    pts_tensor = torch.from_numpy(features).float().cuda()
                    pts_out = fps_gpu(pts_tensor, NUM_POINTS)
                    final_data = pts_out.cpu().numpy()
            except: pass
        
        if final_data is None:
            # Fallback
            idx = np.random.choice(features.shape[0], NUM_POINTS, replace=False)
            final_data = features[idx, :]
            
        # 存檔
        np.save(save_path_str, final_data)
        sys.exit(0) # Success
        
    except Exception:
        sys.exit(1) # Fail

# ==========================================
# 3. MANAGER 模式 (主程式)
# ==========================================
def process_single_file_safe(args):
    """
    啟動一個子程序來跑 run_worker。
    如果子程序崩潰 (return code != 0)，這裡只會回傳 False。
    """
    idx, file_path = args
    save_path = TEMP_DIR / f"{idx}.npy"
    
    # 斷點續傳
    if save_path.exists():
        try:
            d = np.load(save_path)
            if d.shape == (4096, 6):
                return True, str(file_path.relative_to(SOURCE_FOLDER))
        except: pass

    # [關鍵] 呼叫自己 (python make_mae_npy_robust.py --worker ...)
    cmd = [
        sys.executable,  # 當前的 python執行檔
        str(Path(__file__).resolve()), # 這個腳本的路徑
        "--worker",
        "--input", str(file_path),
        "--output", str(save_path)
    ]
    
    try:
        # 執行子程序，並等待它完成
        # 如果崩潰，subprocess 會捕捉到 returncode != 0
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, str(file_path.relative_to(SOURCE_FOLDER))
        else:
            # 您可以取消註解下面這行來查看崩潰原因
            # print(f"\n[Fail] {file_path.name}: {result.stderr}")
            return False, None
    except Exception as e:
        return False, None

def main_manager():
    print(f"🚀 Robust Mode: Scanning '{SOURCE_FOLDER}'...")
    if not SOURCE_FOLDER.exists():
        print("Source folder not found.")
        return

    TEMP_DIR.mkdir(exist_ok=True)
    
    file_list = []
    for root, dirs, files in os.walk(str(SOURCE_FOLDER)):
        for file in files:
            if file.lower().endswith(('.obj', '.stl')):
                file_list.append(Path(root) / file)
    
    file_list.sort(key=lambda p: str(p))
    total_files = len(file_list)
    print(f"📄 Found {total_files} files.")
    
    # 準備任務列表
    tasks = [(i, fp) for i, fp in enumerate(file_list)]
    
    valid_indices = []
    valid_filenames = []
    
    # 使用 ThreadPoolExecutor 來管理 subprocess
    # 這裡可以用 15，因為 subprocess 是開新的 OS process，不受 GIL 限制
    # 且記憶體由 OS 管理，每個 process 跑完就釋放，非常乾淨
    MAX_WORKERS = 15 
    
    print(f"⚙️  Spawning subprocesses with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 這裡的 map 會按順序執行，但我們用 as_completed 比較好監控進度
        futures = {executor.submit(process_single_file_safe, task): task[0] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Robust Processing"):
            idx = futures[future]
            success, rel_path = future.result()
            if success:
                valid_indices.append(idx)
                valid_filenames.append(rel_path)

    # 合併階段
    print(f"\n📦 Merging {len(valid_indices)} files...")
    sorted_pairs = sorted(zip(valid_indices, valid_filenames))
    sorted_indices = [x[0] for x in sorted_pairs]
    final_filenames = [x[1] for x in sorted_pairs]
    
    if len(sorted_indices) > 0:
        all_data = np.zeros((len(sorted_indices), 4096, 6), dtype=np.float32)
        for i, idx in enumerate(tqdm(sorted_indices, desc="Merging")):
            all_data[i] = np.load(TEMP_DIR / f"{idx}.npy")
            
        np.save(OUTPUT_FILENAME, all_data)
        with open(INDEX_FILENAME, "w") as f:
            json.dump(final_filenames, f)
            
        print(f"✅ DONE! Saved to {OUTPUT_FILENAME}")
        
        # 清理
        import shutil
        shutil.rmtree(TEMP_DIR)
    else:
        print("❌ No valid files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run in worker mode")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    if args.worker:
        # 子程序模式：只處理一個檔案，然後結束
        run_worker(args.input, args.output)
    else:
        # 主程序模式
        main_manager()