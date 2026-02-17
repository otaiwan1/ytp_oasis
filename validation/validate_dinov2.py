"""
validate_dinov2.py — DINOv2 top-k validation with Open3D 3D viewer.

For each top-k round (k=1, 3, 5, 10), iterates over 60 test queries.
Each query opens an Open3D window showing the query mesh (gray, top) and
top-k result meshes (light-blue, below).  The dentist inspects the 3D
scene and presses:

    P  →  Pass  (results look clinically similar)
    F  →  Fail  (results are not similar enough)

The window closes automatically and the next query opens.

Supports resuming interrupted sessions via validation_progress.json.
Generates validation_report.json when all rounds are complete.

Usage:
    python validation/validate_dinov2.py
"""

import json
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ─── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

STL_DIR = PROJECT_ROOT / "collecting-data" / "stlFiles"
EMB_PATH = PROJECT_ROOT / "train" / "dinov2_embeddings.npy"
IDS_PATH = PROJECT_ROOT / "train" / "dinov2_filenames.json"

TEST_SCANS = SCRIPT_DIR / "validation_test_scans.json"
BASE_SCANS = SCRIPT_DIR / "validation_base_scans.json"
PROGRESS_FILE = SCRIPT_DIR / "validation_progress.json"
REPORT_FILE = SCRIPT_DIR / "validation_report.json"

# ─── Visual settings ────────────────────────────────────────────────
OFFSET_STEP = 30.0
COLOR_QUERY = [0.8, 0.8, 0.8]   # gray-white
COLOR_MATCH = [0.6, 0.7, 0.9]   # light-blue
TOP_K_VALUES = [1, 3, 5, 10]

# Rotation to stand the teeth up (-90 deg around X)
R_STAND_UP = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))


# ─── Helpers ─────────────────────────────────────────────────────────

def load_stl(filename, color):
    """Load an STL mesh, center it, and paint a uniform color."""
    stl_path = STL_DIR / filename
    if not stl_path.exists():
        print(f"  Warning: File not found: {filename}")
        return None
    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        mesh.paint_uniform_color(color)
        return mesh
    except Exception as e:
        print(f"  Error loading {filename}: {e}")
        return None


def load_data():
    """Load embeddings, filenames, test/base splits, and build index maps."""
    all_embeddings = np.load(EMB_PATH)
    with open(IDS_PATH, 'r') as f:
        all_filenames = json.load(f)
    fname_to_idx = {fn: i for i, fn in enumerate(all_filenames)}

    with open(TEST_SCANS, 'r') as f:
        test_fnames = json.load(f)
    with open(BASE_SCANS, 'r') as f:
        base_fnames = json.load(f)

    test_indices = [fname_to_idx[fn] for fn in test_fnames if fn in fname_to_idx]
    base_indices = [fname_to_idx[fn] for fn in base_fnames if fn in fname_to_idx]

    test_fnames_valid = [all_filenames[i] for i in test_indices]
    base_fnames_valid = [all_filenames[i] for i in base_indices]

    test_embs = all_embeddings[test_indices]
    base_embs = all_embeddings[base_indices]

    return test_fnames_valid, base_fnames_valid, test_embs, base_embs


def load_progress():
    """Load progress from disk, or return empty structure."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Persist progress to disk."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def generate_report(progress, test_fnames):
    """Generate the final validation report from completed progress."""
    report = {"top_k_results": {}, "details": {}}

    for k in TOP_K_VALUES:
        key = f"top_{k}"
        if key not in progress:
            continue
        verdicts = progress[key]
        total = len(verdicts)
        passes = sum(1 for v in verdicts.values() if v["verdict"] == "pass")
        fails = total - passes
        rate = (passes / total * 100) if total > 0 else 0.0

        report["top_k_results"][key] = {
            "total_queries": total,
            "passes": passes,
            "fails": fails,
            "pass_rate_percent": round(rate, 2)
        }
        report["details"][key] = verdicts

    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def print_summary(report):
    """Print a summary table to the terminal."""
    print("\n" + "=" * 60)
    print("  OASIS DINOv2 Validation Report")
    print("=" * 60)
    print(f"  {'Top-K':<10} {'Queries':<10} {'Pass':<8} {'Fail':<8} {'Rate':<10}")
    print("-" * 60)
    for k in TOP_K_VALUES:
        key = f"top_{k}"
        r = report["top_k_results"].get(key, {})
        if not r:
            continue
        print(f"  {key:<10} {r['total_queries']:<10} {r['passes']:<8} "
              f"{r['fails']:<8} {r['pass_rate_percent']:.1f}%")
    print("=" * 60)
    print(f"  Report saved to: {REPORT_FILE}")
    print()


# ─── 3D Viewer with key-based verdict ───────────────────────────────

def show_and_judge(geometries, title):
    """
    Open an Open3D window with the given geometries.
    The user presses P (pass) or F (fail) to record a verdict.
    The window closes automatically after the key press.

    Returns "pass", "fail", or None (if the window was closed without
    pressing P or F, treated as "fail" by the caller).
    """
    verdict = [None]  # mutable container for closure

    def on_pass(vis):
        verdict[0] = "pass"
        vis.destroy_window()
        return False

    def on_fail(vis):
        verdict[0] = "fail"
        vis.destroy_window()
        return False

    # Register both upper and lower case
    key_callbacks = {
        ord('P'): on_pass,
        ord('p'): on_pass,
        ord('F'): on_fail,
        ord('f'): on_fail,
    }

    o3d.visualization.draw_geometries_with_key_callbacks(
        geometries, key_callbacks,
        window_name=title,
        width=1100, height=800,
        left=50, top=50)

    return verdict[0]


# ─── Main validation loop ────────────────────────────────────────────

def run_validation():
    # Validate prerequisites
    for p, desc in [(TEST_SCANS, "validation_test_scans.json"),
                    (BASE_SCANS, "validation_base_scans.json"),
                    (EMB_PATH, "dinov2_embeddings.npy"),
                    (IDS_PATH, "dinov2_filenames.json")]:
        if not p.exists():
            print(f"Missing {desc}. Path: {p}")
            sys.exit(1)

    # Load data
    print("Loading data...")
    test_fnames, base_fnames, test_embs, base_embs = load_data()
    print(f"    Test scans:  {len(test_fnames)}")
    print(f"    Base scans:  {len(base_fnames)}")

    # Load / init progress
    progress = load_progress()

    print("\n" + "-" * 50)
    print("  Controls:  P = Pass   |   F = Fail")
    print("  (press while the 3D window is focused)")
    print("-" * 50)

    # Iterate over top-k rounds
    for k in TOP_K_VALUES:
        key = f"top_{k}"
        if key not in progress:
            progress[key] = {}

        done_count = len(progress[key])
        if done_count >= len(test_fnames):
            print(f"\n  Top-{k} already complete "
                  f"({done_count}/{len(test_fnames)}).")
            continue

        print(f"\n{'='*50}")
        print(f"  Starting Top-{k} validation  "
              f"({done_count}/{len(test_fnames)} done)")
        print(f"{'='*50}")

        for qi, query_fname in enumerate(test_fnames):
            # Skip already-judged queries
            if query_fname in progress[key]:
                continue

            query_num = qi + 1

            # Compute similarities
            query_vec = test_embs[qi].reshape(1, -1)
            sims = cosine_similarity(query_vec, base_embs)[0]

            top_idx = np.argsort(sims)[::-1][:k]
            result_fnames = [base_fnames[i] for i in top_idx]
            result_sims = [float(sims[i]) for i in top_idx]

            # Print to terminal
            print(f"\n  [{key}] Query {query_num}/{len(test_fnames)}: "
                  f"{query_fname}")
            for ri, (fn, sc) in enumerate(
                    zip(result_fnames, result_sims), 1):
                print(f"    Rank {ri}: {fn}  (sim {sc:.4f})")
            print("    >> Press P (pass) or F (fail) in the 3D window")

            # Build geometries
            geometries = []

            # Query mesh (gray, top position)
            qmesh = load_stl(query_fname, COLOR_QUERY)
            if qmesh:
                qmesh.rotate(R_STAND_UP, center=(0, 0, 0))
                geometries.append(qmesh)

            # Result meshes (light-blue, spaced below)
            for i, fn in enumerate(result_fnames, 1):
                rmesh = load_stl(fn, COLOR_MATCH)
                if rmesh:
                    rmesh.rotate(R_STAND_UP, center=(0, 0, 0))
                    rmesh.translate((0, -i * OFFSET_STEP, 0))
                    geometries.append(rmesh)

            if not geometries:
                print("    Warning: No meshes loaded, skipping.")
                continue

            # Show 3D viewer -- blocks until P or F is pressed
            title = (f"Top-{k}  |  Query {query_num}/{len(test_fnames)}"
                     f"  |  P=Pass  F=Fail")
            verdict = show_and_judge(geometries, title)

            # Default to fail if window closed without pressing P/F
            if verdict is None:
                verdict = "fail"

            progress[key][query_fname] = {
                "verdict": verdict,
                "results": [
                    {"filename": fn, "similarity": sc}
                    for fn, sc in zip(result_fnames, result_sims)
                ]
            }
            save_progress(progress)

            symbol = "PASS" if verdict == "pass" else "FAIL"
            print(f"    -> {symbol}")

        print(f"\n  Top-{k} round complete!")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(progress, test_fnames)
    print_summary(report)


if __name__ == "__main__":
    run_validation()
