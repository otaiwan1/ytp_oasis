"""
validate_dinov2.py — DINOv2 top-k validation with unified Open3D GUI windows.

For each top-k round (k=1, 3, 5, 10), iterates over 60 test queries.
Each query opens a single Open3D window showing the query mesh (gray) and
top-k result meshes (light-blue), with Pass / Fail buttons at the bottom.
Clicking a button records the verdict, closes the window, and advances.

Supports resuming interrupted sessions via validation_progress.json.
Generates validation_report.json when all rounds are complete.

Usage:
    python validation/validate_dinov2.py
"""

import os
os.environ['LP_NUM_THREADS'] = '8'

import json
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
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

# Rotation to stand the teeth up (-90° around X)
R_STAND_UP = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))


# ─── Helpers ─────────────────────────────────────────────────────────

def load_stl(filename, color):
    """Load an STL mesh, center it, and paint a uniform color."""
    stl_path = STL_DIR / filename
    if not stl_path.exists():
        print(f"  ⚠️  File not found: {filename}")
        return None
    try:
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        mesh.compute_vertex_normals()
        mesh.translate(-mesh.get_center())
        mesh.paint_uniform_color(color)
        return mesh
    except Exception as e:
        print(f"  ❌  Error loading {filename}: {e}")
        return None


def load_data():
    """Load embeddings, filenames, test/base splits, and build index maps."""
    # Full embedding database
    all_embeddings = np.load(EMB_PATH)
    with open(IDS_PATH, 'r') as f:
        all_filenames = json.load(f)
    fname_to_idx = {fn: i for i, fn in enumerate(all_filenames)}

    # Splits
    with open(TEST_SCANS, 'r') as f:
        test_fnames = json.load(f)
    with open(BASE_SCANS, 'r') as f:
        base_fnames = json.load(f)

    # Build index arrays
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


# ─── Open3D GUI Validation Window ────────────────────────────────────

class ValidationWindow:
    """
    A single Open3D GUI window showing query + top-k result meshes
    with Pass and Fail buttons. Clicking either records the verdict
    and closes the window.
    """

    def __init__(self, app, query_fname, result_fnames, result_sims,
                 top_k, query_num, total_queries):
        self.app = app
        self.verdict = None  # will be set to "pass" or "fail"

        # ── Build info strings ───────────────────────────────────────
        title = (f"OASIS Validation — Top-{top_k}  |  "
                 f"Query {query_num}/{total_queries}")
        info_lines = [
            f"Query:  {query_fname}",
            f"Top-{top_k} results:"
        ]
        for i, (fn, sim) in enumerate(zip(result_fnames, result_sims), 1):
            info_lines.append(f"  Rank {i}:  {fn}  (sim {sim:.4f})")

        # ── Create window ────────────────────────────────────────────
        self.window = app.create_window(title, 1100, 800)
        w = self.window

        # Theme / font
        em = w.theme.font_size

        # ── Layout: vertical stack ───────────────────────────────────
        # Top: info panel  |  Middle: 3D scene  |  Bottom: buttons
        panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em,
                                         0.5 * em, 0.5 * em))

        # Info labels
        info_panel = gui.Vert(0.15 * em, gui.Margins(0, 0, 0, 0))
        for line in info_lines:
            lbl = gui.Label(line)
            info_panel.add_child(lbl)
        panel.add_child(info_panel)

        # 3D scene
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(w.renderer)
        self.scene_widget.scene.set_background([0.15, 0.15, 0.15, 1.0])
        panel.add_child(self.scene_widget)

        # Button row
        btn_row = gui.Horiz(0.5 * em, gui.Margins(0, 0.25 * em, 0, 0))

        # Spacer to center the buttons
        btn_row.add_stretch()

        pass_btn = gui.Button("  ✅  Pass  ")
        pass_btn.background_color = gui.Color(0.2, 0.7, 0.3)
        pass_btn.set_on_clicked(self._on_pass)
        btn_row.add_child(pass_btn)

        fail_btn = gui.Button("  ❌  Fail  ")
        fail_btn.background_color = gui.Color(0.8, 0.25, 0.25)
        fail_btn.set_on_clicked(self._on_fail)
        btn_row.add_child(fail_btn)

        btn_row.add_stretch()

        panel.add_child(btn_row)

        w.add_child(panel)

        # ── Add meshes to the scene ──────────────────────────────────
        self._add_meshes(query_fname, result_fnames)

    def _add_meshes(self, query_fname, result_fnames):
        """Load and arrange meshes in the 3D scene."""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # Query mesh (index 0, top position)
        mesh = load_stl(query_fname, COLOR_QUERY)
        if mesh:
            mesh.rotate(R_STAND_UP, center=(0, 0, 0))
            mesh.translate((0, 0, 0))
            self.scene_widget.scene.add_geometry(
                f"query_{query_fname}", mesh, mat)

        # Result meshes (below the query)
        for i, fn in enumerate(result_fnames, 1):
            rmesh = load_stl(fn, COLOR_MATCH)
            if rmesh:
                rmesh.rotate(R_STAND_UP, center=(0, 0, 0))
                rmesh.translate((0, -i * OFFSET_STEP, 0))
                self.scene_widget.scene.add_geometry(
                    f"result_{i}_{fn}", rmesh, mat)

        # Fit camera to show all geometry
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _on_pass(self):
        self.verdict = "pass"
        self.window.close()

    def _on_fail(self):
        self.verdict = "fail"
        self.window.close()


# ─── Main validation loop ────────────────────────────────────────────

def run_validation():
    # Validate prerequisites
    for p, desc in [(TEST_SCANS, "validation_test_scans.json"),
                    (BASE_SCANS, "validation_base_scans.json"),
                    (EMB_PATH, "dinov2_embeddings.npy"),
                    (IDS_PATH, "dinov2_filenames.json")]:
        if not p.exists():
            print(f"❌  Missing {desc}. Path: {p}")
            sys.exit(1)

    # Load data
    print("🚀  Loading data...")
    test_fnames, base_fnames, test_embs, base_embs = load_data()
    print(f"    Test scans:  {len(test_fnames)}")
    print(f"    Base scans:  {len(base_fnames)}")

    # Load / init progress
    progress = load_progress()

    # Initialize Open3D application
    app = gui.Application.instance
    app.initialize()

    # Iterate over top-k rounds
    for k in TOP_K_VALUES:
        key = f"top_{k}"
        if key not in progress:
            progress[key] = {}

        done_count = len(progress[key])
        if done_count >= len(test_fnames):
            print(f"\n✅  Top-{k} already complete ({done_count}/{len(test_fnames)}).")
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
            for ri, (fn, sc) in enumerate(zip(result_fnames, result_sims), 1):
                print(f"    Rank {ri}: {fn}  (sim {sc:.4f})")

            # Create validation window
            vw = ValidationWindow(
                app, query_fname, result_fnames, result_sims,
                k, query_num, len(test_fnames))

            # Run the event loop until the window is closed
            app.run()

            # Record verdict
            verdict = vw.verdict or "fail"  # default to fail if window closed
            progress[key][query_fname] = {
                "verdict": verdict,
                "results": [
                    {"filename": fn, "similarity": sc}
                    for fn, sc in zip(result_fnames, result_sims)
                ]
            }
            save_progress(progress)

            symbol = "✅" if verdict == "pass" else "❌"
            print(f"    → {symbol} {verdict.upper()}")

        print(f"\n✅  Top-{k} round complete!")

    # Generate report
    print("\n📊  Generating report...")
    report = generate_report(progress, test_fnames)
    print_summary(report)


if __name__ == "__main__":
    run_validation()
