# OASIS DINOv2 Validation Pipeline — Handoff Report

## Project Overview

OASIS is a dental scan similarity search system. It uses **DINOv2 (ViT-L/14, 1024-dim)** to embed multi-view rendered images of dental STL scans, then searches by cosine similarity. This validation pipeline measures search quality by having a dentist judge whether returned results are visually similar to query scans.

**Workspace root**: `/tmp2/b14902031/ytp_oasis`

---

## Pipeline Steps (4 total)

### Step 1: Pick 60 Test Scans → `validation/pick_test_scans.py`

**Purpose**: Randomly select 60 scans from the 641 DINOv2-eligible filenames. The dentist can review and swap individual picks before confirming.

**Current implementation**: Tkinter GUI (`ScanPicker` class) with a table showing selected scans and three buttons: Replace Selected, Reshuffle All, Confirm & Save.

**Known issue**: This script uses tkinter, which requires a graphical display (`$DISPLAY`). It **will fail over SSH** on a headless server with `_tkinter.TclError: no display name and no $DISPLAY environment variable`. It works fine on the user's M4 MacBook Pro (which has a display). If it needs to run on the remote server, it must be rewritten as a terminal-based CLI.

**Inputs**:
- `train/dinov2_filenames.json` — 641 STL filenames (format: `<UUID>_<serial>.stl`)

**Outputs**:
- `validation/validation_test_scans.json` — 60 selected test filenames (JSON array)
- `validation/validation_base_scans.json` — remaining ~581 base filenames (JSON array)

**How it works**:
1. Loads all 641 filenames from `train/dinov2_filenames.json`
2. Randomly picks 60 into `selected`, rest goes to `pool`
3. Tkinter GUI lets the dentist:
   - **Replace Selected**: swap highlighted rows with random files from pool
   - **Reshuffle All**: re-randomize all 60
   - **Confirm & Save**: write both JSON files and exit

---

### Step 2: Run Validation → `validation/validate_dinov2.py`

**Purpose**: For each of 4 top-k rounds (k=1, 3, 5, 10), iterate over all 60 test queries. For each query, compute cosine similarity against the ~581 base embeddings, retrieve top-k results, display the query mesh alongside result meshes in a 3D viewer, and let the dentist press **P** (Pass) or **F** (Fail).

**Implementation**: Uses `open3d.visualization.draw_geometries_with_key_callbacks` for macOS compatibility (avoids the `o3d.visualization.gui` module which has NSApplication threading issues on macOS).

**Inputs**:
- `validation/validation_test_scans.json` (from Step 1)
- `validation/validation_base_scans.json` (from Step 1)
- `train/dinov2_embeddings.npy` — shape `(641, 1024)`, float32, pre-computed DINOv2 embeddings
- `train/dinov2_filenames.json` — 641 filenames, index-aligned with the embeddings
- `collecting-data/stlFiles/` — directory containing the actual STL mesh files

**Outputs**:
- `validation/validation_progress.json` — incremental progress (saved after each verdict)
- `validation/validation_report.json` — final report (generated when all 4 rounds complete)

**How it works**:
1. Loads embeddings and splits them into test (60) and base (~581) subsets using the JSON files from Step 1
2. Loads or creates `validation_progress.json` for resume capability
3. For each k in `[1, 3, 5, 10]`:
   - For each of the 60 test queries:
     - Skip if already judged (resume support)
     - Compute cosine similarity of query embedding vs all base embeddings
     - Select top-k most similar base scans
     - Load STL meshes: query (gray `[0.8, 0.8, 0.8]`) at origin, results (light-blue `[0.6, 0.7, 0.9]`) offset below by `OFFSET_STEP=30.0` units each
     - All meshes are rotated -90° around X to stand teeth upright
     - Open 3D viewer window (1100×800px)
     - **P key** → pass, **F key** → fail; window auto-closes
     - Save verdict + result filenames + similarity scores to progress file
4. After all rounds, generate `validation_report.json`

**3D Viewer details**:
- Window title format: `Top-{k}  |  Query {num}/{total}  |  P=Pass  F=Fail`
- Both upper and lower case P/F are registered as key callbacks
- If window is closed without pressing P/F, defaults to "fail"
- Terminal also prints query filename, ranked result filenames, and similarity scores

---

### Step 3: Report Generation (automatic)

Happens automatically at the end of `validate_dinov2.py`. The report is also printed to the terminal as a summary table.

**Output format** (`validation/validation_report.json`):
```json
{
  "top_k_results": {
    "top_1": {
      "total_queries": 60,
      "passes": 45,
      "fails": 15,
      "pass_rate_percent": 75.0
    },
    "top_3": { ... },
    "top_5": { ... },
    "top_10": { ... }
  },
  "details": {
    "top_1": {
      "<filename>": {
        "verdict": "pass",
        "results": [
          {"filename": "...", "similarity": 0.9234}
        ]
      }
    }
  }
}
```

**Terminal summary format**:
```
============================================================
  OASIS DINOv2 Validation Report
============================================================
  Top-K      Queries    Pass     Fail     Rate
------------------------------------------------------------
  top_1      60         45       15       75.0%
  top_3      60         50       10       83.3%
  top_5      60         52       8        86.7%
  top_10     60         55       5        91.7%
============================================================
```

---

## Key Data Files

| File | Format | Description |
|------|--------|-------------|
| `train/dinov2_embeddings.npy` | numpy float32 `(641, 1024)` | Pre-computed DINOv2 ViT-L/14 embeddings for all scans |
| `train/dinov2_filenames.json` | JSON array of 641 strings | Filenames index-aligned with embeddings. Format: `<UUID>_<serial>.stl` |
| `collecting-data/stlFiles/` | Directory of `.stl` files | Raw dental scan meshes |
| `collecting-data/rendered_images/<UUID>/` | PNG images | Pre-rendered multi-view images (front, back, top, bottom, left, right) |

---

## DINOv2 Embedding Pipeline (for context)

Used by both the website search (`website/routes/search.py`) and pre-computed in `train/extract_dinov2.py`:

1. Load STL mesh with Open3D
2. Center mesh, color by vertex normals
3. Render 5 views (front, left, right, top, bottom) at 512×512px, FOV 60°, black background
4. Resize each view to 224×224, apply ImageNet normalization
5. Forward through `dinov2_vitl14` (PyTorch Hub) → 5 embeddings of dim 1024
6. Mean-pool across views → single 1024-dim vector
7. L2-normalize

---

## Environment & Dependencies

- **Python 3.12** (specifically `cpython-3.12.12-linux-x86_64-gnu` on the server)
- **Key packages**: `open3d` (0.19.0), `torch`, `torchvision`, `numpy`, `scikit-learn`, `Pillow`
- **Tkinter**: Required for `pick_test_scans.py` (needs X11 display)
- **Target machine**: M4 MacBook Pro (macOS) — the dentist runs the validation locally
- **Server**: Linux (headless, no `$DISPLAY`) — where embeddings/data live

---

## Execution Order

```bash
# Step 1: Pick test scans (run on Mac with display, or rewrite for CLI)
python validation/pick_test_scans.py

# Step 2: Run validation (run on Mac with display — needs Open3D 3D viewer)
python validation/validate_dinov2.py

# Report is generated automatically at the end of Step 2
# Output: validation/validation_report.json
```

---

## Known Issues & Notes for the Next Agent

1. **`pick_test_scans.py` uses tkinter** — fails on headless servers (no `$DISPLAY`). Works on Mac. If CLI version is needed, replace the `ScanPicker` class with a `input()`-based interactive loop (commands: `list`, `replace N`, `reshuffle`, `save`, `quit`).

2. **`validate_dinov2.py` uses `draw_geometries_with_key_callbacks`** — this was chosen over `o3d.visualization.gui.Application` because the GUI module has NSApplication threading issues on macOS (calling `app.run()` in a loop fails). The key-callback approach works reliably cross-platform.

3. **Resume support**: `validation_progress.json` stores verdicts incrementally. If the dentist stops mid-session, re-running `validate_dinov2.py` skips already-judged queries.

4. **STL files must be present** in `collecting-data/stlFiles/` for the 3D viewer to load meshes. If a file is missing, it prints a warning and skips that mesh.

5. **The test/base split is disjoint** — test scans are never searched against themselves. The cosine similarity is computed between the 60 test embeddings and the ~581 base embeddings only.

6. **Similarity metric**: Cosine similarity on L2-normalized 1024-dim DINOv2 embeddings. The same metric is used in the website search.

7. **Visual judgment criteria**: Subjective visual similarity — the dentist decides whether the retrieved dental scans "look similar enough" to the query. One verdict (Pass/Fail) per query per top-k round.
