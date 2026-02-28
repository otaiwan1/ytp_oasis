# OASIS — Project Architecture & Flowchart Report

> **OASIS**: Orthodontic AI Similarity Intelligence System  
> **Generated**: 2026-02-28  
> **Scope**: Full system analysis covering data pipelines, model training, embedding infrastructure, web application, and validation workflows.

---

## 1. Executive Summary

OASIS is an end-to-end dental scan similarity search system. It ingests 3D dental STL scans, processes them through multiple AI embedding models, and provides a web-based interface where dentists can upload a scan and find the most visually similar cases in the database using cosine similarity retrieval.

The system supports **6 embedding strategies** across two modalities:
- **3D Point Cloud models**: SimCLR (512-d), Point-MAE (384-d)
- **Multi-view Image models**: DINOv2 (1024-d), DINOv3 (1024-d), DINOv3 Gallery (2048-d), DINOv3 Concat (10240-d)

---

## 2. System Architecture Overview

```mermaid
graph TB
    subgraph DATA["📁 Data Collection & Ingestion"]
        STL["STL Dental Scan Files<br/>(collecting-data/stlFiles/)"]
        SCAN["iTero Scan Zips<br/>(collecting-data/scanfiles/)"]
        RENDER_SCRIPT["render_multiview_final.py<br/>Multi-view Rendering (6 views)"]
        POOL["run_pool_render.py<br/>Parallel Pool (8 workers, GPU)"]
        RENDERED["Rendered PNG Images<br/>(collecting-data/rendered_images/)"]
        CONVERT["convert_stl_to_glb.py<br/>STL → GLB (97% compression)"]
        GLB["GLB Files for Web Viewer<br/>(collecting-data/glbFiles/)"]
    end

    subgraph NORM["⚙️ Normalization & Point Cloud Processing"]
        MAKE_NPY["make_npy.py<br/>SimCLR Dataset Builder"]
        MAKE_MAE["make_mae_npy.py<br/>MAE Dataset Builder"]
        SIMCLR_DS["simclr_dataset.npy<br/>(N, 4096, 3) XYZ"]
        MAE_DS["mae_dataset.npy<br/>(N, 4096, 6) XYZ+Normal"]
    end

    subgraph TRAIN["🧠 Model Training"]
        SIMCLR_TRAIN["train_oasis.py<br/>SimCLR (EdgeConv, NT-Xent)<br/>Multi-GPU, 300 epochs"]
        MAE_TRAIN["train_mae_ddp.py<br/>Point-MAE (Transformer)<br/>DDP, 8000 epochs"]
        DINO_PRETRAINED["DINOv2 / DINOv3<br/>Pre-trained ViT-L<br/>(PyTorch Hub)"]
        SIMCLR_CKP["best_dental_simclr.pth<br/>512-dim backbone"]
        MAE_CKP["best_point_mae_ddp.pth<br/>384-dim embedding"]
    end

    subgraph EMBED["🔗 Unified Embed Module"]
        EMBED_API["embed/__init__.py<br/>embed_stl() API"]
        PREPROC["preprocessing.py<br/>SimCLR / MAE / DINOv2-v3"]
        MODELS["models.py<br/>Inference & Caching"]
        BATCH["batch_embed.py<br/>CLI Batch Embedding"]
    end

    subgraph OUTPUTS["📊 Embedding Outputs"]
        EMB_S["simclr_embeddings.npy (512-d)"]
        EMB_M["mae_embeddings.npy (384-d)"]
        EMB_D2["dinov2_embeddings.npy (1024-d)"]
        EMB_D3["dinov3_embeddings.npy (1024-d)"]
        EMB_D3G["dinov3_gallery_emb.npy (2048-d)"]
        EMB_D3C["dinov3_concat_emb.npy (10240-d)"]
    end

    subgraph VALID["✅ Validation Pipeline"]
        PICK["pick_test_scans.py<br/>60 test / ~581 base split"]
        VALIDATE["validate_dinov2.py<br/>3D Viewer (Open3D)<br/>Pass/Fail by Dentist"]
        WEB_VALID["Website Validation<br/>/validation/judge"]
        REPORT["validation_report.json<br/>Top-K Accuracy"]
    end

    subgraph WEB["🌐 Flask Web Application"]
        APP["app.py (Flask)"]
        AUTH["Auth (Login/Register)"]
        SEARCH["Search (Upload STL → Similar)"]
        UPLOAD["Upload (New Scans)"]
        COLLECTION["Collection (Browse Patients)"]
        DB["SQLite Database<br/>(oasis.db)"]
    end

    STL --> RENDER_SCRIPT
    RENDER_SCRIPT --> POOL
    POOL --> RENDERED
    STL --> CONVERT
    CONVERT --> GLB

    STL --> MAKE_NPY
    STL --> MAKE_MAE
    MAKE_NPY --> SIMCLR_DS
    MAKE_MAE --> MAE_DS

    SIMCLR_DS --> SIMCLR_TRAIN
    MAE_DS --> MAE_TRAIN
    SIMCLR_TRAIN --> SIMCLR_CKP
    MAE_TRAIN --> MAE_CKP

    SIMCLR_CKP --> MODELS
    MAE_CKP --> MODELS
    DINO_PRETRAINED --> MODELS
    PREPROC --> EMBED_API
    MODELS --> EMBED_API
    EMBED_API --> BATCH

    BATCH --> EMB_S
    BATCH --> EMB_M
    BATCH --> EMB_D2
    BATCH --> EMB_D3
    BATCH --> EMB_D3G
    BATCH --> EMB_D3C

    EMB_D2 --> PICK
    PICK --> VALIDATE
    PICK --> WEB_VALID
    VALIDATE --> REPORT
    WEB_VALID --> REPORT
    EMB_S --> WEB_VALID
    EMB_M --> WEB_VALID
    EMB_D3 --> WEB_VALID

    EMB_D2 --> SEARCH
    EMBED_API --> SEARCH
    APP --> AUTH
    APP --> SEARCH
    APP --> UPLOAD
    APP --> COLLECTION
    APP --> WEB_VALID
    AUTH --> DB
    GLB --> WEB
```

---

## 3. End-to-End Pipeline Stages

```mermaid
flowchart LR
    subgraph STEP1["STAGE 1: Data Pipeline"]
        direction TB
        A1["Raw STL Scans<br/>+ iTero Zips"] --> A2["Multi-view Rendering<br/>(Open3D, 6 views, 512px)"]
        A1 --> A3["Point Cloud Sampling<br/>(GPU FPS, 4096 pts)"]
        A1 --> A4["STL → GLB Compression<br/>(Trimesh, 97% reduction)"]
        A2 --> A5["rendered_images/UUID/"]
        A3 --> A6["simclr_dataset.npy<br/>(N, 4096, 3)"]
        A3 --> A7["mae_dataset.npy<br/>(N, 4096, 6)"]
        A4 --> A8["glbFiles/*.glb"]
    end

    subgraph STEP2["STAGE 2: Training"]
        direction TB
        B1["SimCLR<br/>(EdgeConv + NT-Xent Loss)<br/>Multi-GPU, 512-dim"]
        B2["Point-MAE<br/>(Masked Autoencoder)<br/>DDP 4×GPU, 384-dim"]
        B3["DINOv2 ViT-L/14<br/>(Pre-trained, 1024-dim)<br/>Facebook Hub"]
        B4["DINOv3 ViT-L/16<br/>(Pre-trained, 1024-dim)<br/>Gram Anchoring"]
    end

    subgraph STEP3["STAGE 3: Batch Embedding"]
        direction TB
        C1["batch_embed.py<br/>--model name<br/>--gpu N --workers K"]
        C1 --> C2["6 Embedding Sets<br/>SimCLR / MAE / DINOv2<br/>DINOv3 / Gallery / Concat"]
    end

    subgraph STEP4["STAGE 4: Serve & Validate"]
        direction TB
        D1["Flask Web App<br/>:36368"]
        D1 --> D2["Search: Upload STL<br/>→ Cosine Similarity<br/>→ Top-K Results"]
        D1 --> D3["Validate: Dentist<br/>Judges Pass/Fail<br/>per Query × Top-K"]
        D3 --> D4["Report:<br/>Top-1/3/5/10<br/>Accuracy %"]
    end

    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
```

---

## 4. Module Breakdown

### 4.1 Data Collection (`collecting-data/`)

| Component | File | Purpose |
|-----------|------|---------|
| Multi-view Renderer | `render_multiview_final.py` | Renders each STL from 6 viewpoints (front, back, top, bottom, left, right) at 512×512px using Open3D offscreen renderer with EGL |
| Parallel Runner | `run_pool_render.py` | Orchestrates rendering with 8 worker processes on a designated GPU |
| STL → GLB Converter | `tools/convert_stl_to_glb.py` | Simplifies meshes to 15% faces + gzip compresses for web viewing (97% size reduction) |

### 4.2 Normalization (`normalization/`)

| Component | File | Output | Pipeline |
|-----------|------|--------|----------|
| SimCLR Dataset | `make_npy.py` | `simclr_dataset.npy` (N, 4096, 3) | Load STL → Center → PCA-align → Uniform sample 100K → GPU FPS → 4096 XYZ points |
| MAE Dataset | `make_mae_npy.py` | `mae_dataset.npy` (N, 4096, 6) | Load STL → Center → PCA-align → Sample 50K → Unit-sphere normalize → Estimate normals → Concat XYZ+Normal → GPU FPS → 4096 points |

### 4.3 Training (`train/`)

| Model | File | Architecture | Embedding Dim | Training Strategy |
|-------|------|-------------|---------------|-------------------|
| **SimCLR** | `train_oasis.py` | EdgeConv (DGCNN) + Projection Head | 512 | NT-Xent contrastive loss, 3 GPUs, batch 36, 300 epochs |
| **Point-MAE** | `train_mae_ddp.py` | Patch Embed (FPS+KNN) + Transformer | 384 | Masked autoencoder (75% mask), DDP 4 GPUs, batch 64/GPU, 8000 epochs |
| **DINOv2** | Pre-trained | ViT-L/14 | 1024 | Facebook's self-supervised (PyTorch Hub) |
| **DINOv3** | Pre-trained | ViT-L/16 | 1024 | Gram Anchoring (local checkpoint) |

### 4.4 Unified Embed Module (`embed/`)

| Component | File | Role |
|-----------|------|------|
| Public API | `__init__.py` | `embed_stl(stl_path, model)` → `{"embedding", "filename", "model", "dim"}` |
| Configuration | `config.py` | Centralized paths, model settings, rendering parameters |
| Preprocessing | `preprocessing.py` | Model-specific data prep: SimCLR (point cloud), MAE (point cloud + normals), DINOv2/v3 (multi-view rendering) |
| Model Inference | `models.py` | Lazy-loaded model cache, inference functions for all 6 model variants |
| Batch CLI | `batch_embed.py` | CLI to embed all STL files: `python -m embed.batch_embed --model dinov3 --gpu 1 --workers 12` |

### 4.5 Web Application (`website/`)

| Blueprint | Route Prefix | Functionality |
|-----------|-------------|---------------|
| **Auth** | `/auth` | User registration, login, logout (Flask-Login + SQLAlchemy) |
| **Main** | `/` | Home page, about page |
| **Search** | `/search` | Upload STL → real-time DINOv2 embedding → cosine similarity search → top-K results |
| **Upload** | `/upload` | Upload new patient STL scans (auto-naming: `{patient_uid}_{serial}.stl`) |
| **Collection** | `/collection` | Browse all patients, view rendered images by patient UUID |
| **Validation** | `/validation` | Multi-model validation UI — dentist selects model, judges top-K results as Pass/Fail |

### 4.6 Validation (`validation/`)

| Component | File | Purpose |
|-----------|------|---------|
| Test Scan Picker | `pick_test_scans.py` | Interactive CLI to select 60 test scans from 641 eligible (rest become base) |
| Desktop Validator | `validate_dinov2.py` | Open3D 3D viewer — dentist presses P/F per query, resume-capable |
| Web Validator | `website/routes/validation.py` | Browser-based validation for all 6 models, with progress tracking |

---

## 5. Embedding & Search Pipeline Detail

```mermaid
flowchart TD
    START(["User Uploads .STL File"]) --> SAVE["Save to temp path"]
    SAVE --> DETECT{"Model?"}

    DETECT -->|"DINOv2 / DINOv3"| RENDER["Load STL with Open3D<br/>Center mesh, color by normals"]
    DETECT -->|"SimCLR"| SIMCLR_PRE["Load STL with Trimesh<br/>Center, PCA-align<br/>Uniform sample 100K pts"]
    DETECT -->|"Point-MAE"| MAE_PRE["Load STL with Trimesh<br/>Center, PCA-align<br/>Sample 50K, unit-sphere norm<br/>Estimate normals (Open3D)"]

    RENDER --> VIEWS["Render 5 views:<br/>front, left, right, top, bottom<br/>512×512px, FOV 60°, black BG"]
    VIEWS --> RESIZE["Resize to 224×224 (DINOv2)<br/>or 256×256 (DINOv3)"]
    RESIZE --> IMGNET["ImageNet normalization"]
    IMGNET --> VIT["Forward through ViT-L<br/>(5 images → 5 × 1024-dim)"]
    VIT --> POOL["Mean-pool → 1024-dim"]
    POOL --> L2["L2-normalize"]

    SIMCLR_PRE --> FPS_3["GPU Farthest Point Sampling<br/>→ (4096, 3)"]
    FPS_3 --> EDGE["Forward through EdgeConv backbone<br/>→ 512-dim"]
    EDGE --> L2_S["L2-normalize"]

    MAE_PRE --> CONCAT_N["Concat XYZ + Normals → (50K, 6)"]
    CONCAT_N --> FPS_6["GPU FPS on 6-channel<br/>→ (4096, 6)"]
    FPS_6 --> PATCH["Patch Embed (FPS + KNN)<br/>128 patches × 32 points"]
    PATCH --> TRANS["Forward through Transformer<br/>→ 384-dim"]

    L2 --> COSINE["Cosine Similarity<br/>vs. Pre-computed DB"]
    L2_S --> COSINE
    TRANS --> COSINE

    COSINE --> TOPK["Sort & Select Top-K"]
    TOPK --> RESULT(["Return Ranked Results<br/>+ Similarity Scores"])
```

---

## 6. Web Application Architecture

```mermaid
flowchart TD
    subgraph FLASK["Flask App (app.py :36368)"]
        direction TB
        INIT["create_app()<br/>SQLAlchemy + LoginManager"]

        subgraph BP["Blueprints"]
            direction LR
            AUTH_BP["/auth<br/>Login / Register / Logout"]
            MAIN_BP["/<br/>Home / About"]
            SEARCH_BP["/search<br/>STL Upload → Similar Scans"]
            UPLOAD_BP["/upload<br/>Add New Patient Scans"]
            COLL_BP["/collection<br/>Browse All Patients"]
            VALID_BP["/validation<br/>Model Validation Judge"]
        end
    end

    subgraph DB_LAYER["Database Layer"]
        SQLITE["SQLite (oasis.db)"]
        USER_MODEL["User Model<br/>id, username, email, role<br/>password_hash, created_at"]
        EMB_CACHE["Embeddings Cache<br/>embeddings_cache.npy<br/>filenames_cache.json"]
    end

    subgraph EMBED_LAYER["Embed Module (Backend)"]
        EMBED_STL["embed_stl(stl_path, model)"]
        LOAD_EMB["load_embeddings_db()<br/>np.load() cached .npy"]
    end

    INIT --> BP
    AUTH_BP --> SQLITE
    SQLITE --> USER_MODEL
    SEARCH_BP -->|"User uploads .STL"| EMBED_STL
    EMBED_STL -->|"Query embedding"| SEARCH_BP
    SEARCH_BP -->|"Load pre-computed DB"| LOAD_EMB
    LOAD_EMB --> EMB_CACHE
    UPLOAD_BP -->|"Save to stlFiles/"| SQLITE
    COLL_BP -->|"Browse stlFiles/<br/>+ rendered_images/"| SQLITE
    VALID_BP -->|"Load model embeddings<br/>Compute similarity<br/>Record Pass/Fail"| LOAD_EMB
```

---

## 7. Validation Pipeline Flow

```mermaid
flowchart TD
    START(["Start Validation"]) --> SELECT["Select Model<br/>(DINOv2 / SimCLR / MAE / DINOv3 / ...)"]
    SELECT --> PICK["pick_test_scans.py<br/>CLI: Select 60 random test scans"]
    PICK --> TEST_JSON["validation_test_scans.json<br/>(60 test filenames)"]
    PICK --> BASE_JSON["validation_base_scans.json<br/>(~581 base filenames)"]

    TEST_JSON --> LOAD["Load model embeddings<br/>Split into test & base subsets"]
    BASE_JSON --> LOAD

    LOAD --> K1{"Round: Top-1"}
    K1 --> LOOP1["For each of 60 queries:<br/>1. Compute cosine similarity vs base<br/>2. Retrieve top-1 result<br/>3. Dentist judges Pass/Fail"]
    LOOP1 --> SAVE1["Save verdict to<br/>progress_model.json"]

    SAVE1 --> K3{"Round: Top-3"}
    K3 --> LOOP3["Repeat for top-3"]
    LOOP3 --> SAVE3["Save progress"]

    SAVE3 --> K5{"Round: Top-5"}
    K5 --> LOOP5["Repeat for top-5"]
    LOOP5 --> SAVE5["Save progress"]

    SAVE5 --> K10{"Round: Top-10"}
    K10 --> LOOP10["Repeat for top-10"]
    LOOP10 --> SAVE10["Save progress"]

    SAVE10 --> REPORT["Generate Report<br/>validation_report_model.json"]
    REPORT --> SUMMARY(["Summary Table:<br/>Top-K → Queries → Pass → Fail → Rate%"])
```

---

## 8. Key Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Ingest** | Raw `.stl` files | Download from iTero / manual upload | `collecting-data/stlFiles/` |
| **Render** | `.stl` mesh | Open3D offscreen, 6 views × 512px | `rendered_images/<UUID>/*.png` |
| **Compress** | `.stl` mesh | Trimesh simplify + gzip GLB | `glbFiles/<name>.glb` |
| **Normalize (SimCLR)** | `.stl` mesh | Center → PCA → Sample → FPS | `simclr_dataset.npy` (N, 4096, 3) |
| **Normalize (MAE)** | `.stl` mesh | Center → PCA → Sample → Normals → FPS | `mae_dataset.npy` (N, 4096, 6) |
| **Train SimCLR** | Point cloud dataset | EdgeConv + NT-Xent, multi-GPU | `best_dental_simclr.pth` |
| **Train MAE** | Point cloud + normals | Masked Autoencoder, DDP | `best_point_mae_ddp.pth` |
| **Batch Embed** | STL files + models | `embed_stl()` per file | `{model}_embeddings.npy` + `{model}_filenames.json` |
| **Search** | User upload `.stl` | Embed → cosine sim vs DB | Top-K ranked results |
| **Validate** | 60 test scans | Per-model top-K judgments | `validation_report_{model}.json` |

---

## 9. Technology Stack

| Layer | Technologies |
|-------|-------------|
| **3D Processing** | Open3D (rendering, normals), Trimesh (mesh I/O, simplification) |
| **Deep Learning** | PyTorch, DDP, Mixed Precision (AMP), CUDA |
| **Models** | DGCNN/EdgeConv (SimCLR), Transformer (Point-MAE), ViT-L (DINOv2/v3) |
| **Web Framework** | Flask, Flask-Login, Flask-SQLAlchemy |
| **Database** | SQLite |
| **Data Format** | NumPy `.npy` (embeddings), JSON (filenames, configs), STL/GLB (meshes) |
| **Parallelism** | Python multiprocessing (Pool, DDP), ProcessPoolExecutor |
| **GPU** | EGL offscreen rendering, CUDA inference, multi-GPU training |

---

## 10. Directory Structure Reference

```
ytp_oasis/
├── collecting-data/          # Stage 1: Raw data & rendering
│   ├── stlFiles/             #   Raw STL dental scans
│   ├── scanfiles/            #   iTero scan zip archives
│   ├── rendered_images/      #   Multi-view PNG renders per UUID
│   ├── glbFiles/             #   Compressed GLB for web viewer
│   ├── render_multiview_final.py
│   └── run_pool_render.py
│
├── normalization/            # Stage 1b: Point cloud datasets
│   ├── make_npy.py           #   Build SimCLR dataset
│   ├── make_mae_npy.py       #   Build MAE dataset
│   └── *.npy                 #   Output datasets
│
├── train/                    # Stage 2: Model training + embeddings
│   ├── train_oasis.py        #   SimCLR training script
│   ├── train_mae_ddp.py      #   Point-MAE DDP training
│   ├── simclr/               #   SimCLR checkpoint + embeddings
│   ├── mae/                  #   MAE checkpoint + embeddings
│   ├── dinov2/               #   DINOv2 embeddings
│   ├── dinov3/               #   DINOv3 model + embeddings
│   ├── dinov3_concat/        #   DINOv3 concat embeddings
│   └── dinov3_gallery/       #   DINOv3 gallery embeddings
│
├── embed/                    # Stage 3: Unified embedding module
│   ├── __init__.py           #   Public API: embed_stl()
│   ├── config.py             #   Paths & constants
│   ├── preprocessing.py      #   Per-model preprocessing
│   ├── models.py             #   Model loading & inference
│   └── batch_embed.py        #   CLI batch embedding tool
│
├── validation/               # Stage 4a: Validation pipeline
│   ├── pick_test_scans.py    #   60/581 test/base split
│   ├── validate_dinov2.py    #   Desktop 3D validation
│   ├── progress/             #   Per-model progress JSONs
│   └── reports/              #   Per-model report JSONs
│
├── website/                  # Stage 4b: Web application
│   ├── app.py                #   Flask factory
│   ├── config.py             #   App configuration
│   ├── extensions.py         #   SQLAlchemy + LoginManager
│   ├── models/               #   User model (SQLAlchemy)
│   ├── routes/               #   6 blueprints (auth, main, search, upload, collection, validation)
│   ├── templates/            #   Jinja2 HTML templates
│   ├── static/               #   CSS, JS, uploads
│   └── database/             #   SQLite DB + embedding cache
│
└── tools/                    # Utilities
    ├── convert_stl_to_glb.py #   Batch STL→GLB converter
    └── view_stl.py           #   STL previewer
```

---

*Report generated from full codebase analysis of the OASIS project.*
