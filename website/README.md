# OASIS Website

## Orthodontic AI Similarity Intelligent System — Web Interface

### Setup

```bash
cd website
pip install -r requirements.txt
python app.py
```

The server starts at **http://localhost:5000**.

### Project Structure

```
website/
├── app.py                  # Flask application factory + entry point
├── config.py               # Configuration (paths, model params)
├── requirements.txt        # Python dependencies
├── database/               # SQLite DB + embedding caches (auto-created)
├── models/
│   └── user.py             # User model (auth)
├── routes/
│   ├── auth.py             # Login / Register / Logout
│   ├── main.py             # Home + About pages
│   ├── search.py           # Similarity search (model inference)
│   ├── upload.py           # Upload new patient scans
│   └── collection.py       # Browse patient scan collection
├── templates/
│   ├── base.html           # Base layout (navbar, footer, theme)
│   ├── auth/               # Login & Register pages
│   ├── main/               # Home & About pages
│   ├── search/             # Search page
│   ├── upload/             # Upload page
│   └── collection/         # Browse & Patient detail pages
└── static/
    └── uploads/            # Temporary upload storage (auto-created)
```

### Features

1. **Login / Register** — User account system with secure password hashing
2. **Similarity Search** — Upload STL, find top-5 similar scans using SimCLR model
3. **Patient Collection** — Browse all patients, view multi-view rendered images
4. **Upload** — Add new patient scans with drag-and-drop
5. **About** — Project overview and technical details

### Notes

- The model file is loaded from `train/best_dental_simclr_multi_gpu.pth`
- STL scan data should be placed in `collecting-data/stlFiles/`
- Files follow the naming convention: `patientUID_serialNumber.stl`
- Rendered images are served from `collecting-data/rendered_images/`
- Embeddings are cached in `database/` for fast subsequent searches
