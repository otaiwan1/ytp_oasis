"""
embed/convert.py — STL to GLB conversion utility.
"""

import gzip
import trimesh
from pathlib import Path


def stl_to_glb_gzipped(stl_path, glb_path):
    """Convert an STL file to a gzip-compressed GLB file.

    Args:
        stl_path: Path to input .stl file.
        glb_path: Path to output .glb file (will be gzip-compressed).
    """
    stl_path = Path(stl_path)
    glb_path = Path(glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(stl_path), force='mesh')
    glb_data = mesh.export(file_type='glb')

    with open(str(glb_path), 'wb') as f:
        f.write(gzip.compress(glb_data))
