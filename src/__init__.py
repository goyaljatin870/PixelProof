"""
PixelProof - Multimodal Deepfake & Regional Misinformation Detection Engine
"""

import sys
from pathlib import Path

# Ensure package directory is in sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

__version__ = "2.0.0"
