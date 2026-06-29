"""Pytest configuration: make the project ``code`` directory importable."""

import os
import sys

# Insert the code/ directory (parent of this tests/ dir) at the front of the
# path so bare imports like ``from config import get_config`` resolve.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
