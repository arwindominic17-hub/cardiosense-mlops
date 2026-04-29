"""
conftest.py
===========
Pytest configuration — adds src/ and the project root to sys.path
so that both pipeline modules and the api package are importable.
"""

import sys
import os

# Project root (one level up from tests/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")

for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
