"""
Pytest configuration: adds KGeoMIP root to sys.path so that `from src.*` imports work
when running pytest from anywhere in the repository.
"""
import sys
from pathlib import Path

# KGeoMIP/ is the parent of tests/
GEOMIP_ROOT = Path(__file__).resolve().parents[1]
if str(GEOMIP_ROOT) not in sys.path:
    sys.path.insert(0, str(GEOMIP_ROOT))
