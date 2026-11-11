"""Compatibility shim. Canonical package is `aircraft_agent` (under src/).

Old imports (`agent_mvp.tools`, ...) and entry points
(`uvicorn agent_mvp.app:app`, `python -m agent_mvp.cli`) keep working.
New code should import from `aircraft_agent.*`.
"""
import sys as _sys
from pathlib import Path as _Path

_src = _Path(__file__).resolve().parent.parent / "src"
if _src.is_dir() and str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

__version__ = "0.1.0"
