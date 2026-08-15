"""
util.py — safe IO + crash-proofing helpers used across the system.
Never raises: corrupt files are healed (dropped), writes are atomic.
"""
from __future__ import annotations
import os, json


def safe_read_json(path: str, default=None):
    """Read JSON; on any error, remove the corrupt file and return default."""
    try:
        if not path or not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        return default


def safe_write_json(path: str, data) -> bool:
    """Atomic write (tmp + rename). Never raises."""
    try:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False
