from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, **modules: Any) -> None:
    """Save a dict of name -> nn.Module state_dicts plus optional metadata.

    Anything that's not an nn.Module is stored as-is (e.g., int step counter).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for name, obj in modules.items():
        if hasattr(obj, "state_dict"):
            payload[name] = obj.state_dict()
        else:
            payload[name] = obj
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
