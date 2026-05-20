from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Append-only JSON-lines logger. One row per evaluation tick."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(step)}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()
            row[k] = v
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass
