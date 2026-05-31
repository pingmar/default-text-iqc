"""JSON-serializable feature dictionary records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


@dataclass
class FeatureRecord:
    feature_id: int
    layer: int | str
    vector: list[float]
    top_k_image_indices: list[int] = field(default_factory=list)
    activation_histogram: tuple[list[float], list[float]] = field(
        default_factory=lambda: ([], [])
    )
    spatial_activation_maps: list[list[list[float]]] = field(default_factory=list)
    visualization_path: str = ""
    decomposition: str = "pca"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureRecord:
        defaults = {
            "top_k_image_indices": [],
            "activation_histogram": ([], []),
            "spatial_activation_maps": [],
            "visualization_path": "",
            "decomposition": "pca",
        }
        merged = {**defaults, **data}
        hist = merged["activation_histogram"]
        if isinstance(hist, list):
            hist = tuple(hist)
        merged["activation_histogram"] = hist
        return cls(**merged)


@dataclass
class FeatureDictionary:
    records: list[FeatureRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDictionary:
        return cls(
            records=[FeatureRecord.from_dict(item) for item in data.get("records", [])],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureDictionary:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())
