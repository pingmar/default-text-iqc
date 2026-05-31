from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class FeatureRecord:
    feature_id: int
    layer: int
    # Feature direction vector, shape (d_model,). Stored as list for JSON portability.
    vector: list[float]
    top_k_examples: list[str]
    # (bin_edges [n_bins+1], counts [n_bins]) — each stored as list[float].
    activation_histogram: tuple[list[float], list[float]]
    semantic_label: str = ""
    decomposition: str = "pca"
    # 0 = monosemantic, 1 = maximally polysemantic. -1 = not yet computed.
    polysemanticity: float = -1.0


class FeatureDictionary:
    """
    Typed container for FeatureRecord instances.
    Key: (layer, feature_id). Supports lookup, iteration, and JSON I/O.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[int, int], FeatureRecord] = {}

    def add(self, record: FeatureRecord) -> None:
        self._records[(record.layer, record.feature_id)] = record

    def get(self, layer: int, feature_id: int) -> FeatureRecord:
        return self._records[(layer, feature_id)]

    def records_for_layer(self, layer: int) -> list[FeatureRecord]:
        return [r for (l, _), r in self._records.items() if l == layer]

    def all_records(self) -> list[FeatureRecord]:
        return list(self._records.values())

    @classmethod
    def build(
        cls,
        components: dict[int, np.ndarray],
        projections: dict[int, np.ndarray],
        texts: list[str],
        config,
    ) -> "FeatureDictionary":
        from ssm_extract_ufe_text.features import (
            compute_activation_histograms,
            get_top_k_examples,
        )

        fd = cls()
        for layer, comps in components.items():
            projs = projections[layer]
            histograms = compute_activation_histograms(projs, n_bins=config.n_hist_bins)
            top_k = get_top_k_examples(projs, texts, k=config.top_k)
            for feat_id in range(comps.shape[0]):
                edges, counts = histograms[feat_id]
                record = FeatureRecord(
                    feature_id=feat_id,
                    layer=layer,
                    vector=comps[feat_id].tolist(),
                    top_k_examples=top_k[feat_id],
                    activation_histogram=(edges.tolist(), counts.tolist()),
                    decomposition=config.decomposition,
                )
                fd.add(record)
        return fd

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        records_list = [asdict(r) for r in self._records.values()]
        with open(path, "w") as f:
            json.dump(records_list, f)

    @classmethod
    def load(cls, path: str) -> "FeatureDictionary":
        with open(path) as f:
            records_list = json.load(f)
        fd = cls()
        for d in records_list:
            d["activation_histogram"] = tuple(d["activation_histogram"])
            # Backward-compatible: records written before polysemanticity was added.
            d.setdefault("polysemanticity", -1.0)
            fd.add(FeatureRecord(**d))
        return fd

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records.values())
