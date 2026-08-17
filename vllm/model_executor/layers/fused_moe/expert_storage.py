# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-core safetensors backing for independently stored MoE experts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


class SafetensorsExpertStore:
    """Read one canonical gate/up/down expert directly from a checkpoint.

    The store intentionally preserves the checkpoint as the storage format.
    No second repacked copy is required: the safetensors index identifies the
    shard for each tensor and ``safe_open`` mmaps only the requested entries.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise ValueError(
                "moe_expert_storage_path must contain "
                f"model.safetensors.index.json: {self.root}"
            )
        with index_path.open(encoding="utf-8") as src:
            index = json.load(src)
        try:
            self.weight_map: dict[str, str] = index["weight_map"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid safetensors index: {index_path}") from exc

    @staticmethod
    def _source_names(layer_name: str, expert_id: int) -> dict[str, str]:
        prefix = f"{layer_name}.{expert_id}"
        return {
            "gate": f"{prefix}.gate_proj.weight",
            "up": f"{prefix}.up_proj.weight",
            "down": f"{prefix}.down_proj.weight",
            "gate_scale": f"{prefix}.gate_proj.weight_scale_inv",
            "up_scale": f"{prefix}.up_proj.weight_scale_inv",
            "down_scale": f"{prefix}.down_proj.weight_scale_inv",
        }

    def load_expert(
        self,
        *,
        layer_name: str,
        expert_id: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
    ) -> int:
        """Load one expert into pinned canonical vLLM slot tensors."""
        names = self._source_names(layer_name, expert_id)
        required = ["gate", "up", "down"]
        if w13_scale is not None or w2_scale is not None:
            if w13_scale is None or w2_scale is None:
                raise ValueError("Both w13_scale and w2_scale must be provided")
            required.extend(("gate_scale", "up_scale", "down_scale"))

        missing = [names[key] for key in required if names[key] not in self.weight_map]
        if missing:
            raise KeyError(
                "Checkpoint does not contain independently indexed expert tensors: "
                + ", ".join(missing)
            )

        tensors: dict[str, torch.Tensor] = {}
        by_shard: dict[str, list[str]] = {}
        for key in required:
            by_shard.setdefault(self.weight_map[names[key]], []).append(key)
        for shard, keys in by_shard.items():
            with safe_open(self.root / shard, framework="pt", device="cpu") as src:
                for key in keys:
                    tensors[key] = src.get_tensor(names[key])

        intermediate = w13.shape[0] // 2
        if tensors["gate"].shape != w13[:intermediate].shape:
            raise ValueError(
                f"gate expert shape {tuple(tensors['gate'].shape)} does not match "
                f"cache slot {tuple(w13[:intermediate].shape)}"
            )
        if tensors["up"].shape != w13[intermediate:].shape:
            raise ValueError(
                f"up expert shape {tuple(tensors['up'].shape)} does not match "
                f"cache slot {tuple(w13[intermediate:].shape)}"
            )
        if tensors["down"].shape != w2.shape:
            raise ValueError(
                f"down expert shape {tuple(tensors['down'].shape)} does not match "
                f"cache slot {tuple(w2.shape)}"
            )

        w13[:intermediate].copy_(tensors["gate"])
        w13[intermediate:].copy_(tensors["up"])
        w2.copy_(tensors["down"])
        byte_count = sum(tensors[key].nbytes for key in ("gate", "up", "down"))

        if w13_scale is not None:
            assert w2_scale is not None
            scale_intermediate = w13_scale.shape[0] // 2
            w13_scale[:scale_intermediate].copy_(tensors["gate_scale"])
            w13_scale[scale_intermediate:].copy_(tensors["up_scale"])
            w2_scale.copy_(tensors["down_scale"])
            byte_count += sum(
                tensors[key].nbytes for key in ("gate_scale", "up_scale", "down_scale")
            )
        return byte_count


_STORES: dict[str, SafetensorsExpertStore] = {}


def get_safetensors_expert_store(path: str) -> SafetensorsExpertStore:
    key = str(Path(path).expanduser().resolve())
    store = _STORES.get(key)
    if store is None:
        store = SafetensorsExpertStore(key)
        _STORES[key] = store
    return store
