# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.fused_moe.expert_storage import (
    SafetensorsExpertStore,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)


def _checkpoint(root: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    prefix = "model.layers.3.mlp.experts.7"
    tensors = {
        f"{prefix}.gate_proj.weight": torch.arange(12).reshape(3, 4).float(),
        f"{prefix}.up_proj.weight": torch.arange(12, 24).reshape(3, 4).float(),
        f"{prefix}.down_proj.weight": torch.arange(12).reshape(4, 3).float(),
        f"{prefix}.gate_proj.weight_scale_inv": torch.arange(3).reshape(1, 3).float(),
        f"{prefix}.up_proj.weight_scale_inv": torch.arange(3, 6).reshape(1, 3).float(),
        f"{prefix}.down_proj.weight_scale_inv": torch.arange(4).reshape(2, 2).float(),
        "model.layers.3.input_layernorm.weight": torch.ones(4),
    }
    shard = root / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    index = {
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {name: shard.name for name in tensors},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return shard, tensors


def test_store_loads_canonical_fused_expert(tmp_path: Path) -> None:
    _, tensors = _checkpoint(tmp_path)
    store = SafetensorsExpertStore(str(tmp_path))
    w13 = torch.empty(6, 4)
    w2 = torch.empty(4, 3)
    w13_scale = torch.empty(2, 3)
    w2_scale = torch.empty(2, 2)

    byte_count = store.load_expert(
        layer_name="model.layers.3.mlp.experts",
        expert_id=7,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    prefix = "model.layers.3.mlp.experts.7"
    torch.testing.assert_close(w13[:3], tensors[f"{prefix}.gate_proj.weight"])
    torch.testing.assert_close(w13[3:], tensors[f"{prefix}.up_proj.weight"])
    torch.testing.assert_close(w2, tensors[f"{prefix}.down_proj.weight"])
    torch.testing.assert_close(
        w13_scale[:1], tensors[f"{prefix}.gate_proj.weight_scale_inv"]
    )
    torch.testing.assert_close(
        w13_scale[1:], tensors[f"{prefix}.up_proj.weight_scale_inv"]
    )
    torch.testing.assert_close(
        w2_scale, tensors[f"{prefix}.down_proj.weight_scale_inv"]
    )
    assert byte_count == sum(
        tensor.nbytes for name, tensor in tensors.items() if ".experts." in name
    )


def test_iterator_leaves_routed_experts_on_storage(tmp_path: Path) -> None:
    shard, _ = _checkpoint(tmp_path)
    loaded = dict(
        safetensors_weights_iterator([str(shard)], False, skip_routed_experts=True)
    )
    assert set(loaded) == {"model.layers.3.input_layernorm.weight"}
