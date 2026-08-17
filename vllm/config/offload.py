# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch"]
MoECacheSplit = Literal["token", "expert"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    moe_expert_cache_size: int = Field(default=0, ge=0)
    """Number of MoE expert weight rows to keep in a GPU cache buffer."""

    moe_expert_cache_split: MoECacheSplit = "token"
    """How a forward needing more experts than the cache holds is broken up.
    - "token": split the batch by rows. Output matches the uncached path
      exactly, but costs one kernel launch per chunk, approaching one per token
      as the cache approaches top_k.
    - "expert": split the expert set and sum the parts. Costs
      ceil(experts/cache size) launches whatever the batch size, and fetches
      each expert at most once per forward, but each part is rounded to the
      model dtype before being summed, so results differ from the uncached
      path at rounding level."""

    moe_expert_prefetch_trace: str | None = None
    """JSONL route trace used as an oracle prediction schedule."""

    moe_expert_trace_output: str | None = None
    """Write observed per-forward expert sets to this JSONL path."""

    moe_expert_prefetch_lookahead: int = Field(default=0, ge=0)
    """Maximum number of future MoE calls considered by the allocator."""

    moe_expert_prefetch_budget_mb: float = Field(default=0, ge=0)
    """Maximum MiB of predicted expert copies enqueued at each MoE call."""

    moe_expert_prefetch_max_inflight_mb: float = Field(default=0, ge=0)
    """Maximum MiB of predicted expert copies outstanding on H2D streams.
    Zero leaves the per-call budget as the only transfer bound."""

    moe_expert_storage_path: str | None = None
    """Local safetensors checkpoint used as an out-of-core expert store.
    Routed expert tensors are skipped during ordinary model loading and read
    into a bounded pinned-RAM cache on demand or by the predictor."""

    moe_expert_host_cache_size: int = Field(default=0, ge=0)
    """Number of pinned-RAM expert slots per MoE layer when storage backing is
    enabled. This is independent of ``moe_expert_cache_size`` in HBM."""

    moe_expert_storage_prefetch_budget_mb: float = Field(default=0, ge=0)
    """Maximum MiB of NVMe-to-pinned-RAM prediction work admitted per MoE
    call."""

    moe_expert_storage_max_inflight_mb: float = Field(default=0, ge=0)
    """Maximum MiB of NVMe reads outstanding across worker threads. Zero
    leaves the per-call storage budget as the only bound."""

    moe_expert_storage_workers: int = Field(default=4, ge=1)
    """Number of background workers used for predicted storage reads."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        predictive = self.moe_expert_prefetch_trace is not None
        storage = self.moe_expert_storage_path is not None
        if predictive and self.moe_expert_cache_size == 0:
            raise ValueError(
                "moe_expert_prefetch_trace requires moe_expert_cache_size > 0"
            )
        if predictive and self.moe_expert_prefetch_lookahead == 0:
            raise ValueError(
                "moe_expert_prefetch_trace requires moe_expert_prefetch_lookahead > 0"
            )
        if predictive and self.moe_expert_prefetch_budget_mb == 0:
            raise ValueError(
                "moe_expert_prefetch_trace requires moe_expert_prefetch_budget_mb > 0"
            )
        if self.moe_expert_trace_output and self.moe_expert_cache_size == 0:
            raise ValueError(
                "moe_expert_trace_output requires moe_expert_cache_size > 0"
            )
        if storage and self.moe_expert_cache_size == 0:
            raise ValueError(
                "moe_expert_storage_path requires moe_expert_cache_size > 0"
            )
        if storage and self.moe_expert_host_cache_size == 0:
            raise ValueError(
                "moe_expert_storage_path requires moe_expert_host_cache_size > 0"
            )
        if not storage and self.moe_expert_host_cache_size > 0:
            raise ValueError(
                "moe_expert_host_cache_size requires moe_expert_storage_path"
            )
        if predictive and storage and self.moe_expert_storage_prefetch_budget_mb == 0:
            raise ValueError(
                "Predictive storage backing requires "
                "moe_expert_storage_prefetch_budget_mb > 0"
            )
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
