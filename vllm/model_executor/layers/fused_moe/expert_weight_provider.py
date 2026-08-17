# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_prefetch import (
    ExpertPrefetchCoordinator,
    PrefetchOutcome,
)

logger = init_logger(__name__)

_STATS_LOG_INTERVAL = 1000


def _pinned_cpu_copy(src: torch.Tensor) -> torch.Tensor:
    """Pinned CPU copy of *src* with exactly one cross-device transfer.

    ``src.cpu().pin_memory()`` stages GPU tensors through pageable memory,
    copying twice; copying straight into a pinned allocation halves init
    time and transient host RAM for multi-GB expert tensors.
    """
    if src.device.type == "cpu":
        return src if src.is_pinned() else src.pin_memory()
    dst = torch.empty_like(src, device="cpu", pin_memory=True)
    dst.copy_(src)
    return dst


# How a forward too wide for the cache is broken up.
#   "token"  -- split the rows; every token's full sum still happens inside one
#               kernel call, so output matches the uncached path exactly. Costs
#               one launch per chunk, which approaches one per token as
#               capacity approaches top_k.
#   "expert" -- split the experts; one launch per ceil(experts/capacity)
#               regardless of batch size, and each expert is fetched at most
#               once per forward. Each group's partial sum is rounded to the
#               model dtype before being accumulated, so results differ from
#               the uncached path at rounding level.
MoECacheSplit = Literal["token", "expert"]

# Every row: what the expert split passes, since it never cuts the batch.
_ALL_ROWS = slice(None)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption.

    ``expert_map`` follows the expert-parallel convention the kernels already
    understand: global expert id to buffer slot, or -1 for experts that are not
    resident. Pairs routed to -1 are dropped during alignment, which is what
    lets one forward be evaluated a group of experts at a time.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    expert_map: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass
class StoragePrefetchOutcome:
    """Result of admitting predicted NVMe-to-pinned-RAM transfers."""

    requested: int
    resident: int
    loaded: int
    bytes_enqueued: int
    futures: list[Future[int]]


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    prepare() copies any missing experts from CPU to GPU, evicting the
    lowest-scored resident entry when the buffer is full, and returns an
    ExpertWeightResult whose expert_map selects them. Forwards needing more
    experts than fit are handled by run_with_expert_cache(), which splits
    them according to ``split``.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        split: MoECacheSplit = "token",
        layer_name: str = "",
        prefetch_coordinator: ExpertPrefetchCoordinator | None = None,
        num_experts: int | None = None,
        storage_path: str | None = None,
        host_capacity: int = 0,
    ) -> None:
        source_experts = w13_weight.size(0)
        num_experts = num_experts or source_experts
        if storage_path is not None and source_experts != capacity:
            raise ValueError(
                "Storage-backed expert weights must be allocated as physical "
                f"HBM slots ({source_experts} rows != capacity {capacity})"
            )

        self.capacity = capacity
        self.split: MoECacheSplit = split
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self.prefetch_useful = 0
        self.prefetch_wasted = 0
        self.prefetch_waits = 0
        self.prefetch_loaded = 0
        self.prefetch_bytes = 0
        self.storage_hits = 0
        self.storage_misses = 0
        self.storage_prefetch_loaded = 0
        self.storage_prefetch_bytes = 0
        self.storage_prefetch_waits = 0
        self._prepare_calls = 0
        self.layer_name = layer_name
        self._prefetch_coordinator = prefetch_coordinator

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
        else:
            cuda_device = w13_weight.device
        self._storage = None
        self._host_capacity = 0
        if storage_path is None:
            self._cpu_w13 = _pinned_cpu_copy(w13_weight)
            self._cpu_w2 = _pinned_cpu_copy(w2_weight)
            self._buf_w13 = torch.empty(
                capacity,
                *w13_weight.shape[1:],
                dtype=w13_weight.dtype,
                device=cuda_device,
            )
            self._buf_w2 = torch.empty(
                capacity,
                *w2_weight.shape[1:],
                dtype=w2_weight.dtype,
                device=cuda_device,
            )
        else:
            from vllm.model_executor.layers.fused_moe.expert_storage import (
                get_safetensors_expert_store,
            )

            self._storage = get_safetensors_expert_store(storage_path)
            self._host_capacity = min(host_capacity, num_experts)
            self._cpu_w13 = torch.empty(
                self._host_capacity,
                *w13_weight.shape[1:],
                dtype=w13_weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._cpu_w2 = torch.empty(
                self._host_capacity,
                *w2_weight.shape[1:],
                dtype=w2_weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._buf_w13 = w13_weight
            self._buf_w2 = w2_weight
        self.device = self._buf_w13.device

        if w13_scale is not None and w2_scale is not None:
            # Pinned for the same reason the weights are: these are copied on
            # every miss, and pageable source memory forces a staging copy.
            if storage_path is None:
                self._cpu_w13_scale = _pinned_cpu_copy(w13_scale)
                self._cpu_w2_scale = _pinned_cpu_copy(w2_scale)
                self._buf_w13_scale = torch.empty(
                    capacity,
                    *w13_scale.shape[1:],
                    dtype=w13_scale.dtype,
                    device=cuda_device,
                )
                self._buf_w2_scale = torch.empty(
                    capacity,
                    *w2_scale.shape[1:],
                    dtype=w2_scale.dtype,
                    device=cuda_device,
                )
            else:
                self._cpu_w13_scale = torch.empty(
                    self._host_capacity,
                    *w13_scale.shape[1:],
                    dtype=w13_scale.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self._cpu_w2_scale = torch.empty(
                    self._host_capacity,
                    *w2_scale.shape[1:],
                    dtype=w2_scale.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self._buf_w13_scale = w13_scale
                self._buf_w2_scale = w2_scale
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))
        self._pending: dict[int, torch.cuda.Event] = {}
        # Last compute-stream use of each physical slot. A speculative copy
        # runs on a separate stream, so slot reuse must depend on this event or
        # it could overwrite weights while the previous kernel is reading them.
        self._slot_last_use: dict[int, torch.cuda.Event] = {}
        self._reserved_deadlines: dict[int, list[int]] = {}
        self._current_ordinal: int | None = None

        # Bounded storage -> pinned-RAM cache. Entries are reserved before a
        # worker starts and published by completion of the corresponding
        # future. CUDA events protect pinned slots until H2D DMA is finished.
        self._host_lru: dict[int, list] = {}
        self._host_clock = 0
        self._host_free_slots = list(range(self._host_capacity))
        self._host_pending: dict[int, Future[int]] = {}
        self._host_slot_last_use: dict[int, torch.cuda.Event] = {}
        self._host_lock = threading.Lock()

        # Expert map handed to the kernel: expert id to slot for the group
        # being evaluated, -1 for everything else. Rebuilt each prepare() --
        # it must expose exactly the requested group, not whatever else
        # happens to still be resident, or experts already summed in an
        # earlier group would be counted twice. Residency itself lives in
        # _lru; this is only the view the kernel gets. Built in a pinned host
        # mirror and uploaded once, rather than a transfer per entry.
        self._mapping: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=cuda_device
        )
        self._mapping_host: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32
        ).pin_memory()

        if self._prefetch_coordinator is not None:
            self._prefetch_coordinator.register(layer_name, self)

    @property
    def bytes_per_expert(self) -> int:
        tensors = [self._cpu_w13, self._cpu_w2]
        if self._cpu_w13_scale is not None:
            assert self._cpu_w2_scale is not None
            tensors.extend((self._cpu_w13_scale, self._cpu_w2_scale))
        return sum(t[0].numel() * t.element_size() for t in tensors)

    @property
    def has_storage_backing(self) -> bool:
        return self._storage is not None

    def contains_all(self, expert_ids: tuple[int, ...]) -> bool:
        return all(expert_id in self._lru for expert_id in expert_ids)

    def host_contains_all(self, expert_ids: tuple[int, ...]) -> bool:
        if self._storage is None:
            return True
        with self._host_lock:
            return all(expert_id in self._host_lru for expert_id in expert_ids)

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            event = self._pending.pop(expert_id, None)
            if event is not None:
                event.synchronize()
            entry = self._lru.pop(expert_id)
            self.prefetch_wasted += len(self._reserved_deadlines.pop(expert_id, []))
            self._free_slots.append(entry[0])

    def expire_predictions(
        self, current_ordinal: int, actual_experts: set[int]
    ) -> None:
        """Release predictions whose scheduled use has arrived or passed."""
        self._current_ordinal = current_ordinal
        for expert_id, deadlines in list(self._reserved_deadlines.items()):
            if expert_id in actual_experts:
                continue
            future = [d for d in deadlines if d > current_ordinal]
            self.prefetch_wasted += len(deadlines) - len(future)
            if future:
                self._reserved_deadlines[expert_id] = future
            else:
                del self._reserved_deadlines[expert_id]

    def _reserve(self, expert_id: int, deadline: int) -> None:
        deadlines = self._reserved_deadlines.setdefault(expert_id, [])
        if deadline not in deadlines:
            deadlines.append(deadline)
            deadlines.sort()

    def _eviction_key(self, expert_id: int) -> tuple[float, float, float]:
        _, freq, last = self._lru[expert_id]
        age = self._clock - last + 1
        score = freq / age
        deadlines = self._reserved_deadlines.get(expert_id)
        if not deadlines:
            return (0.0, 0.0, score)
        return (1.0, -float(deadlines[0]), score)

    def _reap_pending(self) -> None:
        for expert_id, event in list(self._pending.items()):
            if event.query():
                del self._pending[expert_id]

    def _take_slot(
        self,
        needed: set[int],
        *,
        speculative_deadline: int | None,
    ) -> int | None:
        if self._free_slots:
            return self._free_slots.pop()

        candidates = [expert_id for expert_id in self._lru if expert_id not in needed]
        if speculative_deadline is not None:
            candidates = [
                expert_id
                for expert_id in candidates
                if expert_id not in self._pending
                and (
                    expert_id not in self._reserved_deadlines
                    or self._reserved_deadlines[expert_id][0] > speculative_deadline
                )
            ]
        else:
            ready = [e for e in candidates if e not in self._pending]
            if ready:
                candidates = ready
        if not candidates:
            return None

        victim = min(candidates, key=self._eviction_key)
        event = self._pending.pop(victim, None)
        if event is not None:
            event.synchronize()
        self.prefetch_wasted += len(self._reserved_deadlines.pop(victim, []))
        return self._lru.pop(victim)[0]

    def _host_eviction_key(self, expert_id: int) -> tuple[float, float, float]:
        _, freq, last = self._host_lru[expert_id]
        score = freq / (self._host_clock - last + 1)
        deadlines = self._reserved_deadlines.get(expert_id)
        if not deadlines:
            return (0.0, 0.0, score)
        return (1.0, -float(deadlines[0]), score)

    def _take_host_slot_locked(
        self, needed: set[int], speculative_deadline: int | None
    ) -> int | None:
        if self._host_free_slots:
            return self._host_free_slots.pop()
        candidates = [
            expert_id
            for expert_id in self._host_lru
            if expert_id not in needed and expert_id not in self._host_pending
        ]
        if speculative_deadline is not None:
            candidates = [
                expert_id
                for expert_id in candidates
                if expert_id not in self._reserved_deadlines
                or self._reserved_deadlines[expert_id][0] > speculative_deadline
            ]
        if not candidates:
            return None
        victim = min(candidates, key=self._host_eviction_key)
        return self._host_lru.pop(victim)[0]

    def _load_host_slot(self, expert_id: int, slot: int) -> int:
        assert self._storage is not None
        with self._host_lock:
            last_use = self._host_slot_last_use.pop(slot, None)
        if last_use is not None:
            last_use.synchronize()
        return self._storage.load_expert(
            layer_name=self.layer_name,
            expert_id=expert_id,
            w13=self._cpu_w13[slot],
            w2=self._cpu_w2[slot],
            w13_scale=(
                self._cpu_w13_scale[slot] if self._cpu_w13_scale is not None else None
            ),
            w2_scale=(
                self._cpu_w2_scale[slot] if self._cpu_w2_scale is not None else None
            ),
        )

    def _reap_host_pending(self) -> None:
        if self._storage is None:
            return
        completed: list[tuple[int, Future[int]]] = []
        with self._host_lock:
            for expert_id, future in list(self._host_pending.items()):
                if future.done():
                    completed.append((expert_id, future))
                    del self._host_pending[expert_id]
        for expert_id, future in completed:
            try:
                future.result()
            except Exception:
                with self._host_lock:
                    entry = self._host_lru.pop(expert_id, None)
                    if entry is not None:
                        self._host_free_slots.append(entry[0])
                raise

    def _host_is_ready(self, expert_id: int) -> bool:
        if self._storage is None:
            return True
        with self._host_lock:
            future = self._host_pending.get(expert_id)
            return expert_id in self._host_lru and (
                future is None or (future.done() and future.exception() is None)
            )

    def _ensure_host(self, expert_id: int, needed: set[int]) -> int:
        if self._storage is None:
            return expert_id
        self._reap_host_pending()
        future: Future[int] | None
        with self._host_lock:
            entry = self._host_lru.get(expert_id)
            future = self._host_pending.get(expert_id)
        if entry is not None:
            if future is not None:
                self.storage_prefetch_waits += 1
                try:
                    future.result()
                except Exception:
                    with self._host_lock:
                        self._host_pending.pop(expert_id, None)
                        failed = self._host_lru.pop(expert_id, None)
                        if failed is not None:
                            self._host_free_slots.append(failed[0])
                    raise
                with self._host_lock:
                    self._host_pending.pop(expert_id, None)
            with self._host_lock:
                self._host_clock += 1
                entry = self._host_lru[expert_id]
                entry[1] += 1
                entry[2] = self._host_clock
                slot = entry[0]
            self.storage_hits += 1
            return slot

        with self._host_lock:
            slot = self._take_host_slot_locked(needed, speculative_deadline=None)
            if slot is None:
                raise RuntimeError(
                    "Pinned expert cache has no evictable slot; increase "
                    "--moe-expert-host-cache-size"
                )
            self._host_clock += 1
            self._host_lru[expert_id] = [slot, 1, self._host_clock]
        try:
            self._load_host_slot(expert_id, slot)
        except Exception:
            with self._host_lock:
                self._host_lru.pop(expert_id, None)
                self._host_free_slots.append(slot)
            raise
        self.storage_misses += 1
        return slot

    @torch.compiler.disable
    def prefetch_host(
        self,
        unique_ids: list[int],
        *,
        deadline: int,
        max_bytes: int,
        executor: ThreadPoolExecutor,
    ) -> StoragePrefetchOutcome:
        """Admit predicted storage reads without blocking the model thread."""
        if self._storage is None:
            return StoragePrefetchOutcome(len(unique_ids), len(unique_ids), 0, 0, [])
        self._reap_host_pending()
        unique_ids = list(dict.fromkeys(unique_ids))
        requested = len(unique_ids)
        resident = 0
        loaded = 0
        futures: list[Future[int]] = []
        remaining = max_bytes
        needed = set(unique_ids)
        for expert_id in unique_ids:
            if not 0 <= expert_id < self._num_experts:
                raise ValueError(
                    f"Predicted expert {expert_id} outside [0, {self._num_experts})"
                )
            with self._host_lock:
                present = expert_id in self._host_lru
            if present:
                self._reserve(expert_id, deadline)
                resident += 1
                continue
            if self.bytes_per_expert > remaining:
                break
            with self._host_lock:
                slot = self._take_host_slot_locked(
                    needed, speculative_deadline=deadline
                )
                if slot is None:
                    break
                self._host_clock += 1
                self._host_lru[expert_id] = [slot, 0, self._host_clock]
                future = executor.submit(self._load_host_slot, expert_id, slot)
                self._host_pending[expert_id] = future
            self._reserve(expert_id, deadline)
            futures.append(future)
            resident += 1
            loaded += 1
            remaining -= self.bytes_per_expert

        byte_count = loaded * self.bytes_per_expert
        self.storage_prefetch_loaded += loaded
        self.storage_prefetch_bytes += byte_count
        return StoragePrefetchOutcome(
            requested=requested,
            resident=resident,
            loaded=loaded,
            bytes_enqueued=byte_count,
            futures=futures,
        )

    def _copy_expert(
        self, expert_id: int, slot: int, host_needed: set[int] | None = None
    ) -> None:
        source_slot = self._ensure_host(expert_id, host_needed or {expert_id})
        last_use = self._slot_last_use.pop(slot, None)
        if last_use is not None:
            torch.cuda.current_stream(self.device).wait_event(last_use)
        self._buf_w13[slot].copy_(self._cpu_w13[source_slot], non_blocking=True)
        self._buf_w2[slot].copy_(self._cpu_w2[source_slot], non_blocking=True)
        if self._buf_w13_scale is not None:
            assert self._cpu_w13_scale is not None
            assert self._cpu_w2_scale is not None
            assert self._buf_w2_scale is not None
            self._buf_w13_scale[slot].copy_(
                self._cpu_w13_scale[source_slot], non_blocking=True
            )
            self._buf_w2_scale[slot].copy_(
                self._cpu_w2_scale[source_slot], non_blocking=True
            )
        if self._storage is not None:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            with self._host_lock:
                self._host_slot_last_use[source_slot] = event

    def mark_used(self, expert_ids: list[int]) -> None:
        """Publish when the compute stream has finished reading cache slots."""
        if (
            self._prefetch_coordinator is None
            or self._prefetch_coordinator.schedule is None
        ):
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        for expert_id in expert_ids:
            entry = self._lru.get(expert_id)
            if entry is not None:
                self._slot_last_use[entry[0]] = event

    @torch.compiler.disable
    def prefetch(
        self,
        unique_ids: list[int],
        *,
        deadline: int,
        max_bytes: int,
        stream: torch.cuda.Stream,
    ) -> PrefetchOutcome:
        """Admit predicted experts and enqueue their copies on *stream*."""
        self._reap_pending()
        self._reap_host_pending()
        unique_ids = list(dict.fromkeys(unique_ids))
        requested = len(unique_ids)
        resident = 0
        missing: list[int] = []
        for expert_id in unique_ids:
            if not 0 <= expert_id < self._num_experts:
                raise ValueError(
                    f"Predicted expert {expert_id} outside [0, {self._num_experts})"
                )
            if expert_id in self._lru:
                self._reserve(expert_id, deadline)
                resident += 1
            else:
                # Storage reads run on CPU workers. Never turn an HBM
                # prediction into a synchronous NVMe wait: a later allocator
                # tick will promote it after the pinned tier is ready.
                if self._host_is_ready(expert_id):
                    missing.append(expert_id)

        remaining = max_bytes
        loaded: list[int] = []
        needed = set(unique_ids)
        with torch.cuda.stream(stream):
            for expert_id in missing:
                if self.bytes_per_expert > remaining:
                    break
                slot = self._take_slot(needed, speculative_deadline=deadline)
                if slot is None:
                    break
                self._copy_expert(expert_id, slot, needed)
                self._clock += 1
                self._lru[expert_id] = [slot, 0, self._clock]
                self._reserve(expert_id, deadline)
                loaded.append(expert_id)
                resident += 1
                remaining -= self.bytes_per_expert

            event: torch.cuda.Event | None = None
            if loaded:
                event = torch.cuda.Event()
                event.record(stream)
                for expert_id in loaded:
                    self._pending[expert_id] = event

        byte_count = len(loaded) * self.bytes_per_expert
        self.prefetch_loaded += len(loaded)
        self.prefetch_bytes += byte_count
        return PrefetchOutcome(
            requested=requested,
            resident=resident,
            loaded=len(loaded),
            bytes_enqueued=byte_count,
            event=event,
        )

    def begin_forward(self, actual_experts: list[int]) -> None:
        if self._prefetch_coordinator is not None:
            self._prefetch_coordinator.begin_forward(self.layer_name, actual_experts)

    @torch.compiler.disable
    def plan_chunks(self, topk_ids: torch.Tensor) -> list[tuple[slice, list[int]]]:
        """Row ranges whose combined unique expert count fits the cache.

        Routing is per token, so evaluating a subset of rows and concatenating
        the results is equivalent to evaluating the whole batch -- and every
        token's sum still happens in a single kernel call, which is why this
        split reproduces the uncached output exactly.

        Returns a single full-width slice when the batch already fits, which is
        the common case, so callers pay nothing extra for it. Each slice comes
        with the expert ids it needs, computed from host data this method
        already has, so ``prepare()`` need not synchronize again per chunk.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ``(row slice, unique expert ids)`` pairs covering ``topk_ids``.

        Raises:
            RuntimeError: if one token alone routes to more experts than the
                cache can hold, which no amount of splitting can fix.
        """
        num_rows = topk_ids.size(0)
        if num_rows == 0:
            return [(slice(0, 0), [])]

        # The common case only needs the distinct ids, which the device can
        # reduce far more cheaply than transferring every row.
        unique = topk_ids.unique()
        if unique.numel() <= self.capacity:
            return [(slice(0, num_rows), unique.tolist())]

        rows = topk_ids.tolist()
        chunks: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        for i, row in enumerate(rows):
            row_ids = set(row)
            if len(seen | row_ids) > self.capacity:
                if i == start:
                    raise RuntimeError(
                        f"CachedWeightProvider: one token routes to "
                        f"{len(row_ids)} experts but "
                        f"--moe-expert-cache-size={self.capacity}. "
                        f"Set --moe-expert-cache-size >= {len(row_ids)}."
                    )
                chunks.append((slice(start, i), sorted(seen)))
                start = i
                seen = row_ids
            else:
                seen |= row_ids
        chunks.append((slice(start, num_rows), sorted(seen)))
        return chunks

    @torch.compiler.disable
    def plan_expert_groups(self, topk_ids: torch.Tensor) -> list[list[int]]:
        """Split the forward's experts into groups the cache can hold at once.

        A token's output is the weighted sum of its experts' outputs, so the
        sum can be taken a few experts at a time and accumulated: run the
        kernel once per group with an ``expert_map`` that hides the others,
        then add the results. Every (token, expert) pair is still computed
        exactly once, in whichever group owns its expert.

        Cost is ``ceil(experts_used / capacity)`` kernel launches, independent
        of how many tokens are in the batch, and each expert is fetched at most
        once per forward. Splitting the token axis instead costs one launch per
        chunk -- one per token once capacity approaches ``top_k`` -- and
        refetches experts as the cache thrashes.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            Groups of global expert ids, each no larger than ``capacity``. A
            single group when everything already fits, which is the common
            case.
        """
        unique = topk_ids.unique().tolist()
        if len(unique) <= self.capacity:
            return [unique]
        return [
            unique[i : i + self.capacity] for i in range(0, len(unique), self.capacity)
        ]

    @torch.compiler.disable
    def prepare(
        self, topk_ids: torch.Tensor, unique_ids: list[int] | None = None
    ) -> ExpertWeightResult:
        """Make a set of experts resident and return the map selecting them.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs. Only
                read when ``unique_ids`` is not supplied.
            unique_ids: The experts to make resident. Both planners already
                know this, and passing it avoids a device synchronization --
                which matters, because a split forward calls this once per
                piece.

        Returns:
            ExpertWeightResult holding the GPU buffers and an ``expert_map``
            exposing exactly these experts, everything else -1.

        Raises:
            RuntimeError: if more experts are requested than the cache holds.
        """
        if unique_ids is None:
            unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but --moe-expert-cache-size={self.capacity}. "
                f"Set --moe-expert-cache-size >= {len(unique_ids)}."
            )

        # Experts requested here must never be evicted to make room for
        # another one in the same call -- their slot would be handed to a
        # different expert while they are still expected to be resident. A
        # freshly loaded expert has freq=1, exactly the lowest LFRU score, so
        # it is the first eviction candidate. The map built at the end reads
        # every requested expert back out of _lru, so violating this raises
        # rather than corrupting silently, but it must not happen at all.
        needed = set(unique_ids)

        for expert_id in unique_ids:
            if expert_id in self._lru:
                event = self._pending.pop(expert_id, None)
                if event is not None:
                    torch.cuda.current_stream(self.device).wait_event(event)
                    self.prefetch_waits += 1
                deadlines = self._reserved_deadlines.get(expert_id)
                if deadlines and (
                    self._current_ordinal is None
                    or deadlines[0] <= self._current_ordinal
                ):
                    self.prefetch_useful += 1
                    deadlines.pop(0)
                    if not deadlines:
                        del self._reserved_deadlines[expert_id]
                # Cache hit: update frequency and recency
                self._clock += 1
                entry = self._lru[expert_id]
                entry[1] += 1  # freq
                entry[2] = self._clock  # last access
                self.hits += 1
            else:
                # Cache miss: need to load expert
                slot = self._take_slot(needed, speculative_deadline=None)
                # len(unique_ids) <= capacity guarantees that some slot is
                # available once any pending speculative copy is completed.
                assert slot is not None
                self._copy_expert(expert_id, slot, needed)

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self.misses += 1

        self._prepare_calls += 1
        if self._prepare_calls % _STATS_LOG_INTERVAL == 0:
            total = self.hits + self.misses
            if total > 0:
                logger.debug(
                    "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                    self.hits,
                    self.misses,
                    100.0 * self.hits / total,
                )

        # Expose exactly this group. Blocking on purpose: the host mirror is
        # rewritten by the next group, so an async copy could still be reading
        # it when that happens.
        self._mapping_host.fill_(-1)
        for expert_id in unique_ids:
            self._mapping_host[expert_id] = self._lru[expert_id][0]
        self._mapping.copy_(self._mapping_host)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            expert_map=self._mapping,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )


def run_with_expert_cache(
    provider: CachedWeightProvider,
    topk_ids: torch.Tensor,
    run: Callable[[ExpertWeightResult, slice, bool], torch.Tensor],
) -> torch.Tensor:
    """Evaluate a MoE forward through the expert cache.

    ``run`` receives the resident weights with the ``expert_map`` selecting
    them, the rows it should evaluate, and whether it should include work that
    belongs to the forward as a whole rather than to this call -- shared
    experts, most importantly. It sees the original ``topk_ids``; the map is
    what restricts each call to the resident experts.

    The two splits differ only in what is cut. ``"token"`` cuts rows and
    concatenates, so each token's sum stays inside one kernel call and the
    result matches the uncached path exactly. ``"expert"`` cuts the expert set
    and sums, which costs far fewer launches when capacity is small but rounds
    each group's partial sum to the model dtype.

    When everything fits -- the common case for both splits -- ``run`` is
    called exactly once and its result returned untouched.
    """
    if provider.split == "expert":
        groups = provider.plan_expert_groups(topk_ids)
        provider.begin_forward([expert for group in groups for expert in group])
        if len(groups) == 1:
            result = run(provider.prepare(topk_ids, groups[0]), _ALL_ROWS, True)
            provider.mark_used(groups[0])
            return result

        # Accumulate in fp32. It does not recover what each group already lost
        # rounding to the model dtype, but it keeps the sum from losing more.
        accumulator: torch.Tensor | None = None
        out_dtype: torch.dtype | None = None
        for i, expert_ids in enumerate(groups):
            part = run(provider.prepare(topk_ids, expert_ids), _ALL_ROWS, i == 0)
            provider.mark_used(expert_ids)
            if accumulator is None:
                accumulator, out_dtype = part.float(), part.dtype
            else:
                accumulator += part.float()
        assert accumulator is not None and out_dtype is not None
        return accumulator.to(out_dtype)

    plan = provider.plan_chunks(topk_ids)
    provider.begin_forward(
        sorted({expert for _, unique_ids in plan for expert in unique_ids})
    )
    if len(plan) == 1:
        rows, unique_ids = plan[0]
        result = run(provider.prepare(topk_ids, unique_ids), rows, True)
        provider.mark_used(unique_ids)
        return result
    parts = []
    for rows, unique_ids in plan:
        parts.append(run(provider.prepare(topk_ids[rows], unique_ids), rows, True))
        provider.mark_used(unique_ids)
    return torch.cat(parts, dim=0)
