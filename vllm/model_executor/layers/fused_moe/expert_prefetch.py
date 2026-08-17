# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-driven expert prefetch coordination."""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIB = 1024 * 1024
_SUMMARY_INTERVAL = 1000


@dataclass
class PrefetchOutcome:
    """Result of one provider prefetch admission decision."""

    requested: int
    resident: int
    loaded: int
    bytes_enqueued: int
    event: torch.cuda.Event | None = None


class ExpertPrefetchSchedule:
    """Sparse-call predictions loaded from a JSONL route trace."""

    def __init__(self, path: str) -> None:
        self.events: dict[tuple[int, str], tuple[int, ...]] = {}
        self.layer_order: list[str] = []
        seen_layers: set[str] = set()

        with Path(path).open(encoding="utf-8") as src:
            for line_number, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    step = int(item["step"])
                    layer = str(item["layer"])
                    experts = tuple(sorted({int(e) for e in item["experts"]}))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid expert trace record at {path}:{line_number}"
                    ) from exc
                key = (step, layer)
                if key in self.events:
                    raise ValueError(
                        f"Duplicate expert trace event step={step}, layer={layer!r}"
                    )
                self.events[key] = experts
                if layer not in seen_layers:
                    seen_layers.add(layer)
                    self.layer_order.append(layer)

        if not self.events:
            raise ValueError(f"Expert prefetch trace is empty: {path}")

    def get(self, step: int, layer: str) -> tuple[int, ...] | None:
        return self.events.get((step, layer))


class ExpertPrefetchCoordinator:
    """Spend a bounded H2D budget on earliest trace-predicted experts."""

    def __init__(
        self,
        *,
        trace_path: str | None,
        trace_output: str | None,
        lookahead: int,
        budget_mb: float,
        max_inflight_mb: float,
    ) -> None:
        self.schedule = (
            ExpertPrefetchSchedule(_rank_path(trace_path)) if trace_path else None
        )
        self.lookahead = lookahead
        self.budget_bytes = int(budget_mb * _MIB)
        self.max_inflight_bytes = int(max_inflight_mb * _MIB)
        self.providers: dict[str, Any] = {}
        self._layer_steps: dict[str, int] = {}
        self._scheduled: set[tuple[int, str]] = set()
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._inflight: list[tuple[torch.cuda.Event, int]] = []
        self._inflight_bytes = 0
        self._observations = 0
        self.prefetch_bytes = 0
        self.prefetch_experts = 0
        self.budget_blocked = 0
        self.capacity_blocked = 0
        self._trace_handle: TextIO | None = None
        self._closed = False

        if trace_output:
            output_path = Path(_rank_path(trace_output))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_handle = output_path.open("w", encoding="utf-8", buffering=1)
        atexit.register(self.close)

    def register(self, layer_name: str, provider: Any) -> None:
        if layer_name in self.providers:
            raise ValueError(f"Duplicate expert-cache layer name: {layer_name}")
        self.providers[layer_name] = provider

    def begin_forward(self, layer_name: str, actual_experts: list[int]) -> None:
        step = self._layer_steps.get(layer_name, 0)
        self._layer_steps[layer_name] = step + 1
        self._observations += 1

        if self._trace_handle is not None:
            record = {
                "step": step,
                "layer": layer_name,
                "experts": sorted(set(actual_experts)),
            }
            self._trace_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

        if self.schedule is None or layer_name not in self.schedule.layer_order:
            return

        layer_count = len(self.schedule.layer_order)
        layer_index = self.schedule.layer_order.index(layer_name)
        current_ordinal = step * layer_count + layer_index
        provider = self.providers.get(layer_name)
        if provider is not None:
            provider.expire_predictions(current_ordinal, set(actual_experts))

        self._reap_inflight()
        remaining = self.budget_bytes
        if self.max_inflight_bytes > 0:
            remaining = min(
                remaining,
                max(0, self.max_inflight_bytes - self._inflight_bytes),
            )
        if remaining <= 0:
            self.budget_blocked += 1
            return

        for distance in range(1, self.lookahead + 1):
            target_ordinal = current_ordinal + distance
            target_step, target_index = divmod(target_ordinal, layer_count)
            target_layer = self.schedule.layer_order[target_index]
            key = (target_step, target_layer)
            if key in self._scheduled:
                continue
            target_provider = self.providers.get(target_layer)
            if target_provider is None or target_provider is provider:
                continue
            predicted = self.schedule.get(target_step, target_layer)
            if predicted is None:
                continue

            stream = self._stream_for(target_provider.device)
            outcome = target_provider.prefetch(
                list(predicted),
                deadline=target_ordinal,
                max_bytes=remaining,
                stream=stream,
            )
            remaining -= outcome.bytes_enqueued
            self.prefetch_bytes += outcome.bytes_enqueued
            self.prefetch_experts += outcome.loaded
            if outcome.event is not None and outcome.bytes_enqueued > 0:
                self._inflight.append((outcome.event, outcome.bytes_enqueued))
                self._inflight_bytes += outcome.bytes_enqueued
            if outcome.resident == outcome.requested:
                self._scheduled.add(key)
            elif outcome.requested > outcome.resident:
                self.capacity_blocked += 1
            if remaining < target_provider.bytes_per_expert:
                self.budget_blocked += 1

        if self._observations % _SUMMARY_INTERVAL == 0:
            self.log_summary()

    def _stream_for(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._streams[device] = stream
        return stream

    def _reap_inflight(self) -> None:
        pending: list[tuple[torch.cuda.Event, int]] = []
        inflight_bytes = 0
        for event, byte_count in self._inflight:
            if not event.query():
                pending.append((event, byte_count))
                inflight_bytes += byte_count
        self._inflight = pending
        self._inflight_bytes = inflight_bytes

    def log_summary(self) -> None:
        useful = sum(p.prefetch_useful for p in self.providers.values())
        wasted = sum(p.prefetch_wasted for p in self.providers.values())
        waits = sum(p.prefetch_waits for p in self.providers.values())
        logger.info(
            "Expert prefetch: %.3f GiB, %d loaded, %d useful, %d wasted, "
            "%d demand waits, %d budget blocks, %d capacity blocks",
            self.prefetch_bytes / (1024**3),
            self.prefetch_experts,
            useful,
            wasted,
            waits,
            self.budget_blocked,
            self.capacity_blocked,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
        if self._observations > 0 and self.schedule is not None:
            self.log_summary()


_COORDINATORS: dict[int, ExpertPrefetchCoordinator] = {}


def get_expert_prefetch_coordinator(config: Any) -> ExpertPrefetchCoordinator | None:
    """Return the process-local coordinator associated with an offload config."""
    trace_path = config.moe_expert_prefetch_trace
    trace_output = config.moe_expert_trace_output
    if not trace_path and not trace_output:
        return None
    key = id(config)
    coordinator = _COORDINATORS.get(key)
    if coordinator is None:
        coordinator = ExpertPrefetchCoordinator(
            trace_path=trace_path,
            trace_output=trace_output,
            lookahead=config.moe_expert_prefetch_lookahead,
            budget_mb=config.moe_expert_prefetch_budget_mb,
            max_inflight_mb=config.moe_expert_prefetch_max_inflight_mb,
        )
        _COORDINATORS[key] = coordinator
    return coordinator


def _rank_path(path: str) -> str:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "{rank}" in path:
        return path.format(rank=rank)
    if world_size <= 1:
        return path
    output = Path(path)
    return str(output.with_name(f"{output.stem}.rank{rank}{output.suffix}"))
