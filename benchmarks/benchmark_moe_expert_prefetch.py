# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure reactive and trace-driven MoE expert caching on one request."""

import argparse
import hashlib
import json
import time

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write a Python merge sort.")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--cache-split", choices=("token", "expert"), default="token")
    parser.add_argument("--moe-backend", default="triton")
    parser.add_argument("--trace-output")
    parser.add_argument("--prefetch-trace")
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--budget-mb", type=float, default=0)
    parser.add_argument("--max-inflight-mb", type=float, default=0)
    parser.add_argument("--expert-storage-path")
    parser.add_argument("--host-cache-size", type=int, default=0)
    parser.add_argument("--storage-budget-mb", type=float, default=0)
    parser.add_argument("--storage-max-inflight-mb", type=float, default=0)
    parser.add_argument("--storage-workers", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        moe_expert_cache_size=args.cache_size,
        moe_expert_cache_split=args.cache_split,
        moe_backend=args.moe_backend,
        moe_expert_trace_output=args.trace_output,
        moe_expert_prefetch_trace=args.prefetch_trace,
        moe_expert_prefetch_lookahead=args.lookahead,
        moe_expert_prefetch_budget_mb=args.budget_mb,
        moe_expert_prefetch_max_inflight_mb=args.max_inflight_mb,
        moe_expert_storage_path=args.expert_storage_path,
        moe_expert_host_cache_size=args.host_cache_size,
        moe_expert_storage_prefetch_budget_mb=args.storage_budget_mb,
        moe_expert_storage_max_inflight_mb=args.storage_max_inflight_mb,
        moe_expert_storage_workers=args.storage_workers,
        enforce_eager=args.enforce_eager,
    )
    init_seconds = time.perf_counter() - init_start

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        seed=args.seed,
    )
    generate_start = time.perf_counter()
    request_output = llm.generate([args.prompt], sampling)[0]
    generate_seconds = time.perf_counter() - generate_start
    token_ids = request_output.outputs[0].token_ids
    metrics = request_output.metrics
    decode_seconds = None
    decode_tokens_per_second = None
    if (
        metrics is not None
        and len(token_ids) > 1
        and metrics.last_token_ts > metrics.first_token_ts
    ):
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        decode_tokens_per_second = (len(token_ids) - 1) / decode_seconds
    token_bytes = ",".join(str(token_id) for token_id in token_ids).encode()
    result = {
        "model": args.model,
        "cache_size": args.cache_size,
        "trace_output": args.trace_output,
        "prefetch_trace": args.prefetch_trace,
        "lookahead": args.lookahead,
        "budget_mb": args.budget_mb,
        "max_inflight_mb": args.max_inflight_mb,
        "expert_storage_path": args.expert_storage_path,
        "host_cache_size": args.host_cache_size,
        "storage_budget_mb": args.storage_budget_mb,
        "storage_max_inflight_mb": args.storage_max_inflight_mb,
        "storage_workers": args.storage_workers,
        "init_seconds": init_seconds,
        "generate_seconds": generate_seconds,
        "output_tokens": len(token_ids),
        "output_tokens_per_second": len(token_ids) / generate_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tokens_per_second,
        "first_token_latency": (
            metrics.first_token_latency if metrics is not None else None
        ),
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "finish_reason": request_output.outputs[0].finish_reason,
        "text_preview": request_output.outputs[0].text[:500],
    }
    print("MOE_PREFETCH_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
