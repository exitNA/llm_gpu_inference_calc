from __future__ import annotations

from typing import Any

from .calculations import (
    determine_dominant_constraints,
    estimate_activated_params_billion,
    estimate_capacity_backprojection,
    estimate_memory_based_gpu_count,
    estimate_request_stats,
    estimate_throughput_based_gpu_count,
)
from .helpers import bytes_to_gib, format_adaptive_token_volume, format_calc_number, round_optional
from .models import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig
from .process import build_calculation_process_sections, format_calculation_process_text
from .validation import validate_runtime_config, validate_traffic_config


def evaluate_single_model(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, Any]:
    validate_traffic_config(traffic)
    validate_runtime_config(runtime)

    request_stats = estimate_request_stats(traffic)
    memory_info = estimate_memory_based_gpu_count(model, traffic, gpu, runtime)
    throughput_info = estimate_throughput_based_gpu_count(model, traffic, gpu, runtime)

    candidate_counts = [int(memory_info["gpu_count_by_memory"])]
    if throughput_info["prefill_gpu_count_by_throughput"] is not None:
        candidate_counts.append(int(throughput_info["prefill_gpu_count_by_throughput"]))
    if throughput_info["decode_gpu_count_by_throughput"] is not None:
        candidate_counts.append(int(throughput_info["decode_gpu_count_by_throughput"]))
    business_gpu_count = max(candidate_counts)

    capacity_info = estimate_capacity_backprojection(
        traffic=traffic,
        memory_info=memory_info,
        throughput_info=throughput_info,
        business_gpu_count=business_gpu_count,
    )

    dominant_constraints = determine_dominant_constraints(
        gpu_count_by_memory=int(memory_info["gpu_count_by_memory"]),
        g_pre=throughput_info["prefill_gpu_count_by_throughput"],
        g_dec=throughput_info["decode_gpu_count_by_throughput"],
        business_gpu_count=business_gpu_count,
    )

    estimated_total_cost = None
    if gpu.unit_price is not None:
        estimated_total_cost = business_gpu_count * gpu.unit_price

    result = {
        "model_config": model,
        "gpu_config": gpu,
        "traffic_config": traffic,
        "runtime_config": runtime,
        "model_name": model.model_name,
        "gpu_name": gpu.gpu_name,
        "precision": runtime.precision,
        "kv_cache_dtype": runtime.kv_cache_dtype,
        "arch_family": model.arch_family,
        "attention_type": model.attention_type,
        "activated_params_billion": round_optional(estimate_activated_params_billion(model), 2),
        "dominant_constraints": dominant_constraints,
        **request_stats,
        **memory_info,
        **throughput_info,
        **capacity_info,
        "weight_with_overhead_gib": round(bytes_to_gib(float(memory_info["weight_bytes"])), 2),
        "runtime_overhead_gib": round(bytes_to_gib(float(memory_info["runtime_fixed_bytes"])), 2),
        "p95_kv_gib_per_request": round(bytes_to_gib(float(memory_info["p95_cache_bytes_per_request"])), 2),
        "p95_total_memory_gib": round(bytes_to_gib(float(memory_info["total_memory_bytes"])), 2),
        "usable_vram_gib_per_gpu": round(bytes_to_gib(float(memory_info["usable_vram_bytes_per_gpu"])), 2),
        "memory_for_sizing_gib": round(bytes_to_gib(float(memory_info["total_memory_bytes"])), 2),
        "memory_sizing_basis": "peak_qps_plus_p95_length",
        "daily_decode_token_capacity": round_optional(capacity_info["daily_decode_token_capacity_p95"], 1),
        "daily_prefill_token_capacity": round_optional(capacity_info["daily_prefill_token_capacity_p95"], 1),
        "cluster_decode_tps_capacity": round_optional(capacity_info["cluster_decode_tps_capacity"], 2),
        "cluster_prefill_tps_capacity": round_optional(capacity_info["cluster_prefill_tps_capacity_p95"], 2),
        "g_memory": int(memory_info["gpu_count_by_memory"]),
        "g_prefill": throughput_info["prefill_gpu_count_by_throughput"],
        "g_decode": throughput_info["decode_gpu_count_by_throughput"],
        "g_biz": business_gpu_count,
        "business_gpu_count": business_gpu_count,
        "estimated_total_cost": estimated_total_cost,
        "unit_price": gpu.unit_price,
    }

    result["request_profile_rows"] = [
        {
            "name": "峰值/P95口径",
            "qps": traffic.lambda_peak_qps,
            "input_tokens": traffic.p95_input_tokens,
            "output_tokens": traffic.p95_output_tokens,
            "total_tokens": traffic.p95_total_tokens,
            "ttft_target_sec": traffic.ttft_p95_target_sec,
            "e2e_target_sec": traffic.e2e_p95_target_sec,
        },
    ]
    result["kv_profile_rows"] = [
        {
            "name": "P95长度",
            "seq_len_total": traffic.p95_total_tokens,
            "kv_cache_gib_per_request": result["p95_kv_gib_per_request"],
            "max_concurrency_by_memory": capacity_info["max_concurrency_by_memory_p95"],
        },
    ]

    calculation_process_sections = build_calculation_process_sections(result)
    result["calculation_process_sections"] = calculation_process_sections
    result["calculation_process_text"] = format_calculation_process_text(calculation_process_sections)
    return result


def format_result_text(result: dict[str, Any]) -> str:
    lines = [
        "=" * 88,
        f"模型: {result['model_name']}",
        f"GPU:  {result['gpu_name']} ({result['precision']}/{result['kv_cache_dtype']})",
        "=" * 88,
        "",
        "[主结果]",
        f"- G_req / 业务基线: {result['business_gpu_count']} 卡",
        f"- 主导约束: {' / '.join(result['dominant_constraints'])}",
        "",
        "[约束拆解]",
        f"- G_mem: {result['gpu_count_by_memory']}",
        f"- G_pre: {result['prefill_gpu_count_by_throughput']}",
        f"- G_dec: {result['decode_gpu_count_by_throughput']}",
        f"- Prefill 时延必要条件: {'满足' if result['prefill_latency_ok'] else '不满足'}",
        f"- Decode 时延必要条件: {'满足' if result['decode_latency_ok'] else '不满足'}",
        "",
        "[能力回推]",
        f"- 保守可持续 QPS: {format_calc_number(result['sustainable_qps_p95'])} req/s",
        f"- 保守每日 Prefill token: {format_adaptive_token_volume(result['daily_prefill_token_capacity_p95'])}",
        f"- 保守每日 Decode token: {format_adaptive_token_volume(result['daily_decode_token_capacity_p95'])}",
        f"- 显存最大在途请求量(P95): {result['max_concurrency_by_memory_p95']}",
        f"- 时延风险: {result['latency_risk_level']} ({result['latency_risk_note']})",
        "",
        "[计算过程]",
        result["calculation_process_text"],
    ]
    return "\n".join(lines)
