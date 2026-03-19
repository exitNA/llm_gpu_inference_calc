from __future__ import annotations

from .constants import SECONDS_PER_DAY
from .helpers import ceil_div, divide_optional, floor_div, multiply_optional, precision_to_bytes, vendor_vram_gb_to_bytes
from .models import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig


def resolve_head_dim(model: ModelConfig) -> int | None:
    if model.head_dim is not None:
        return model.head_dim
    if model.num_heads and model.hidden_size % model.num_heads == 0:
        return model.hidden_size // model.num_heads
    return None


def estimate_activated_params_billion(model: ModelConfig) -> float:
    if model.arch_family.lower() == "moe":
        if model.activated_params_billion is None:
            raise ValueError("MoE 模型需要 activated_params_billion")
        return model.activated_params_billion
    return model.activated_params_billion or model.num_params_billion


def get_peak_compute_tflops(gpu: GPUConfig, precision: str) -> float | None:
    precision_key = precision.lower()
    if precision_key == "fp32":
        return gpu.fp32_tflops
    if precision_key == "fp16":
        return gpu.fp16_tflops
    if precision_key == "bf16":
        return gpu.bf16_tflops
    if precision_key == "fp8":
        return gpu.fp8_tflops
    if precision_key == "int8":
        return gpu.int8_tflops
    if precision_key == "int4":
        return gpu.int4_tflops or gpu.int8_tflops
    return None


def estimate_cache_bytes_per_token_per_layer(model: ModelConfig, runtime: RuntimeConfig) -> float:
    if model.cache_bytes_per_token_per_layer is not None:
        return model.cache_bytes_per_token_per_layer

    kv_bytes = precision_to_bytes(runtime.kv_cache_dtype)
    attention_type = model.attention_type.lower()
    head_dim = resolve_head_dim(model)

    if attention_type == "mha":
        return 2 * model.hidden_size * kv_bytes
    if attention_type == "gqa":
        if model.num_kv_heads is None or head_dim is None:
            raise ValueError("GQA 模型需要 num_kv_heads 和可解析的 head_dim")
        return 2 * model.num_kv_heads * head_dim * kv_bytes
    if attention_type == "mqa":
        if head_dim is None:
            raise ValueError("MQA 模型需要 head_dim 或可解析的 num_heads")
        return 2 * head_dim * kv_bytes
    if attention_type == "mla":
        if model.latent_cache_dim is None:
            raise ValueError("MLA 模型需要 latent_cache_dim")
        return model.latent_cache_dim * kv_bytes + model.cache_aux_bytes_per_token_per_layer
    if attention_type in {"sparse", "hybrid", "hybrid_attention", "sparse_attention"}:
        raise ValueError("Sparse/Hybrid attention 需要显式提供 cache_bytes_per_token_per_layer")
    raise ValueError(f"Unsupported attention_type: {model.attention_type}")


def estimate_attention_compute_coefficient(model: ModelConfig, runtime: RuntimeConfig) -> float:
    if runtime.attention_compute_coefficient is not None:
        return runtime.attention_compute_coefficient

    head_dim = resolve_head_dim(model)
    attention_type = model.attention_type.lower()
    if attention_type == "mla" and model.latent_cache_dim is not None:
        return 2.0 * model.num_layers * model.latent_cache_dim
    if model.num_kv_heads is not None and head_dim is not None:
        return 4.0 * model.num_layers * model.num_kv_heads * head_dim
    return 4.0 * model.num_layers * model.hidden_size


def estimate_request_stats(traffic: TrafficConfig) -> dict[str, float]:
    return {
        "p95_input_tokens": float(traffic.p95_input_tokens),
        "p95_output_tokens": float(traffic.p95_output_tokens),
        "p95_total_tokens": float(traffic.p95_total_tokens),
        "decode_p95_budget_sec": traffic.decode_p95_budget_sec,
    }


def estimate_workload_targets(traffic: TrafficConfig) -> dict[str, float]:
    c_peak_budget = traffic.lambda_peak_qps * traffic.e2e_p95_target_sec * traffic.concurrency_safety_factor
    return {
        "tps_pre_target_peak": traffic.lambda_peak_qps * traffic.p95_input_tokens,
        "tps_dec_target_peak": traffic.lambda_peak_qps * traffic.p95_output_tokens,
        "c_peak_budget": c_peak_budget,
    }


def estimate_weight_memory(model: ModelConfig, runtime: RuntimeConfig) -> dict[str, float]:
    total_params = model.num_params_billion * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    raw_weight_bytes = total_params * bytes_per_param
    weight_bytes = raw_weight_bytes * (1.0 + runtime.weight_overhead_ratio)
    runtime_bytes = weight_bytes * runtime.runtime_overhead_ratio
    return {
        "raw_weight_bytes": raw_weight_bytes,
        "weight_bytes": weight_bytes,
        "runtime_fixed_bytes": runtime_bytes,
    }


def estimate_cache_memory(
    model: ModelConfig,
    traffic: TrafficConfig,
    runtime: RuntimeConfig,
    c_peak_budget: float,
) -> dict[str, float]:
    cache_bytes_per_token_per_layer = estimate_cache_bytes_per_token_per_layer(model, runtime)
    p95_cache_bytes_per_request = model.num_layers * traffic.p95_total_tokens * cache_bytes_per_token_per_layer
    cache_total_peak_bytes = c_peak_budget * p95_cache_bytes_per_request
    return {
        "cache_bytes_per_token_per_layer": cache_bytes_per_token_per_layer,
        "p95_cache_bytes_per_request": p95_cache_bytes_per_request,
        "cache_total_peak_bytes": cache_total_peak_bytes,
    }


def estimate_memory_based_gpu_count(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, float | int]:
    weight_info = estimate_weight_memory(model, runtime)
    workload_info = estimate_workload_targets(traffic)
    cache_info = estimate_cache_memory(model, traffic, runtime, workload_info["c_peak_budget"])

    total_memory_bytes = weight_info["weight_bytes"] + weight_info["runtime_fixed_bytes"] + cache_info["cache_total_peak_bytes"]
    usable_vram_bytes_per_gpu = vendor_vram_gb_to_bytes(gpu.vram_gb) * runtime.usable_vram_ratio
    gpu_count_by_memory = ceil_div(total_memory_bytes, usable_vram_bytes_per_gpu)

    return {
        **weight_info,
        **cache_info,
        "total_memory_bytes": total_memory_bytes,
        "usable_vram_bytes_per_gpu": usable_vram_bytes_per_gpu,
        "gpu_count_by_memory": gpu_count_by_memory,
    }


def estimate_prefill_tps_per_gpu(
    model: ModelConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
    input_tokens: int,
) -> dict[str, float | None]:
    activated_params = estimate_activated_params_billion(model) * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    bytes_per_token = None
    tps_bw = None
    if input_tokens > 0:
        bytes_per_token = activated_params * bytes_per_param / input_tokens
        if gpu.memory_bandwidth_gb_per_sec and bytes_per_token > 0:
            tps_bw = gpu.memory_bandwidth_gb_per_sec * 1e9 * runtime.bandwidth_efficiency / bytes_per_token

    peak_compute_tflops = get_peak_compute_tflops(gpu, runtime.precision)
    attention_coeff = estimate_attention_compute_coefficient(model, runtime)
    tps_cmp = None
    if peak_compute_tflops and peak_compute_tflops > 0:
        denom = (2.0 * activated_params) + (attention_coeff * input_tokens)
        if denom > 0:
            tps_cmp = peak_compute_tflops * 1e12 * runtime.compute_efficiency / denom

    candidates = [value for value in (tps_bw, tps_cmp) if value is not None]
    tps_card = min(candidates) if candidates else None
    return {
        "prefill_bytes_per_token": bytes_per_token,
        "prefill_flops_attention_coeff": attention_coeff,
        "prefill_tps_bw_limited": tps_bw,
        "prefill_tps_compute_limited": tps_cmp,
        "prefill_tps_card": tps_card,
    }


def estimate_decode_tps_per_gpu(
    model: ModelConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, float | None]:
    activated_params = estimate_activated_params_billion(model) * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    bytes_per_token = activated_params * bytes_per_param
    flops_per_token = 2.0 * activated_params

    tps_bw = None
    if gpu.memory_bandwidth_gb_per_sec and bytes_per_token > 0:
        tps_bw = gpu.memory_bandwidth_gb_per_sec * 1e9 * runtime.bandwidth_efficiency / bytes_per_token

    peak_compute_tflops = get_peak_compute_tflops(gpu, runtime.precision)
    tps_cmp = None
    if peak_compute_tflops and peak_compute_tflops > 0 and flops_per_token > 0:
        tps_cmp = peak_compute_tflops * 1e12 * runtime.compute_efficiency / flops_per_token

    candidates = [value for value in (tps_bw, tps_cmp) if value is not None]
    tps_card = min(candidates) if candidates else None
    return {
        "decode_bytes_per_token": bytes_per_token,
        "decode_flops_per_token": flops_per_token,
        "decode_tps_bw_limited": tps_bw,
        "decode_tps_compute_limited": tps_cmp,
        "decode_tps_card": tps_card,
        "decode_ms_per_token": divide_optional(1000.0, tps_card),
    }


def estimate_throughput_based_gpu_count(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, float | int | None]:
    workload_info = estimate_workload_targets(traffic)
    prefill_p95 = estimate_prefill_tps_per_gpu(model, gpu, runtime, traffic.p95_input_tokens)
    decode = estimate_decode_tps_per_gpu(model, gpu, runtime)

    g_pre = None
    if prefill_p95["prefill_tps_card"] and prefill_p95["prefill_tps_card"] > 0:
        g_pre = ceil_div(workload_info["tps_pre_target_peak"], prefill_p95["prefill_tps_card"])

    g_dec = None
    if decode["decode_tps_card"] and decode["decode_tps_card"] > 0:
        g_dec = ceil_div(workload_info["tps_dec_target_peak"], decode["decode_tps_card"])

    return {
        **workload_info,
        "prefill_tps_p95_bw_limited": prefill_p95["prefill_tps_bw_limited"],
        "prefill_tps_p95_compute_limited": prefill_p95["prefill_tps_compute_limited"],
        "prefill_tps_p95_card": prefill_p95["prefill_tps_card"],
        "prefill_bytes_per_token_p95": prefill_p95["prefill_bytes_per_token"],
        "prefill_flops_attention_coeff": prefill_p95["prefill_flops_attention_coeff"],
        "decode_tps_bw_limited": decode["decode_tps_bw_limited"],
        "decode_tps_compute_limited": decode["decode_tps_compute_limited"],
        "decode_tps_card": decode["decode_tps_card"],
        "decode_bytes_per_token": decode["decode_bytes_per_token"],
        "decode_flops_per_token": decode["decode_flops_per_token"],
        "decode_ms_per_token": decode["decode_ms_per_token"],
        "prefill_gpu_count_by_throughput": g_pre,
        "decode_gpu_count_by_throughput": g_dec,
    }


def estimate_latency_necessity(
    traffic: TrafficConfig,
    throughput_info: dict[str, float | int | None],
) -> dict[str, float | bool | None]:
    prefill_tps_p95 = throughput_info["prefill_tps_p95_card"]
    decode_tps = throughput_info["decode_tps_card"]
    prefill_time = divide_optional(traffic.p95_input_tokens, prefill_tps_p95)
    decode_time = divide_optional(traffic.p95_output_tokens, decode_tps)
    prefill_ok = prefill_time is not None and prefill_time <= traffic.ttft_p95_target_sec
    decode_ok = decode_time is not None and decode_time <= traffic.decode_p95_budget_sec
    return {
        "prefill_latency_check_sec": prefill_time,
        "decode_latency_check_sec": decode_time,
        "prefill_latency_ok": prefill_ok,
        "decode_latency_ok": decode_ok,
    }


def estimate_capacity_backprojection(
    traffic: TrafficConfig,
    memory_info: dict[str, float | int],
    throughput_info: dict[str, float | int | None],
    business_gpu_count: int,
) -> dict[str, float | int | None | str]:
    prefill_cap_p95 = multiply_optional(throughput_info["prefill_tps_p95_card"], business_gpu_count)
    decode_cap = multiply_optional(throughput_info["decode_tps_card"], business_gpu_count)

    lambda_pre_p95 = divide_optional(prefill_cap_p95 or 0.0, traffic.p95_input_tokens)
    lambda_dec_p95 = divide_optional(decode_cap or 0.0, traffic.p95_output_tokens)
    lambda_p95 = min(lambda_pre_p95, lambda_dec_p95) if lambda_pre_p95 is not None and lambda_dec_p95 is not None else None

    daily_prefill_tokens_p95 = multiply_optional(prefill_cap_p95, SECONDS_PER_DAY)
    daily_decode_tokens_p95 = multiply_optional(decode_cap, SECONDS_PER_DAY)
    daily_requests_p95 = multiply_optional(lambda_p95, SECONDS_PER_DAY)

    cluster_effective_vram_bytes = business_gpu_count * float(memory_info["usable_vram_bytes_per_gpu"])
    cache_avail_bytes = cluster_effective_vram_bytes - float(memory_info["weight_bytes"]) - float(memory_info["runtime_fixed_bytes"])
    p95_cache_per_request = float(memory_info["p95_cache_bytes_per_request"])
    c_max_p95 = max(floor_div(cache_avail_bytes, p95_cache_per_request), 0) if p95_cache_per_request > 0 else None

    rho_conc_p95 = None
    c_peak_budget = float(throughput_info["c_peak_budget"])
    if c_peak_budget > 0 and c_max_p95 is not None:
        rho_conc_p95 = c_max_p95 / c_peak_budget

    latency_info = estimate_latency_necessity(traffic, throughput_info)
    if not latency_info["prefill_latency_ok"] or not latency_info["decode_latency_ok"]:
        latency_risk = "high"
        latency_note = "单卡时延必要条件不满足，仅增加总卡数通常无法直接解决时延目标。"
    elif lambda_p95 is not None and traffic.lambda_peak_qps >= lambda_p95 * 0.9:
        latency_risk = "medium"
        latency_note = "总量能力接近保守可持续 QPS，上线后排队与调度可能放大实际时延。"
    else:
        latency_risk = "low"
        latency_note = "单卡时延必要条件满足，且峰值流量低于保守可持续 QPS。"

    return {
        "cluster_prefill_tps_capacity_p95": prefill_cap_p95,
        "cluster_decode_tps_capacity": decode_cap,
        "sustainable_qps_p95": lambda_p95,
        "sustainable_prefill_qps_p95": lambda_pre_p95,
        "sustainable_decode_qps_p95": lambda_dec_p95,
        "daily_prefill_token_capacity_p95": daily_prefill_tokens_p95,
        "daily_decode_token_capacity_p95": daily_decode_tokens_p95,
        "daily_request_capacity_p95": daily_requests_p95,
        "cluster_effective_vram_bytes": cluster_effective_vram_bytes,
        "cache_available_bytes": cache_avail_bytes,
        "max_concurrency_by_memory_p95": c_max_p95,
        "concurrency_margin_ratio_p95": rho_conc_p95,
        "latency_risk_level": latency_risk,
        "latency_risk_note": latency_note,
        **latency_info,
    }


def determine_dominant_constraints(
    gpu_count_by_memory: int,
    g_pre: int | None,
    g_dec: int | None,
    business_gpu_count: int,
) -> list[str]:
    dominant: list[str] = []
    if gpu_count_by_memory == business_gpu_count:
        dominant.append("显存")
    if g_pre == business_gpu_count:
        dominant.append("Prefill 吞吐")
    if g_dec == business_gpu_count:
        dominant.append("Decode 吞吐")
    return dominant or ["显存"]
