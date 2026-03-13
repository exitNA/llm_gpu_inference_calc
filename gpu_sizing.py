from dataclasses import dataclass
from math import ceil
from typing import Any

PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class RequestShape:
    name: str
    ratio: float
    avg_input_tokens: int
    avg_output_tokens: int


@dataclass
class ModelConfig:
    model_name: str
    num_params_billion: float
    num_layers: int
    hidden_size: int
    arch_family: str = "dense"
    attention_type: str = "mha"
    activated_params_billion: float | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    latent_cache_dim: int | None = None
    cache_aux_bytes_per_token_per_layer: float = 0.0
    cache_bytes_per_token_per_layer: float | None = None


@dataclass
class RuntimeConfig:
    precision: str = "bf16"
    kv_cache_dtype: str = "fp16"
    weight_overhead_ratio: float = 0.15
    runtime_overhead_ratio: float = 0.08
    runtime_overhead_gb: float = 2.0
    usable_vram_ratio: float = 0.90
    decode_efficiency: float = 0.40
    prefill_efficiency: float = 0.55
    compute_efficiency: float = 0.60
    prefill_memory_reuse_factor: float = 24.0


@dataclass
class TrafficConfig:
    concurrency: int
    target_decode_tps_total: float
    batch_size_per_request: int = 1
    target_prefill_tps_total: float | None = None
    request_shapes: list[RequestShape] | None = None

    @property
    def target_tokens_per_sec_total(self) -> float:
        return self.target_decode_tps_total

    @property
    def prefill_tokens_per_sec_total(self) -> float | None:
        return self.target_prefill_tps_total


@dataclass
class GPUConfig:
    gpu_name: str
    vram_gb: float
    memory_bandwidth_gb_per_sec: float | None = None
    fp32_tflops: float | None = None
    fp16_tflops: float | None = None
    bf16_tflops: float | None = None
    fp8_tflops: float | None = None
    int8_tflops: float | None = None
    int4_tflops: float | None = None
    unit_price: float | None = None


@dataclass
class HAConfig:
    ha_mode: str = "n_plus_1"
    replica_count: int = 2
    failover_reserve_ratio: float = 0.0


def precision_to_bytes(precision: str) -> float:
    precision_key = precision.lower()
    if precision_key not in PRECISION_BYTES:
        raise ValueError(f"Unsupported precision: {precision}")
    return PRECISION_BYTES[precision_key]


def bytes_to_gb(num_bytes: float) -> float:
    return num_bytes / 1e9


def ceil_div(a: float, b: float) -> int:
    return ceil(a / b)


def round_optional(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def validate_request_shapes(request_shapes: list[RequestShape]) -> None:
    if not request_shapes:
        raise ValueError("request_shapes 不能为空")

    total_ratio = sum(item.ratio for item in request_shapes)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"request_shapes 的 ratio 总和必须为 1.0，当前为 {total_ratio}")

    for item in request_shapes:
        if item.ratio < 0:
            raise ValueError(f"{item.name} 的 ratio 不能小于 0")
        if item.avg_input_tokens < 0 or item.avg_output_tokens < 0:
            raise ValueError(f"{item.name} 的 token 长度不能为负数")


def estimate_weight_memory_gb(model: ModelConfig, runtime: RuntimeConfig) -> dict[str, float]:
    num_params = model.num_params_billion * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    raw_weight_gb = bytes_to_gb(num_params * bytes_per_param)
    weight_with_overhead_gb = raw_weight_gb * (1.0 + runtime.weight_overhead_ratio)

    return {
        "num_params": num_params,
        "bytes_per_param": bytes_per_param,
        "raw_weight_gb": raw_weight_gb,
        "weight_with_overhead_gb": weight_with_overhead_gb,
    }


def estimate_sequence_distribution_stats(traffic: TrafficConfig) -> dict[str, Any]:
    if not traffic.request_shapes:
        raise ValueError("TrafficConfig.request_shapes 未提供")

    validate_request_shapes(traffic.request_shapes)

    avg_input = sum(item.ratio * item.avg_input_tokens for item in traffic.request_shapes)
    avg_output = sum(item.ratio * item.avg_output_tokens for item in traffic.request_shapes)
    avg_total = avg_input + avg_output

    detail = [
        {
            "name": item.name,
            "ratio": item.ratio,
            "avg_input_tokens": item.avg_input_tokens,
            "avg_output_tokens": item.avg_output_tokens,
            "seq_len_total": item.avg_input_tokens + item.avg_output_tokens,
        }
        for item in traffic.request_shapes
    ]

    sorted_shapes = sorted(
        traffic.request_shapes,
        key=lambda shape: shape.avg_input_tokens + shape.avg_output_tokens,
    )

    cumulative = 0.0
    p95_total_seq = None
    p95_input = None
    p95_output = None

    for item in sorted_shapes:
        cumulative += item.ratio
        if cumulative >= 0.95:
            p95_total_seq = item.avg_input_tokens + item.avg_output_tokens
            p95_input = item.avg_input_tokens
            p95_output = item.avg_output_tokens
            break

    return {
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "avg_total_tokens": avg_total,
        "p95_input_tokens": p95_input,
        "p95_output_tokens": p95_output,
        "p95_total_tokens": p95_total_seq,
        "request_shape_details": detail,
    }


def _resolve_head_dim(model: ModelConfig) -> int | None:
    if model.head_dim is not None:
        return model.head_dim
    if model.num_heads and model.hidden_size % model.num_heads == 0:
        return model.hidden_size // model.num_heads
    return None


def estimate_cache_bytes_per_token_per_layer(model: ModelConfig, runtime: RuntimeConfig) -> float:
    if model.cache_bytes_per_token_per_layer is not None:
        return model.cache_bytes_per_token_per_layer

    kv_bytes = precision_to_bytes(runtime.kv_cache_dtype)
    attention_type = model.attention_type.lower()
    head_dim = _resolve_head_dim(model)

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

    if attention_type in {"sparse", "hybrid", "sparse_attention", "hybrid_attention"}:
        raise ValueError("Sparse/Hybrid attention 需要显式提供 cache_bytes_per_token_per_layer")

    raise ValueError(f"Unsupported attention_type: {model.attention_type}")


def estimate_kv_cache_gb_for_seq(
    model: ModelConfig,
    runtime: RuntimeConfig,
    seq_len_total: float,
    batch_size_per_request: int,
) -> float:
    cache_bytes = (
        model.num_layers
        * seq_len_total
        * batch_size_per_request
        * estimate_cache_bytes_per_token_per_layer(model, runtime)
    )
    return bytes_to_gb(cache_bytes)


def estimate_kv_distribution_gb(model: ModelConfig, traffic: TrafficConfig, runtime: RuntimeConfig) -> dict[str, Any]:
    stats = estimate_sequence_distribution_stats(traffic)
    avg_kv_gb_per_request = estimate_kv_cache_gb_for_seq(
        model=model,
        runtime=runtime,
        seq_len_total=stats["avg_total_tokens"],
        batch_size_per_request=traffic.batch_size_per_request,
    )
    p95_kv_gb_per_request = estimate_kv_cache_gb_for_seq(
        model=model,
        runtime=runtime,
        seq_len_total=stats["p95_total_tokens"],
        batch_size_per_request=traffic.batch_size_per_request,
    )

    per_shape_kv = []
    for item in stats["request_shape_details"]:
        kv_gb = estimate_kv_cache_gb_for_seq(
            model=model,
            runtime=runtime,
            seq_len_total=item["seq_len_total"],
            batch_size_per_request=traffic.batch_size_per_request,
        )
        per_shape_kv.append({**item, "kv_cache_gb_per_request": kv_gb})

    return {
        **stats,
        "avg_kv_gb_per_request": avg_kv_gb_per_request,
        "p95_kv_gb_per_request": p95_kv_gb_per_request,
        "kv_distribution_details": per_shape_kv,
    }


def estimate_runtime_overhead_gb(runtime: RuntimeConfig, weight_with_overhead_gb: float) -> float:
    proportional_runtime = weight_with_overhead_gb * runtime.runtime_overhead_ratio
    return max(runtime.runtime_overhead_gb, proportional_runtime)


def estimate_memory_with_distribution(model: ModelConfig, traffic: TrafficConfig, runtime: RuntimeConfig) -> dict[str, Any]:
    weight_info = estimate_weight_memory_gb(model, runtime)
    kv_info = estimate_kv_distribution_gb(model, traffic, runtime)

    avg_total_kv_for_concurrency_gb = kv_info["avg_kv_gb_per_request"] * traffic.concurrency
    p95_total_kv_for_concurrency_gb = kv_info["p95_kv_gb_per_request"] * traffic.concurrency
    runtime_overhead_gb = estimate_runtime_overhead_gb(
        runtime,
        weight_info["weight_with_overhead_gb"],
    )

    avg_total_memory_gb = (
        weight_info["weight_with_overhead_gb"]
        + avg_total_kv_for_concurrency_gb
        + runtime_overhead_gb
    )
    p95_total_memory_gb = (
        weight_info["weight_with_overhead_gb"]
        + p95_total_kv_for_concurrency_gb
        + runtime_overhead_gb
    )

    return {
        **weight_info,
        **kv_info,
        "avg_total_kv_for_concurrency_gb": avg_total_kv_for_concurrency_gb,
        "p95_total_kv_for_concurrency_gb": p95_total_kv_for_concurrency_gb,
        "runtime_overhead_gb": runtime_overhead_gb,
        "avg_total_memory_gb": avg_total_memory_gb,
        "p95_total_memory_gb": p95_total_memory_gb,
    }


def estimate_memory_based_gpu_count(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
    use_p95: bool = True,
) -> dict[str, Any]:
    mem_info = estimate_memory_with_distribution(model, traffic, runtime)
    usable_vram_gb = gpu.vram_gb * runtime.usable_vram_ratio
    memory_for_sizing = (
        mem_info["p95_total_memory_gb"] if use_p95 else mem_info["avg_total_memory_gb"]
    )
    gpu_count_by_memory = ceil_div(memory_for_sizing, usable_vram_gb)

    return {
        **mem_info,
        "usable_vram_gb_per_gpu": usable_vram_gb,
        "memory_for_sizing_gb": memory_for_sizing,
        "gpu_count_by_memory": gpu_count_by_memory,
        "memory_sizing_basis": "p95" if use_p95 else "average",
    }


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


def estimate_throughput_per_gpu_by_spec(
    model: ModelConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, float | None]:
    activated_params = estimate_activated_params_billion(model) * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    bytes_per_token = activated_params * bytes_per_param
    flops_per_token = activated_params * 2.0

    decode_memory_limited_tps = None
    if gpu.memory_bandwidth_gb_per_sec and gpu.memory_bandwidth_gb_per_sec > 0 and bytes_per_token > 0:
        decode_memory_limited_tps = (gpu.memory_bandwidth_gb_per_sec * 1e9) / bytes_per_token

    prefill_memory_limited_tps = None
    if decode_memory_limited_tps is not None:
        prefill_memory_limited_tps = decode_memory_limited_tps * runtime.prefill_memory_reuse_factor

    peak_compute_tflops = get_peak_compute_tflops(gpu, runtime.precision)
    compute_limited_tps = None
    if peak_compute_tflops and peak_compute_tflops > 0 and flops_per_token > 0:
        compute_limited_tps = (
            peak_compute_tflops
            * 1e12
            * runtime.compute_efficiency
            / flops_per_token
        )

    decode_candidates = [
        value for value in (decode_memory_limited_tps, compute_limited_tps) if value is not None
    ]
    prefill_candidates = [
        value for value in (prefill_memory_limited_tps, compute_limited_tps) if value is not None
    ]
    decode_tps_spec = min(decode_candidates) * runtime.decode_efficiency if decode_candidates else None
    prefill_tps_spec = min(prefill_candidates) * runtime.prefill_efficiency if prefill_candidates else None
    decode_tpot_ms = (
        1000.0 / decode_memory_limited_tps
        if decode_memory_limited_tps and decode_memory_limited_tps > 0
        else None
    )

    return {
        "activated_params_billion": estimate_activated_params_billion(model),
        "bytes_per_token": bytes_per_token,
        "flops_per_token": flops_per_token,
        "decode_tps_per_gpu_memory_limited": decode_memory_limited_tps,
        "decode_tps_per_gpu_compute_limited": compute_limited_tps,
        "prefill_tps_per_gpu_memory_limited": prefill_memory_limited_tps,
        "prefill_tps_per_gpu_compute_limited": compute_limited_tps,
        "decode_tps_per_gpu_spec": decode_tps_spec,
        "prefill_tps_per_gpu_spec": prefill_tps_spec,
        "theoretical_tpot_ms_bandwidth_limited": decode_tpot_ms,
        "theoretical_tokens_per_sec_bandwidth_limited": decode_memory_limited_tps,
    }


def estimate_throughput_based_gpu_count(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    runtime: RuntimeConfig,
) -> dict[str, int | float | None]:
    throughput = estimate_throughput_per_gpu_by_spec(model, gpu, runtime)
    decode_gpu_count = None
    prefill_gpu_count = None

    if throughput["decode_tps_per_gpu_spec"] and throughput["decode_tps_per_gpu_spec"] > 0:
        decode_gpu_count = ceil_div(
            traffic.target_decode_tps_total,
            throughput["decode_tps_per_gpu_spec"],
        )

    if (
        traffic.target_prefill_tps_total is not None
        and throughput["prefill_tps_per_gpu_spec"]
        and throughput["prefill_tps_per_gpu_spec"] > 0
    ):
        prefill_gpu_count = ceil_div(
            traffic.target_prefill_tps_total,
            throughput["prefill_tps_per_gpu_spec"],
        )

    return {
        **throughput,
        "decode_gpu_count_by_throughput": decode_gpu_count,
        "prefill_gpu_count_by_throughput": prefill_gpu_count,
    }


def estimate_ha_gpu_count(base_gpu_count: int, ha: HAConfig) -> dict[str, Any]:
    if base_gpu_count <= 0:
        raise ValueError("base_gpu_count 必须大于 0")

    ha_mode = ha.ha_mode.lower()
    if ha_mode == "none":
        total_gpu_count = base_gpu_count
    elif ha_mode == "active_active":
        total_gpu_count = base_gpu_count * ha.replica_count
    elif ha_mode == "active_standby":
        total_gpu_count = base_gpu_count * 2
    elif ha_mode == "n_plus_1":
        total_gpu_count = base_gpu_count + 1
    else:
        raise ValueError(f"Unsupported ha_mode: {ha.ha_mode}")

    # 故障冗余仅在启用 HA 时生效
    if ha_mode != "none" and ha.failover_reserve_ratio > 0:
        total_gpu_count = ceil(total_gpu_count * (1.0 + ha.failover_reserve_ratio))

    return {
        "ha_mode": ha.ha_mode,
        "replica_count": ha.replica_count,
        "failover_reserve_ratio": ha.failover_reserve_ratio,
        "business_gpu_count": base_gpu_count,
        "total_gpu_count_after_ha": total_gpu_count,
        "ha_extra_gpu_count": total_gpu_count - base_gpu_count,
    }


def evaluate_single_model_with_ha(
    model: ModelConfig,
    traffic: TrafficConfig,
    gpu: GPUConfig,
    ha: HAConfig,
    runtime: RuntimeConfig,
    use_p95_for_memory_sizing: bool = True,
) -> dict[str, Any]:
    mem_info = estimate_memory_based_gpu_count(
        model=model,
        traffic=traffic,
        gpu=gpu,
        runtime=runtime,
        use_p95=use_p95_for_memory_sizing,
    )
    throughput_info = estimate_throughput_based_gpu_count(model, traffic, gpu, runtime)

    candidate_counts = [mem_info["gpu_count_by_memory"]]
    decode_gpu_count = throughput_info["decode_gpu_count_by_throughput"]
    prefill_gpu_count = throughput_info["prefill_gpu_count_by_throughput"]

    if decode_gpu_count is not None:
        candidate_counts.append(decode_gpu_count)
    if prefill_gpu_count is not None:
        candidate_counts.append(prefill_gpu_count)

    business_gpu_count = max(candidate_counts)
    ha_info = estimate_ha_gpu_count(base_gpu_count=business_gpu_count, ha=ha)

    estimated_total_cost = None
    if gpu.unit_price is not None:
        estimated_total_cost = ha_info["total_gpu_count_after_ha"] * gpu.unit_price

    return {
        "model_config": model,
        "gpu_config": gpu,
        "traffic_config": traffic,
        "ha_config": ha,
        "runtime_config": runtime, # Added this line
        "model_name": model.model_name,
        "gpu_name": gpu.gpu_name,
        "precision": runtime.precision,
        "kv_cache_dtype": runtime.kv_cache_dtype,
        "arch_family": model.arch_family,
        "attention_type": model.attention_type,
        "activated_params_billion": round(throughput_info["activated_params_billion"], 2),
        "avg_input_tokens": round(mem_info["avg_input_tokens"], 2),
        "avg_output_tokens": round(mem_info["avg_output_tokens"], 2),
        "avg_total_tokens": round(mem_info["avg_total_tokens"], 2),
        "p95_total_tokens": mem_info["p95_total_tokens"],
        "raw_weight_gb": round(mem_info["raw_weight_gb"], 2),
        "weight_with_overhead_gb": round(mem_info["weight_with_overhead_gb"], 2),
        "runtime_overhead_gb": round(mem_info["runtime_overhead_gb"], 2),
        "avg_kv_gb_per_request": round(mem_info["avg_kv_gb_per_request"], 2),
        "p95_kv_gb_per_request": round(mem_info["p95_kv_gb_per_request"], 2),
        "avg_total_memory_gb": round(mem_info["avg_total_memory_gb"], 2),
        "p95_total_memory_gb": round(mem_info["p95_total_memory_gb"], 2),
        "usable_vram_gb_per_gpu": round(mem_info["usable_vram_gb_per_gpu"], 2),
        "memory_for_sizing_gb": round(mem_info["memory_for_sizing_gb"], 2),
        "memory_sizing_basis": mem_info["memory_sizing_basis"],
        "gpu_count_by_memory": mem_info["gpu_count_by_memory"],
        "decode_tps_per_gpu_spec": round_optional(throughput_info["decode_tps_per_gpu_spec"], 2),
        "prefill_tps_per_gpu_spec": round_optional(throughput_info["prefill_tps_per_gpu_spec"], 2),
        "decode_tps_per_gpu_memory_limited": round_optional(
            throughput_info["decode_tps_per_gpu_memory_limited"],
            2,
        ),
        "decode_tps_per_gpu_compute_limited": round_optional(
            throughput_info["decode_tps_per_gpu_compute_limited"],
            2,
        ),
        "prefill_tps_per_gpu_memory_limited": round_optional(
            throughput_info["prefill_tps_per_gpu_memory_limited"],
            2,
        ),
        "prefill_tps_per_gpu_compute_limited": round_optional(
            throughput_info["prefill_tps_per_gpu_compute_limited"],
            2,
        ),
        "decode_gpu_count_by_throughput": decode_gpu_count,
        "prefill_gpu_count_by_throughput": prefill_gpu_count,
        "theoretical_tpot_ms_bandwidth_limited": round_optional(
            throughput_info["theoretical_tpot_ms_bandwidth_limited"],
            3,
        ),
        "theoretical_tokens_per_sec_bandwidth_limited": round_optional(
            throughput_info["theoretical_tokens_per_sec_bandwidth_limited"],
            2,
        ),
        "business_gpu_count": ha_info["business_gpu_count"],
        "ha_mode": ha_info["ha_mode"],
        "replica_count": ha_info["replica_count"],
        "ha_extra_gpu_count": ha_info["ha_extra_gpu_count"],
        "total_gpu_count_after_ha": ha_info["total_gpu_count_after_ha"],
        "unit_price": gpu.unit_price,
        "estimated_total_cost": estimated_total_cost,
        "request_shape_details": mem_info["request_shape_details"],
        "kv_distribution_details": mem_info["kv_distribution_details"],
        "g_memory": mem_info["gpu_count_by_memory"],
        "g_decode": decode_gpu_count,
        "g_prefill": prefill_gpu_count,
        "g_biz": ha_info["business_gpu_count"],
        "g_final": ha_info["total_gpu_count_after_ha"],
    }


def format_result_text(result: dict[str, Any]) -> str:
    lines = [
        "=" * 120,
        f"模型: {result['model_name']}",
        f"GPU:  {result['gpu_name']}",
        "=" * 120,
        "",
        "[长度分布]",
        f"- 平均输入长度: {result['avg_input_tokens']} tokens",
        f"- 平均输出长度: {result['avg_output_tokens']} tokens",
        f"- 平均总长度:   {result['avg_total_tokens']} tokens",
        f"- P95 总长度:   {result['p95_total_tokens']} tokens",
        "",
        "[显存估算]",
        f"- 权重显存(含开销): {result['weight_with_overhead_gb']} GB",
        f"- 运行时开销: {result['runtime_overhead_gb']} GB",
        f"- 平均单请求 KV Cache: {result['avg_kv_gb_per_request']} GB",
        f"- P95 单请求 KV Cache: {result['p95_kv_gb_per_request']} GB",
        f"- 平均总显存: {result['avg_total_memory_gb']} GB",
        f"- P95 总显存: {result['p95_total_memory_gb']} GB",
        (
            "- 显存 sizing 基准: "
            f"{result['memory_sizing_basis']} -> {result['memory_for_sizing_gb']} GB"
        ),
        f"- 单卡安全可用显存: {result['usable_vram_gb_per_gpu']} GB",
        f"- G_memory: {result['g_memory']}",
        "",
        "[吞吐估算 | 规格推导]",
        f"- 单卡 decode 规格吞吐: {result['decode_tps_per_gpu_spec']} tok/s",
        f"- 单卡 prefill 规格吞吐: {result['prefill_tps_per_gpu_spec']} tok/s",
        f"- G_decode: {result['g_decode']}",
        f"- G_prefill: {result['g_prefill']}",
        (
            "- 理论带宽 TPOT: "
            f"{result['theoretical_tpot_ms_bandwidth_limited']} ms/token"
        ),
        (
            "- 理论带宽吞吐上限: "
            f"{result['theoretical_tokens_per_sec_bandwidth_limited']} tok/s"
        ),
        "",
        "[高可用结果]",
        f"- 高可用模式: {result['ha_mode']}",
        f"- G_biz: {result['g_biz']}",
        f"- 高可用附加卡数: {result['ha_extra_gpu_count']}",
        f"- G_final: {result['g_final']}",
    ]

    if result["estimated_total_cost"] is not None:
        lines.append(f"- 估算总成本: {result['estimated_total_cost']:.0f}")

    lines.extend(["", "[请求分布明细]"])
    for item in result["request_shape_details"]:
        lines.append(
            "  * "
            f"{item['name']}: 占比={item['ratio']:.0%}, 输入={item['avg_input_tokens']}, "
            f"输出={item['avg_output_tokens']}, 总长={item['seq_len_total']}"
        )

    return "\n".join(lines)
