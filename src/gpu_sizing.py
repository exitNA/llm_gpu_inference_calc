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

SECONDS_PER_DAY = 24 * 60 * 60


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


def multiply_optional(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def divide_optional(value: float, divisor: float | None) -> float | None:
    if divisor is None or divisor <= 0:
        return None
    return value / divisor


def format_calc_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    rounded = round(float(value), digits)
    if float(rounded).is_integer():
        return f"{int(rounded):,}"
    return f"{rounded:,.{digits}f}".rstrip("0").rstrip(".")


def estimate_average_conversation_duration_sec(
    avg_input_tokens: float,
    avg_output_tokens: float,
    concurrency: int,
    cluster_prefill_tps_capacity: float | None,
    cluster_decode_tps_capacity: float | None,
) -> dict[str, float | None]:
    prefill_tps_per_request = divide_optional(cluster_prefill_tps_capacity, concurrency)
    decode_tps_per_request = divide_optional(cluster_decode_tps_capacity, concurrency)
    avg_prefill_duration_sec = divide_optional(avg_input_tokens, prefill_tps_per_request)
    avg_decode_duration_sec = divide_optional(avg_output_tokens, decode_tps_per_request)

    if avg_prefill_duration_sec is None or avg_decode_duration_sec is None:
        avg_conversation_duration_sec = None
    else:
        avg_conversation_duration_sec = avg_prefill_duration_sec + avg_decode_duration_sec

    return {
        "prefill_tps_per_request_at_concurrency": prefill_tps_per_request,
        "decode_tps_per_request_at_concurrency": decode_tps_per_request,
        "avg_prefill_duration_sec": avg_prefill_duration_sec,
        "avg_decode_duration_sec": avg_decode_duration_sec,
        "avg_conversation_duration_sec": avg_conversation_duration_sec,
    }


def _build_weighted_expression(
    request_shapes: list[RequestShape],
    token_attr: str,
) -> str:
    parts = []
    for item in request_shapes:
        token_value = getattr(item, token_attr)
        parts.append(f"{format_calc_number(item.ratio, 2)} x {token_value}")
    return " + ".join(parts)


def format_ratio_percent(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def build_calc_step(
    label: str,
    formula: str,
    substitution: str,
    result: str,
    note: str | None = None,
) -> dict[str, str]:
    step = {
        "label": label,
        "formula": formula,
        "substitution": substitution,
        "result": result,
    }
    if note:
        step["note"] = note
    return step


def _describe_p95_selection(request_shapes: list[RequestShape]) -> str:
    sorted_shapes = sorted(
        request_shapes,
        key=lambda shape: shape.avg_input_tokens + shape.avg_output_tokens,
    )
    cumulative = 0.0
    steps = []
    selected_name = "-"
    selected_total = 0
    for item in sorted_shapes:
        cumulative += item.ratio
        total_tokens = item.avg_input_tokens + item.avg_output_tokens
        steps.append(f"{item.name}({format_ratio_percent(cumulative)})")
        if cumulative >= 0.95 and selected_name == "-":
            selected_name = item.name
            selected_total = total_tokens
    return (
        "按总长度从小到大累计占比: "
        + " -> ".join(steps)
        + f"；首个达到 95% 的请求类型是“{selected_name}”，"
        + f"因此 P95 总长度取 {selected_total} tokens"
    )


def _describe_cache_formula(
    model: ModelConfig,
    runtime: RuntimeConfig,
) -> str:
    kv_bytes = precision_to_bytes(runtime.kv_cache_dtype)
    head_dim = _resolve_head_dim(model)
    attention_type = model.attention_type.lower()
    if model.cache_bytes_per_token_per_layer is not None:
        return (
            "每 token 每层缓存字节数使用显式输入值 = "
            f"{format_calc_number(model.cache_bytes_per_token_per_layer)} bytes"
        )
    if attention_type == "mha":
        return (
            "每 token 每层缓存字节数 = 2 × hidden_size × kv_bytes = "
            f"2 × {model.hidden_size} × {format_calc_number(kv_bytes)} = "
            f"{format_calc_number(estimate_cache_bytes_per_token_per_layer(model, runtime))} bytes"
        )
    if attention_type == "gqa":
        return (
            "每 token 每层缓存字节数 = 2 × num_kv_heads × head_dim × kv_bytes = "
            f"2 × {model.num_kv_heads} × {head_dim} × {format_calc_number(kv_bytes)} = "
            f"{format_calc_number(estimate_cache_bytes_per_token_per_layer(model, runtime))} bytes"
        )
    if attention_type == "mqa":
        return (
            "每 token 每层缓存字节数 = 2 × head_dim × kv_bytes = "
            f"2 × {head_dim} × {format_calc_number(kv_bytes)} = "
            f"{format_calc_number(estimate_cache_bytes_per_token_per_layer(model, runtime))} bytes"
        )
    if attention_type == "mla":
        return (
            "每 token 每层缓存字节数 = latent_cache_dim × kv_bytes + aux = "
            f"{model.latent_cache_dim} × {format_calc_number(kv_bytes)} + "
            f"{format_calc_number(model.cache_aux_bytes_per_token_per_layer)} = "
            f"{format_calc_number(estimate_cache_bytes_per_token_per_layer(model, runtime))} bytes"
        )
    return (
        "每 token 每层缓存字节数 = "
        f"{format_calc_number(estimate_cache_bytes_per_token_per_layer(model, runtime))} bytes"
    )


def build_calculation_process_sections(result: dict[str, Any]) -> list[dict[str, Any]]:
    model: ModelConfig = result["model_config"]
    gpu: GPUConfig = result["gpu_config"]
    traffic: TrafficConfig = result["traffic_config"]
    runtime: RuntimeConfig = result["runtime_config"]
    request_shapes = traffic.request_shapes or []

    avg_input_expr = " + ".join(
        f"{format_ratio_percent(item.ratio)} × {item.avg_input_tokens}"
        for item in request_shapes
    )
    avg_output_expr = " + ".join(
        f"{format_ratio_percent(item.ratio)} × {item.avg_output_tokens}"
        for item in request_shapes
    )
    bytes_per_param = precision_to_bytes(runtime.precision)
    cache_bytes_per_token_per_layer = estimate_cache_bytes_per_token_per_layer(model, runtime)
    peak_compute_tflops = get_peak_compute_tflops(gpu, runtime.precision)
    activated_params_billion = estimate_activated_params_billion(model)
    activated_params = activated_params_billion * 1e9
    weight_bytes_per_token = activated_params * bytes_per_param
    flops_per_token = activated_params * 2.0
    kv_cache_bytes_per_step = (
        model.num_layers
        * result["avg_total_tokens"]
        * cache_bytes_per_token_per_layer
    )

    memory_steps = [
        build_calc_step(
            label="原始权重显存",
            formula="原始权重显存 = 参数量 × 每参数字节数",
            substitution=(
                f"{format_calc_number(model.num_params_billion)} B × "
                f"{format_calc_number(bytes_per_param)} byte"
            ),
            result=f"{format_calc_number(result['raw_weight_gb'])} GB",
        ),
        build_calc_step(
            label="含冗余权重显存",
            formula="权重显存 = 原始权重显存 × (1 + 权重冗余比例)",
            substitution=(
                f"{format_calc_number(result['raw_weight_gb'])} × "
                f"(1 + {format_calc_number(runtime.weight_overhead_ratio, 2)})"
            ),
            result=f"{format_calc_number(result['weight_with_overhead_gb'])} GB",
        ),
        build_calc_step(
            label="每 token 每层缓存字节数",
            formula="按注意力结构推导 KV Cache 字节数",
            substitution=_describe_cache_formula(model, runtime),
            result=f"{format_calc_number(cache_bytes_per_token_per_layer)} bytes",
        ),
        build_calc_step(
            label="平均单请求 KV Cache",
            formula="KV Cache = 层数 × 平均总长度 × batch × 每 token 每层缓存字节数 ÷ 1e9",
            substitution=(
                f"{model.num_layers} × {format_calc_number(result['avg_total_tokens'])} × "
                f"{traffic.batch_size_per_request} × {format_calc_number(cache_bytes_per_token_per_layer)} ÷ 1e9"
            ),
            result=f"{format_calc_number(result['avg_kv_gb_per_request'])} GB",
        ),
        build_calc_step(
            label="P95 单请求 KV Cache",
            formula="P95 KV Cache = 层数 × P95 总长度 × batch × 每 token 每层缓存字节数 ÷ 1e9",
            substitution=(
                f"{model.num_layers} × {result['p95_total_tokens']} × "
                f"{traffic.batch_size_per_request} × {format_calc_number(cache_bytes_per_token_per_layer)} ÷ 1e9"
            ),
            result=f"{format_calc_number(result['p95_kv_gb_per_request'])} GB",
        ),
        build_calc_step(
            label="运行时开销",
            formula="运行时开销 = max(保底值, 权重显存 × 运行时比例)",
            substitution=(
                f"max({format_calc_number(runtime.runtime_overhead_gb)}, "
                f"{format_calc_number(result['weight_with_overhead_gb'])} × "
                f"{format_calc_number(runtime.runtime_overhead_ratio, 2)})"
            ),
            result=f"{format_calc_number(result['runtime_overhead_gb'])} GB",
        ),
        build_calc_step(
            label="平均总显存",
            formula="平均总显存 = 权重显存 + 平均单请求 KV Cache × 并发 + 运行时开销",
            substitution=(
                f"{format_calc_number(result['weight_with_overhead_gb'])} + "
                f"{format_calc_number(result['avg_kv_gb_per_request'])} × {traffic.concurrency} + "
                f"{format_calc_number(result['runtime_overhead_gb'])}"
            ),
            result=f"{format_calc_number(result['avg_total_memory_gb'])} GB",
        ),
        build_calc_step(
            label="P95 总显存",
            formula="P95 总显存 = 权重显存 + P95 单请求 KV Cache × 并发 + 运行时开销",
            substitution=(
                f"{format_calc_number(result['weight_with_overhead_gb'])} + "
                f"{format_calc_number(result['p95_kv_gb_per_request'])} × {traffic.concurrency} + "
                f"{format_calc_number(result['runtime_overhead_gb'])}"
            ),
            result=f"{format_calc_number(result['p95_total_memory_gb'])} GB",
        ),
        build_calc_step(
            label="显存 sizing 口径",
            formula="按用户选择的显存口径取值",
            substitution=f"当前选择：{result['memory_sizing_basis']}",
            result=f"{format_calc_number(result['memory_for_sizing_gb'])} GB",
        ),
        build_calc_step(
            label="单卡安全可用显存",
            formula="单卡安全可用显存 = 单卡总显存 × 安全水位线",
            substitution=(
                f"{format_calc_number(gpu.vram_gb)} × "
                f"{format_calc_number(runtime.usable_vram_ratio, 2)}"
            ),
            result=f"{format_calc_number(result['usable_vram_gb_per_gpu'])} GB",
        ),
        build_calc_step(
            label="显存约束卡数",
            formula="显存约束卡数 = ⌈ 显存需求 ÷ 单卡安全可用显存 ⌉",
            substitution=(
                f"⌈ {format_calc_number(result['memory_for_sizing_gb'])} ÷ "
                f"{format_calc_number(result['usable_vram_gb_per_gpu'])} ⌉"
            ),
            result=f"{result['gpu_count_by_memory']} 卡",
        ),
    ]

    throughput_steps = [
        build_calc_step(
            label="单 token 激活参数量",
            formula="激活参数量 = Dense 用总参数量，MoE 用 activated params",
            substitution=f"当前模型取 {format_calc_number(activated_params_billion)} B",
            result=f"{format_calc_number(activated_params_billion)} B",
        ),
        build_calc_step(
            label="每 token 读取权重字节数",
            formula="权重字节数 = 激活参数量 × 每参数字节数",
            substitution=(
                f"{format_calc_number(activated_params_billion)}e9 × "
                f"{format_calc_number(bytes_per_param)}"
            ),
            result=f"{format_calc_number(weight_bytes_per_token)} bytes",
        ),
        build_calc_step(
            label="每步 KV 访问字节数",
            formula="KV 字节数 = 层数 × 平均总长度 × 每 token 每层缓存字节数",
            substitution=(
                f"{model.num_layers} × {format_calc_number(result['avg_total_tokens'])} × "
                f"{format_calc_number(cache_bytes_per_token_per_layer)}"
            ),
            result=f"{format_calc_number(kv_cache_bytes_per_step)} bytes",
        ),
        build_calc_step(
            label="每 token 总访问字节数",
            formula="总字节数 = 权重字节数 + KV 字节数",
            substitution=(
                f"{format_calc_number(weight_bytes_per_token)} + "
                f"{format_calc_number(kv_cache_bytes_per_step)}"
            ),
            result=f"{format_calc_number(weight_bytes_per_token + kv_cache_bytes_per_step)} bytes",
        ),
        build_calc_step(
            label="单卡 Decode 吞吐上限（带宽约束）",
            formula="Decode 吞吐 = 带宽 × 1e9 ÷ 每 token 总访问字节数",
            substitution=(
                f"{format_calc_number(gpu.memory_bandwidth_gb_per_sec)} × 1e9 ÷ "
                f"{format_calc_number(weight_bytes_per_token + kv_cache_bytes_per_step)}"
            ),
            result=f"{format_calc_number(result['decode_tps_per_gpu_memory_limited'])} tok/s",
        ),
        build_calc_step(
            label="单卡 Prefill 吞吐上限（带宽约束）",
            formula="Prefill 吞吐 = Decode 带宽上限 × Prefill 复用系数",
            substitution=(
                f"{format_calc_number(result['decode_tps_per_gpu_memory_limited'])} × "
                f"{format_calc_number(runtime.prefill_memory_reuse_factor)}"
            ),
            result=f"{format_calc_number(result['prefill_tps_per_gpu_memory_limited'])} tok/s",
        ),
        build_calc_step(
            label="每 token 计算量",
            formula="FLOPs = 激活参数量 × 2",
            substitution=f"{format_calc_number(activated_params)} × 2",
            result=f"{format_calc_number(flops_per_token)} FLOPs",
        ),
        build_calc_step(
            label="单卡 Decode 吞吐上限（算力约束）",
            formula="Decode 吞吐 = 峰值算力 × 1e12 × 算力效率 ÷ 每 token 计算量",
            substitution=(
                f"{format_calc_number(peak_compute_tflops)} × 1e12 × "
                f"{format_calc_number(runtime.compute_efficiency, 2)} ÷ "
                f"{format_calc_number(flops_per_token)}"
            ),
            result=f"{format_calc_number(result['decode_tps_per_gpu_compute_limited'])} tok/s",
        ),
        build_calc_step(
            label="单卡 Decode 规格吞吐",
            formula="Decode 规格吞吐 = min(带宽上限, 算力上限) × Decode 效率",
            substitution=(
                f"min({format_calc_number(result['decode_tps_per_gpu_memory_limited'])}, "
                f"{format_calc_number(result['decode_tps_per_gpu_compute_limited'])}) × "
                f"{format_calc_number(runtime.decode_efficiency, 2)}"
            ),
            result=f"{format_calc_number(result['decode_tps_per_gpu_spec'])} tok/s",
        ),
        build_calc_step(
            label="单卡 Prefill 规格吞吐",
            formula="Prefill 规格吞吐 = min(带宽上限, 算力上限) × Prefill 效率",
            substitution=(
                f"min({format_calc_number(result['prefill_tps_per_gpu_memory_limited'])}, "
                f"{format_calc_number(result['prefill_tps_per_gpu_compute_limited'])}) × "
                f"{format_calc_number(runtime.prefill_efficiency, 2)}"
            ),
            result=f"{format_calc_number(result['prefill_tps_per_gpu_spec'])} tok/s",
        ),
        build_calc_step(
            label="Decode 吞吐约束卡数",
            formula="Decode 卡数 = ⌈ 集群 Decode 目标 ÷ 单卡 Decode 规格吞吐 ⌉",
            substitution=(
                f"⌈ {format_calc_number(traffic.target_decode_tps_total)} ÷ "
                f"{format_calc_number(result['decode_tps_per_gpu_spec'])} ⌉"
            ),
            result=f"{format_calc_number(result['decode_gpu_count_by_throughput'])} 卡",
        ),
        build_calc_step(
            label="Prefill 吞吐约束卡数",
            formula="Prefill 卡数 = ⌈ 集群 Prefill 目标 ÷ 单卡 Prefill 规格吞吐 ⌉",
            substitution=(
                f"⌈ {format_calc_number(traffic.target_prefill_tps_total)} ÷ "
                f"{format_calc_number(result['prefill_tps_per_gpu_spec'])} ⌉"
            ),
            result=f"{format_calc_number(result['prefill_gpu_count_by_throughput'])} 卡",
        ),
    ]

    if result["ha_mode"] == "none":
        ha_substitution = f"{result['business_gpu_count']}"
    elif result["ha_mode"] == "active_active":
        ha_substitution = (
            f"{result['business_gpu_count']} × {result['replica_count']}"
        )
    elif result["ha_mode"] == "active_standby":
        ha_substitution = f"{result['business_gpu_count']} × 2"
    else:
        ha_substitution = f"{result['business_gpu_count']} + 1"

    derived_steps = [
        build_calc_step(
            label="业务基线卡数",
            formula="业务基线卡数 = max(显存约束卡数, Decode 吞吐约束卡数, Prefill 吞吐约束卡数)",
            substitution=(
                f"max({result['gpu_count_by_memory']}, "
                f"{format_calc_number(result['decode_gpu_count_by_throughput'])}, "
                f"{format_calc_number(result['prefill_gpu_count_by_throughput'])})"
            ),
            result=f"{result['business_gpu_count']} 卡",
        ),
        build_calc_step(
            label="HA 后总卡数",
            formula="总卡数 = 按 HA 模式在业务基线卡数上加冗余",
            substitution=ha_substitution,
            result=f"{result['total_gpu_count_after_ha']} 卡",
            note=f"当前 HA 模式：{result['ha_mode']}",
        ),
        build_calc_step(
            label="集群 Decode 理论上限",
            formula="集群 Decode 上限 = 业务基线卡数 × 单卡 Decode 规格吞吐",
            substitution=(
                f"{result['business_gpu_count']} × "
                f"{format_calc_number(result['decode_tps_per_gpu_spec'])}"
            ),
            result=f"{format_calc_number(result['cluster_decode_tps_capacity'])} tok/s",
        ),
        build_calc_step(
            label="集群 Prefill 理论上限",
            formula="集群 Prefill 上限 = 业务基线卡数 × 单卡 Prefill 规格吞吐",
            substitution=(
                f"{result['business_gpu_count']} × "
                f"{format_calc_number(result['prefill_tps_per_gpu_spec'])}"
            ),
            result=f"{format_calc_number(result['cluster_prefill_tps_capacity'])} tok/s",
        ),
        build_calc_step(
            label="每日 Decode token 上限",
            formula="每日 Decode 上限 = 集群 Decode 上限 × 86,400",
            substitution=(
                f"{format_calc_number(result['cluster_decode_tps_capacity'])} × 86,400"
            ),
            result=f"{format_calc_number(result['daily_decode_token_capacity'])} tokens/day",
        ),
        build_calc_step(
            label="每日 Prefill token 上限",
            formula="每日 Prefill 上限 = 集群 Prefill 上限 × 86,400",
            substitution=(
                f"{format_calc_number(result['cluster_prefill_tps_capacity'])} × 86,400"
            ),
            result=f"{format_calc_number(result['daily_prefill_token_capacity'])} tokens/day",
        ),
        build_calc_step(
            label="平均 Prefill 耗时",
            formula="平均 Prefill 耗时 = 平均输入长度 ÷ (集群 Prefill 上限 ÷ 并发)",
            substitution=(
                f"{format_calc_number(result['avg_input_tokens'])} ÷ "
                f"({format_calc_number(result['cluster_prefill_tps_capacity'])} ÷ {traffic.concurrency})"
            ),
            result=f"{format_calc_number(result['avg_prefill_duration_sec'])} s",
        ),
        build_calc_step(
            label="平均 Decode 耗时",
            formula="平均 Decode 耗时 = 平均输出长度 ÷ (集群 Decode 上限 ÷ 并发)",
            substitution=(
                f"{format_calc_number(result['avg_output_tokens'])} ÷ "
                f"({format_calc_number(result['cluster_decode_tps_capacity'])} ÷ {traffic.concurrency})"
            ),
            result=f"{format_calc_number(result['avg_decode_duration_sec'])} s",
        ),
        build_calc_step(
            label="平均一次对话耗时",
            formula="平均对话耗时 = 平均 Prefill 耗时 + 平均 Decode 耗时",
            substitution=(
                f"{format_calc_number(result['avg_prefill_duration_sec'])} + "
                f"{format_calc_number(result['avg_decode_duration_sec'])}"
            ),
            result=f"{format_calc_number(result['avg_conversation_duration_sec'])} s",
        ),
    ]

    return [
        {
            "title": "请求画像统计",
            "summary": "先把业务请求画像折算成容量规划要用的平均长度与 P95 长度。",
            "steps": [
                build_calc_step(
                    label="平均输入长度",
                    formula="平均输入长度 = Σ(占比 × 输入长度)",
                    substitution=avg_input_expr,
                    result=f"{format_calc_number(result['avg_input_tokens'])} tokens",
                ),
                build_calc_step(
                    label="平均输出长度",
                    formula="平均输出长度 = Σ(占比 × 输出长度)",
                    substitution=avg_output_expr,
                    result=f"{format_calc_number(result['avg_output_tokens'])} tokens",
                ),
                build_calc_step(
                    label="平均总长度",
                    formula="平均总长度 = 平均输入长度 + 平均输出长度",
                    substitution=(
                        f"{format_calc_number(result['avg_input_tokens'])} + "
                        f"{format_calc_number(result['avg_output_tokens'])}"
                    ),
                    result=f"{format_calc_number(result['avg_total_tokens'])} tokens",
                ),
                build_calc_step(
                    label="P95 总长度",
                    formula="P95 总长度 = 按总长度排序后，累计占比首次达到 95% 的请求长度",
                    substitution=_describe_p95_selection(request_shapes),
                    result=f"{result['p95_total_tokens']} tokens",
                ),
            ],
        },
        {
            "title": "显存估算",
            "summary": "先算模型权重、KV Cache 和运行时开销，再得到显存约束下至少需要多少卡。",
            "steps": memory_steps,
        },
        {
            "title": "吞吐估算",
            "summary": "分别从带宽上限和算力上限估算单卡能力，再折算成 Decode / Prefill 所需卡数。",
            "steps": throughput_steps,
        },
        {
            "title": "最终结果与派生指标",
            "summary": "将显存与吞吐三类卡数取最大值作为业务基线，再推导 HA、日供给和平均耗时。",
            "steps": derived_steps,
        },
    ]


def format_calculation_process_text(sections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(f"[{section['title']}]")
        if section.get("summary"):
            lines.append(f"- 说明: {section['summary']}")
        for step in section["steps"]:
            lines.append(f"- {step['label']}")
            lines.append(f"  公式: {step['formula']}")
            lines.append(f"  代入: {step['substitution']}")
            lines.append(f"  结果: {step['result']}")
            if step.get("note"):
                lines.append(f"  备注: {step['note']}")
        lines.append("")
    return "\n".join(lines).strip()


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
    avg_total_tokens: float,
) -> dict[str, float | None]:
    activated_params = estimate_activated_params_billion(model) * 1e9
    bytes_per_param = precision_to_bytes(runtime.precision)
    weight_bytes_per_token = activated_params * bytes_per_param
    flops_per_token = activated_params * 2.0
    
    # Calculate KV cache bytes per step (per token)
    cache_bytes_per_token_per_layer = estimate_cache_bytes_per_token_per_layer(model, runtime)
    kv_cache_bytes_per_step = model.num_layers * avg_total_tokens * cache_bytes_per_token_per_layer
    
    total_bytes_per_token = weight_bytes_per_token + kv_cache_bytes_per_step

    decode_memory_limited_tps = None
    if gpu.memory_bandwidth_gb_per_sec and gpu.memory_bandwidth_gb_per_sec > 0 and total_bytes_per_token > 0:
        decode_memory_limited_tps = (gpu.memory_bandwidth_gb_per_sec * 1e9) / total_bytes_per_token

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
        "total_bytes_per_token": total_bytes_per_token,
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
    avg_total_tokens: float,
) -> dict[str, int | float | None]:
    throughput = estimate_throughput_per_gpu_by_spec(model, gpu, runtime, avg_total_tokens)
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
    throughput_info = estimate_throughput_based_gpu_count(
        model=model, 
        traffic=traffic, 
        gpu=gpu, 
        runtime=runtime,
        avg_total_tokens=mem_info["avg_total_tokens"]
    )

    candidate_counts = [mem_info["gpu_count_by_memory"]]
    decode_gpu_count = throughput_info["decode_gpu_count_by_throughput"]
    prefill_gpu_count = throughput_info["prefill_gpu_count_by_throughput"]

    if decode_gpu_count is not None:
        candidate_counts.append(decode_gpu_count)
    if prefill_gpu_count is not None:
        candidate_counts.append(prefill_gpu_count)

    business_gpu_count = max(candidate_counts)
    ha_info = estimate_ha_gpu_count(base_gpu_count=business_gpu_count, ha=ha)
    cluster_decode_tps_capacity = multiply_optional(
        throughput_info["decode_tps_per_gpu_spec"],
        business_gpu_count,
    )
    cluster_prefill_tps_capacity = multiply_optional(
        throughput_info["prefill_tps_per_gpu_spec"],
        business_gpu_count,
    )
    daily_decode_token_capacity = multiply_optional(
        cluster_decode_tps_capacity,
        SECONDS_PER_DAY,
    )
    daily_prefill_token_capacity = multiply_optional(
        cluster_prefill_tps_capacity,
        SECONDS_PER_DAY,
    )
    conversation_duration_info = estimate_average_conversation_duration_sec(
        avg_input_tokens=mem_info["avg_input_tokens"],
        avg_output_tokens=mem_info["avg_output_tokens"],
        concurrency=traffic.concurrency,
        cluster_prefill_tps_capacity=cluster_prefill_tps_capacity,
        cluster_decode_tps_capacity=cluster_decode_tps_capacity,
    )

    estimated_total_cost = None
    if gpu.unit_price is not None:
        estimated_total_cost = ha_info["total_gpu_count_after_ha"] * gpu.unit_price

    result = {
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
        "cluster_decode_tps_capacity": round_optional(cluster_decode_tps_capacity, 1),
        "cluster_prefill_tps_capacity": round_optional(cluster_prefill_tps_capacity, 1),
        "daily_decode_token_capacity": round_optional(daily_decode_token_capacity, 1),
        "daily_prefill_token_capacity": round_optional(daily_prefill_token_capacity, 1),
        "avg_prefill_duration_sec": round_optional(
            conversation_duration_info["avg_prefill_duration_sec"],
            1,
        ),
        "avg_decode_duration_sec": round_optional(
            conversation_duration_info["avg_decode_duration_sec"],
            1,
        ),
        "avg_conversation_duration_sec": round_optional(
            conversation_duration_info["avg_conversation_duration_sec"],
            1,
        ),
        "token_capacity_basis": "business_gpu_count",
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
    calculation_process_sections = build_calculation_process_sections(result)
    result["calculation_process_sections"] = calculation_process_sections
    result["calculation_process_text"] = format_calculation_process_text(
        calculation_process_sections,
    )
    return result


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
        f"- 集群 decode 理论上限: {result['cluster_decode_tps_capacity']} tok/s",
        f"- 集群 prefill 理论上限: {result['cluster_prefill_tps_capacity']} tok/s",
        f"- 每日 decode token 上限: {result['daily_decode_token_capacity']} tokens/day",
        f"- 每日 prefill token 上限: {result['daily_prefill_token_capacity']} tokens/day",
        f"- 平均一次对话耗时: {result['avg_conversation_duration_sec']} s",
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
        "- token 日供给口径: 按 G_biz 计算，不将 HA 冗余重复计入可供给产能",
        "- 对话耗时口径: 平均输入 prefill 时间 + 平均输出 decode 时间，按 G_biz 与当前并发近似",
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

    if result.get("calculation_process_text"):
        lines.extend(["", "[计算过程]", result["calculation_process_text"]])

    return "\n".join(lines)
