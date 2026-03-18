from __future__ import annotations

from typing import Any

from gpu_sizing_core.models import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig
from presets import (
    DEFAULT_BUSINESS_PROFILE,
    get_default_gpu_key,
    get_default_model_key,
    get_gpu_preset,
    get_model_preset,
)


def default_component_values() -> tuple[Any, ...]:
    profile = DEFAULT_BUSINESS_PROFILE
    return (
        get_default_model_key(),
        "bf16",
        get_default_gpu_key(),
        profile.lambda_avg_qps,
        profile.lambda_peak_qps,
        profile.avg_input_tokens,
        profile.avg_output_tokens,
        profile.p95_input_tokens,
        profile.p95_output_tokens,
        profile.ttft_avg_sec,
        profile.ttft_p95_sec,
        profile.e2e_avg_sec,
        profile.e2e_p95_sec,
        int(profile.concurrency_safety_factor * 100),
        15,
        8,
        90,
        65,
        60,
    )


def to_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"{field_name} 不能为空")
    int_value = int(float(value))
    if int_value != float(value):
        raise ValueError(f"{field_name} 必须是整数")
    return int_value


def to_float(value: Any, field_name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{field_name} 不能为空")
    return float(value)


def ensure_positive(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} 必须大于 0")


def ensure_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} 不能小于 0")


def build_configs(
    model_dropdown: Any,
    precision_override: Any,
    gpu_preset_key: Any,
    lambda_avg_qps: Any,
    lambda_peak_qps: Any,
    avg_input_tokens: Any,
    avg_output_tokens: Any,
    p95_input_tokens: Any,
    p95_output_tokens: Any,
    ttft_avg_sec: Any,
    ttft_p95_sec: Any,
    e2e_avg_sec: Any,
    e2e_p95_sec: Any,
    concurrency_safety_factor_pct: Any,
    weight_overhead_ratio: Any,
    runtime_overhead_ratio: Any,
    usable_vram_ratio: Any,
    bandwidth_efficiency: Any,
    compute_efficiency: Any,
) -> tuple[ModelConfig, TrafficConfig, GPUConfig, RuntimeConfig]:
    lambda_avg_qps_value = to_float(lambda_avg_qps, "平均 QPS")
    lambda_peak_qps_value = to_float(lambda_peak_qps, "峰值 QPS")
    avg_input_tokens_value = to_int(avg_input_tokens, "平均输入长度")
    avg_output_tokens_value = to_int(avg_output_tokens, "平均输出长度")
    p95_input_tokens_value = to_int(p95_input_tokens, "P95 输入长度")
    p95_output_tokens_value = to_int(p95_output_tokens, "P95 输出长度")
    ttft_avg_sec_value = to_float(ttft_avg_sec, "平均 TTFT")
    ttft_p95_sec_value = to_float(ttft_p95_sec, "P95 TTFT")
    e2e_avg_sec_value = to_float(e2e_avg_sec, "平均 E2E")
    e2e_p95_sec_value = to_float(e2e_p95_sec, "P95 E2E")
    concurrency_safety_factor_value = (
        to_float(concurrency_safety_factor_pct, "峰值在途安全系数") / 100.0
    )

    ensure_positive(lambda_avg_qps_value, "平均 QPS")
    ensure_positive(lambda_peak_qps_value, "峰值 QPS")
    ensure_positive(avg_input_tokens_value, "平均输入长度")
    ensure_non_negative(avg_output_tokens_value, "平均输出长度")
    ensure_positive(p95_input_tokens_value, "P95 输入长度")
    ensure_non_negative(p95_output_tokens_value, "P95 输出长度")
    ensure_positive(ttft_avg_sec_value, "平均 TTFT")
    ensure_positive(ttft_p95_sec_value, "P95 TTFT")
    ensure_positive(e2e_avg_sec_value, "平均 E2E")
    ensure_positive(e2e_p95_sec_value, "P95 E2E")
    ensure_positive(concurrency_safety_factor_value, "峰值在途安全系数")

    weight_overhead_ratio_value = to_float(weight_overhead_ratio, "权重附加系数") / 100.0
    runtime_overhead_ratio_value = to_float(runtime_overhead_ratio, "运行时固定显存系数") / 100.0
    usable_vram_ratio_value = to_float(usable_vram_ratio, "可用显存比例") / 100.0
    bandwidth_efficiency_value = to_float(bandwidth_efficiency, "带宽利用率") / 100.0
    compute_efficiency_value = to_float(compute_efficiency, "算力利用率") / 100.0

    ensure_non_negative(weight_overhead_ratio_value, "权重附加系数")
    ensure_non_negative(runtime_overhead_ratio_value, "运行时固定显存系数")
    if usable_vram_ratio_value <= 0 or usable_vram_ratio_value > 1:
        raise ValueError("可用显存比例必须在 0 到 1 之间")
    if bandwidth_efficiency_value <= 0 or bandwidth_efficiency_value > 1:
        raise ValueError("带宽利用率必须在 0 到 1 之间")
    if compute_efficiency_value <= 0 or compute_efficiency_value > 1:
        raise ValueError("算力利用率必须在 0 到 1 之间")

    base_model = get_model_preset(str(model_dropdown)).config
    base_gpu = get_gpu_preset(str(gpu_preset_key)).config

    target_prec = str(precision_override).lower()
    target_kv = "fp8" if target_prec == "fp8" else "fp16"

    traffic = TrafficConfig(
        lambda_avg_qps=lambda_avg_qps_value,
        lambda_peak_qps=lambda_peak_qps_value,
        avg_input_tokens=avg_input_tokens_value,
        avg_output_tokens=avg_output_tokens_value,
        p95_input_tokens=p95_input_tokens_value,
        p95_output_tokens=p95_output_tokens_value,
        ttft_avg_target_sec=ttft_avg_sec_value,
        ttft_p95_target_sec=ttft_p95_sec_value,
        e2e_avg_target_sec=e2e_avg_sec_value,
        e2e_p95_target_sec=e2e_p95_sec_value,
        concurrency_safety_factor=concurrency_safety_factor_value,
    )
    runtime = RuntimeConfig(
        precision=target_prec,
        kv_cache_dtype=target_kv,
        weight_overhead_ratio=weight_overhead_ratio_value,
        runtime_overhead_ratio=runtime_overhead_ratio_value,
        usable_vram_ratio=usable_vram_ratio_value,
        bandwidth_efficiency=bandwidth_efficiency_value,
        compute_efficiency=compute_efficiency_value,
    )
    return base_model, traffic, base_gpu, runtime


def build_default_configs() -> tuple[ModelConfig, TrafficConfig, GPUConfig, RuntimeConfig]:
    return build_configs(*default_component_values())
