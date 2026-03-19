from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpu_sizing_core.models import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig
from presets import (
    DEFAULT_BUSINESS_PROFILE,
    get_default_gpu_key,
    get_default_model_key,
    get_gpu_preset,
    get_model_preset,
)


@dataclass(frozen=True)
class DefaultComponentValues:
    model_dropdown: str
    precision_override: str
    gpu_preset_key: str
    lambda_peak_qps: float
    p95_input_tokens: int
    p95_output_tokens: int
    ttft_p95_sec: float
    e2e_p95_sec: float
    concurrency_safety_factor_pct: int
    weight_overhead_ratio: int
    runtime_overhead_ratio: int
    usable_vram_ratio: int
    bandwidth_efficiency: int
    compute_efficiency: int

    @classmethod
    def default(cls) -> DefaultComponentValues:
        profile = DEFAULT_BUSINESS_PROFILE
        return cls(
            model_dropdown=get_default_model_key(),
            precision_override="fp8",
            gpu_preset_key=get_default_gpu_key(),
            lambda_peak_qps=profile.lambda_peak_qps,
            p95_input_tokens=profile.p95_input_tokens,
            p95_output_tokens=profile.p95_output_tokens,
            ttft_p95_sec=profile.ttft_p95_sec,
            e2e_p95_sec=profile.e2e_p95_sec,
            concurrency_safety_factor_pct=int(profile.concurrency_safety_factor * 100),
            weight_overhead_ratio=15,
            runtime_overhead_ratio=8,
            usable_vram_ratio=90,
            bandwidth_efficiency=65,
            compute_efficiency=60,
        )

    def gradio_values(self) -> tuple[Any, ...]:
        return (
            self.model_dropdown,
            self.precision_override,
            self.gpu_preset_key,
            self.lambda_peak_qps,
            self.p95_input_tokens,
            self.p95_output_tokens,
            self.ttft_p95_sec,
            self.e2e_p95_sec,
            self.concurrency_safety_factor_pct,
            self.weight_overhead_ratio,
            self.runtime_overhead_ratio,
            self.usable_vram_ratio,
            self.bandwidth_efficiency,
            self.compute_efficiency,
        )

    @classmethod
    def from_raw_inputs(cls, raw_inputs: tuple[Any, ...]) -> DefaultComponentValues:
        return cls(*raw_inputs)

    def build_configs(self) -> tuple[ModelConfig, TrafficConfig, GPUConfig, RuntimeConfig]:
        lambda_peak_qps_value = to_float(self.lambda_peak_qps, "QPS")
        p95_input_tokens_value = to_int(self.p95_input_tokens, "输入长度")
        p95_output_tokens_value = to_int(self.p95_output_tokens, "输出长度")
        ttft_p95_sec_value = to_float(self.ttft_p95_sec, "TTFT 目标")
        e2e_p95_sec_value = to_float(self.e2e_p95_sec, "E2E 目标")
        concurrency_safety_factor_value = (
            to_float(self.concurrency_safety_factor_pct, "在途安全系数") / 100.0
        )

        ensure_positive(lambda_peak_qps_value, "QPS")
        ensure_positive(p95_input_tokens_value, "输入长度")
        ensure_non_negative(p95_output_tokens_value, "输出长度")
        ensure_positive(ttft_p95_sec_value, "TTFT 目标")
        ensure_positive(e2e_p95_sec_value, "E2E 目标")
        ensure_positive(concurrency_safety_factor_value, "在途安全系数")

        weight_overhead_ratio_value = to_float(self.weight_overhead_ratio, "权重附加系数") / 100.0
        runtime_overhead_ratio_value = to_float(self.runtime_overhead_ratio, "运行时固定显存系数") / 100.0
        usable_vram_ratio_value = to_float(self.usable_vram_ratio, "可用显存比例") / 100.0
        bandwidth_efficiency_value = to_float(self.bandwidth_efficiency, "带宽利用率") / 100.0
        compute_efficiency_value = to_float(self.compute_efficiency, "算力利用率") / 100.0

        ensure_non_negative(weight_overhead_ratio_value, "权重附加系数")
        ensure_non_negative(runtime_overhead_ratio_value, "运行时固定显存系数")
        if usable_vram_ratio_value <= 0 or usable_vram_ratio_value > 1:
            raise ValueError("可用显存比例必须在 0 到 1 之间")
        if bandwidth_efficiency_value <= 0 or bandwidth_efficiency_value > 1:
            raise ValueError("带宽利用率必须在 0 到 1 之间")
        if compute_efficiency_value <= 0 or compute_efficiency_value > 1:
            raise ValueError("算力利用率必须在 0 到 1 之间")

        base_model = get_model_preset(str(self.model_dropdown)).config
        base_gpu = get_gpu_preset(str(self.gpu_preset_key)).config

        target_prec = str(self.precision_override).lower()
        target_kv = "fp8" if target_prec == "fp8" else "fp16"

        traffic = TrafficConfig(
            lambda_peak_qps=lambda_peak_qps_value,
            p95_input_tokens=p95_input_tokens_value,
            p95_output_tokens=p95_output_tokens_value,
            ttft_p95_target_sec=ttft_p95_sec_value,
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


def default_component_values() -> DefaultComponentValues:
    return DefaultComponentValues.default()


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
    values: DefaultComponentValues,
) -> tuple[ModelConfig, TrafficConfig, GPUConfig, RuntimeConfig]:
    return values.build_configs()


def build_default_configs() -> tuple[ModelConfig, TrafficConfig, GPUConfig, RuntimeConfig]:
    return DefaultComponentValues.default().build_configs()
