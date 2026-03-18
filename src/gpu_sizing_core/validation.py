from __future__ import annotations

from .models import RuntimeConfig, TrafficConfig


def validate_traffic_config(traffic: TrafficConfig) -> None:
    if traffic.lambda_avg_qps <= 0:
        raise ValueError("lambda_avg_qps 必须大于 0")
    if traffic.lambda_peak_qps < traffic.lambda_avg_qps:
        raise ValueError("lambda_peak_qps 必须大于等于 lambda_avg_qps")
    if traffic.avg_input_tokens <= 0 or traffic.avg_output_tokens < 0:
        raise ValueError("平均输入输出长度必须合法")
    if traffic.p95_input_tokens <= 0 or traffic.p95_output_tokens < 0:
        raise ValueError("P95 输入输出长度必须合法")
    if traffic.ttft_avg_target_sec <= 0:
        raise ValueError("ttft_avg_target_sec 必须大于 0")
    if traffic.ttft_p95_target_sec < traffic.ttft_avg_target_sec:
        raise ValueError("ttft_p95_target_sec 必须大于等于 ttft_avg_target_sec")
    if traffic.e2e_avg_target_sec <= traffic.ttft_avg_target_sec:
        raise ValueError("e2e_avg_target_sec 必须大于 ttft_avg_target_sec")
    if traffic.e2e_p95_target_sec <= traffic.ttft_p95_target_sec:
        raise ValueError("e2e_p95_target_sec 必须大于 ttft_p95_target_sec")
    if traffic.concurrency_safety_factor < 1.0:
        raise ValueError("concurrency_safety_factor 必须大于等于 1.0")


def validate_runtime_config(runtime: RuntimeConfig) -> None:
    if not (0 < runtime.usable_vram_ratio <= 1):
        raise ValueError("usable_vram_ratio 必须在 (0, 1] 内")
    if not (0 < runtime.bandwidth_efficiency <= 1):
        raise ValueError("bandwidth_efficiency 必须在 (0, 1] 内")
    if not (0 < runtime.compute_efficiency <= 1):
        raise ValueError("compute_efficiency 必须在 (0, 1] 内")
    if runtime.weight_overhead_ratio < 0:
        raise ValueError("weight_overhead_ratio 不能小于 0")
    if runtime.runtime_overhead_ratio < 0:
        raise ValueError("runtime_overhead_ratio 不能小于 0")
    if runtime.attention_compute_coefficient is not None and runtime.attention_compute_coefficient < 0:
        raise ValueError("attention_compute_coefficient 不能小于 0")
