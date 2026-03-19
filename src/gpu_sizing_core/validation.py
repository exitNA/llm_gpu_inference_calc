from __future__ import annotations

from .models import RuntimeConfig, TrafficConfig


def validate_traffic_config(traffic: TrafficConfig) -> None:
    if traffic.p95_input_tokens <= 0 or traffic.p95_output_tokens < 0:
        raise ValueError("P95 输入输出长度必须合法")
    if traffic.ttft_p95_target_sec <= 0:
        raise ValueError("ttft_p95_target_sec 必须大于 0")
    if traffic.e2e_p95_target_sec <= traffic.ttft_p95_target_sec:
        raise ValueError("e2e_p95_target_sec 必须大于 ttft_p95_target_sec")
    if traffic.qps_estimation_mode not in {"direct_peak_qps", "poisson_from_daily_requests"}:
        raise ValueError("qps_estimation_mode 不合法")
    if traffic.concurrency_estimation_mode not in {"little_law", "direct_peak_concurrency"}:
        raise ValueError("concurrency_estimation_mode 不合法")
    if traffic.qps_estimation_mode == "direct_peak_qps":
        if traffic.lambda_peak_qps <= 0:
            raise ValueError("直接峰值 QPS 模式要求 lambda_peak_qps 大于 0")
    else:
        if traffic.daily_request_count is None or traffic.daily_request_count <= 0:
            raise ValueError("Poisson QPS 模式要求 daily_request_count 大于 0")
        if traffic.qps_burst_factor < 1.0:
            raise ValueError("qps_burst_factor 必须大于等于 1.0")
        if traffic.poisson_time_window_sec <= 0:
            raise ValueError("poisson_time_window_sec 必须大于 0")
        if not 0 < traffic.poisson_qps_quantile < 1:
            raise ValueError("poisson_qps_quantile 必须在 (0, 1) 内")
    if traffic.concurrency_estimation_mode == "direct_peak_concurrency":
        if traffic.direct_peak_concurrency is None or traffic.direct_peak_concurrency <= 0:
            raise ValueError("direct_peak_concurrency 必须大于 0")
    elif traffic.concurrency_safety_factor < 1.0:
        raise ValueError("Little 定律模式要求 concurrency_safety_factor 大于等于 1.0")


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
