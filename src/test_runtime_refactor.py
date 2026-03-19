from gradio_ui.analysis import build_default_result
from gradio_ui.config import UIInputs
from gpu_sizing_core import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig
from gpu_sizing_core.service import evaluate_single_model

try:
    model, traffic, gpu, runtime = UIInputs.default().build_configs()
    defaults = UIInputs.default()
    print(
        "UIInputs.default().build_configs() returned:",
        type(model),
        type(traffic),
        type(gpu),
        type(runtime),
    )
    assert isinstance(model, ModelConfig)
    assert isinstance(traffic, TrafficConfig)
    assert isinstance(gpu, GPUConfig)
    assert isinstance(runtime, RuntimeConfig)
    assert defaults.qps_estimation_mode == "poisson_from_daily_requests"
    assert defaults.daily_request_count == 50000
    assert defaults.poisson_time_window_sec == 10

    result = build_default_result()
    print("build_default_result() passed. G_req:", result.get("business_gpu_count"))
    assert "gpu_count_by_memory" in result
    assert "prefill_gpu_count_by_throughput" in result
    assert "decode_gpu_count_by_throughput" in result
    assert "prefill_latency_ok" in result
    assert "decode_latency_ok" in result
    assert "sustainable_qps_p95" in result
    assert "max_concurrency_by_memory_p95" in result
    assert "calculation_process_sections" in result
    assert result["calculation_process_sections"]

    poisson_inputs = UIInputs.default()
    poisson_inputs = UIInputs(
        *(
            poisson_inputs.model_dropdown,
            poisson_inputs.precision_override,
            poisson_inputs.gpu_preset_key,
            "poisson_from_daily_requests",
            poisson_inputs.lambda_peak_qps,
            432000,
            180,
            1,
            99,
            poisson_inputs.p95_input_tokens,
            poisson_inputs.p95_output_tokens,
            poisson_inputs.ttft_p95_sec,
            poisson_inputs.e2e_p95_sec,
            poisson_inputs.concurrency_estimation_mode,
            poisson_inputs.direct_peak_concurrency,
            poisson_inputs.concurrency_safety_factor_pct,
            poisson_inputs.weight_overhead_ratio,
            poisson_inputs.runtime_overhead_ratio,
            poisson_inputs.usable_vram_ratio,
            poisson_inputs.bandwidth_efficiency,
            poisson_inputs.compute_efficiency,
        )
    )
    model, traffic, gpu, runtime = poisson_inputs.build_configs()
    poisson_result = evaluate_single_model(model=model, traffic=traffic, gpu=gpu, runtime=runtime)
    assert poisson_result["qps_model_label"] == "泊松日均反推峰值 QPS"
    assert poisson_result["lambda_peak_qps_effective"] >= traffic.lambda_peak_qps

    direct_concurrency_inputs = UIInputs.default()
    direct_concurrency_inputs = UIInputs(
        *(
            direct_concurrency_inputs.model_dropdown,
            direct_concurrency_inputs.precision_override,
            direct_concurrency_inputs.gpu_preset_key,
            direct_concurrency_inputs.qps_estimation_mode,
            direct_concurrency_inputs.lambda_peak_qps,
            direct_concurrency_inputs.daily_request_count,
            direct_concurrency_inputs.qps_burst_factor_pct,
            direct_concurrency_inputs.poisson_time_window_sec,
            direct_concurrency_inputs.poisson_qps_quantile_pct,
            direct_concurrency_inputs.p95_input_tokens,
            direct_concurrency_inputs.p95_output_tokens,
            direct_concurrency_inputs.ttft_p95_sec,
            direct_concurrency_inputs.e2e_p95_sec,
            "direct_peak_concurrency",
            320,
            direct_concurrency_inputs.concurrency_safety_factor_pct,
            direct_concurrency_inputs.weight_overhead_ratio,
            direct_concurrency_inputs.runtime_overhead_ratio,
            direct_concurrency_inputs.usable_vram_ratio,
            direct_concurrency_inputs.bandwidth_efficiency,
            direct_concurrency_inputs.compute_efficiency,
        )
    )
    model, traffic, gpu, runtime = direct_concurrency_inputs.build_configs()
    direct_concurrency_result = evaluate_single_model(model=model, traffic=traffic, gpu=gpu, runtime=runtime)
    assert direct_concurrency_result["concurrency_model_label"] == "直接输入峰值在途请求量"
    assert direct_concurrency_result["c_peak_budget"] == 320
    print("SUCCESS")
except Exception:
    import traceback

    traceback.print_exc()
