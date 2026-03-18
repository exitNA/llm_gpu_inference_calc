from gradio_ui.analysis import build_default_result
from gradio_ui.config import build_default_configs
from gpu_sizing_core import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig

try:
    model, traffic, gpu, runtime = build_default_configs()
    print(
        "build_default_configs() returned:",
        type(model),
        type(traffic),
        type(gpu),
        type(runtime),
    )
    assert isinstance(model, ModelConfig)
    assert isinstance(traffic, TrafficConfig)
    assert isinstance(gpu, GPUConfig)
    assert isinstance(runtime, RuntimeConfig)

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
    print("SUCCESS")
except Exception:
    import traceback

    traceback.print_exc()
