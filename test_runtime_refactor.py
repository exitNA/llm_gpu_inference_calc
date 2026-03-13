from gradio_app import build_default_configs, build_default_result
from gpu_sizing import ModelConfig, GPUConfig, TrafficConfig, HAConfig, RuntimeConfig

try:
    model, traffic, gpu, ha, runtime, use_p95 = build_default_configs()
    print("build_default_configs() returned:", type(model), type(traffic), type(gpu), type(ha), type(runtime))
    assert isinstance(runtime, RuntimeConfig)
    
    result = build_default_result()
    print("build_default_result() passed. Memory: ", result.get("avg_total_memory_gb"))
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
