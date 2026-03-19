from __future__ import annotations

from dataclasses import dataclass


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
    precision: str = "fp8"
    kv_cache_dtype: str = "fp16"
    weight_overhead_ratio: float = 0.15
    runtime_overhead_ratio: float = 0.05
    usable_vram_ratio: float = 0.95
    bandwidth_efficiency: float = 0.65
    compute_efficiency: float = 0.60
    attention_compute_coefficient: float | None = None
    framework: str = "vllm"


@dataclass
class TrafficConfig:
    lambda_peak_qps: float
    p95_input_tokens: int
    p95_output_tokens: int
    ttft_p95_target_sec: float
    e2e_p95_target_sec: float
    p95_total_tokens_override: int | None = None
    concurrency_safety_factor: float = 1.0

    @property
    def p95_total_tokens(self) -> int:
        if self.p95_total_tokens_override is not None:
            return self.p95_total_tokens_override
        return self.p95_input_tokens + self.p95_output_tokens

    @property
    def decode_p95_budget_sec(self) -> float:
        return self.e2e_p95_target_sec - self.ttft_p95_target_sec


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
