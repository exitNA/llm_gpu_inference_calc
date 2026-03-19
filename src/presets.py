from dataclasses import dataclass

from data_loader import load_all_gpu_configs, load_all_model_configs
from gpu_sizing_core.models import GPUConfig, ModelConfig, TrafficConfig


@dataclass(frozen=True)
class ModelPreset:
    config: ModelConfig


@dataclass(frozen=True)
class GPUPreset:
    config: GPUConfig


@dataclass(frozen=True)
class BusinessProfile:
    name: str
    lambda_peak_qps: float
    p95_input_tokens: int
    p95_output_tokens: int
    ttft_p95_sec: float
    e2e_p95_sec: float
    concurrency_safety_factor: float
    note: str


MODEL_PRESETS: dict[str, ModelPreset] = {}
GPU_PRESETS: dict[str, GPUPreset] = {}

for _stem, _cfg in load_all_model_configs().items():
    MODEL_PRESETS[_stem] = ModelPreset(config=_cfg)

for _stem, _cfg in load_all_gpu_configs().items():
    GPU_PRESETS[_stem] = GPUPreset(config=_cfg)


DEFAULT_BUSINESS_PROFILE = BusinessProfile(
    name="标准在线推理",
    lambda_peak_qps=5,
    p95_input_tokens=20000,
    p95_output_tokens=10000,
    ttft_p95_sec=5,
    e2e_p95_sec=60,
    concurrency_safety_factor=1.1,
    note="默认画像使用峰值 QPS 与 P95 输入输出长度、P95 时延目标作为 sizing 主口径。",
)

DEFAULT_MODEL_PRESET_KEY = "deepseek_r1_671b"
DEFAULT_GPU_PRESET_KEY = "h20"


def get_model_choices() -> list[tuple[str, str]]:
    return [(preset.config.model_name, key) for key, preset in MODEL_PRESETS.items()]


def get_default_model_key() -> str:
    return DEFAULT_MODEL_PRESET_KEY


def get_gpu_choices() -> list[tuple[str, str]]:
    return [(preset.config.gpu_name, key) for key, preset in GPU_PRESETS.items()]


def get_default_gpu_key() -> str:
    return DEFAULT_GPU_PRESET_KEY


def get_model_preset(preset_key: str) -> ModelPreset:
    return MODEL_PRESETS[preset_key]


def get_gpu_preset(preset_key: str) -> GPUPreset:
    return GPU_PRESETS[preset_key]


def build_default_traffic_config() -> TrafficConfig:
    profile = DEFAULT_BUSINESS_PROFILE
    return TrafficConfig(
        lambda_peak_qps=profile.lambda_peak_qps,
        p95_input_tokens=profile.p95_input_tokens,
        p95_output_tokens=profile.p95_output_tokens,
        ttft_p95_target_sec=profile.ttft_p95_sec,
        e2e_p95_target_sec=profile.e2e_p95_sec,
        concurrency_safety_factor=profile.concurrency_safety_factor,
    )
