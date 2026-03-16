from dataclasses import dataclass

from data_loader import load_all_gpu_configs, load_all_model_configs
from gpu_sizing import GPUConfig, ModelConfig, RequestShape, TrafficConfig


@dataclass(frozen=True)
class ModelPreset:
    config: ModelConfig


@dataclass(frozen=True)
class GPUPreset:
    config: GPUConfig


@dataclass(frozen=True)
class TrafficProfile:
    name: str
    batch_size_per_request: int
    decode_tps_per_concurrency: float
    prefill_tps_per_concurrency: float
    request_shapes: tuple[RequestShape, ...]
    note: str


MODEL_PRESETS: dict[str, ModelPreset] = {}

GPU_PRESETS: dict[str, GPUPreset] = {}

for _stem, _cfg in load_all_model_configs().items():
    MODEL_PRESETS[_stem] = ModelPreset(config=_cfg)

for _stem, _cfg in load_all_gpu_configs().items():
    GPU_PRESETS[_stem] = GPUPreset(config=_cfg)

# ── Traffic profile & defaults ──────────────────────────────────────

TRAFFIC_PROFILE = TrafficProfile(
    name="标准在线问答",
    batch_size_per_request=1,
    decode_tps_per_concurrency=20.0,
    prefill_tps_per_concurrency=500.0,
    request_shapes=(
        RequestShape(name="轻问答", ratio=0.6, avg_input_tokens=1200, avg_output_tokens=220),
        RequestShape(name="中等分析", ratio=0.3, avg_input_tokens=4000, avg_output_tokens=1000),
        RequestShape(name="长上下文", ratio=0.1, avg_input_tokens=20000, avg_output_tokens=10000),
    ),
    note="固定使用 60% 轻问答、30% 中等分析、10% 长上下文的混合流量画像。",
)

DEFAULT_MODEL_PRESET_KEY = "deepseek_r1_671b"
DEFAULT_GPU_PRESET_KEY = "h20"
DEFAULT_CONCURRENCY = 5


def build_default_traffic_targets(concurrency: int) -> tuple[float, float]:
    return (
        concurrency * TRAFFIC_PROFILE.prefill_tps_per_concurrency,
        concurrency * TRAFFIC_PROFILE.decode_tps_per_concurrency,
    )


def get_model_choices() -> list[tuple[str, str]]:
    """Return [(display_name, key), ...] for the model dropdown."""
    return [(preset.config.model_name, key) for key, preset in MODEL_PRESETS.items()]

def get_default_model_key() -> str:
    return DEFAULT_MODEL_PRESET_KEY


def get_gpu_choices() -> list[tuple[str, str]]:
    """Return [(display_name, key), ...] for the GPU dropdown."""
    return [(preset.config.gpu_name, key) for key, preset in GPU_PRESETS.items()]

def get_default_gpu_key() -> str:
    return DEFAULT_GPU_PRESET_KEY


def get_model_preset(preset_key: str) -> ModelPreset:
    return MODEL_PRESETS[preset_key]


def get_gpu_preset(preset_key: str) -> GPUPreset:
    return GPU_PRESETS[preset_key]


def build_traffic_config(concurrency: int) -> TrafficConfig:
    target_prefill_tps_total, target_decode_tps_total = build_default_traffic_targets(concurrency)
    return TrafficConfig(
        concurrency=concurrency,
        target_decode_tps_total=target_decode_tps_total,
        batch_size_per_request=TRAFFIC_PROFILE.batch_size_per_request,
        target_prefill_tps_total=target_prefill_tps_total,
        request_shapes=list(TRAFFIC_PROFILE.request_shapes),
    )
