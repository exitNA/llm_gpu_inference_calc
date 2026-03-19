"""Load model / GPU configurations from JSON data files in ``data/``."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_sizing_core.models import GPUConfig, ModelConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
GPUS_DIR = DATA_DIR / "gpus"

def load_model_config(path: Path) -> ModelConfig:
    """Read a model JSON file and return a ``ModelConfig``."""
    data: dict = json.loads(path.read_text(encoding="utf-8"))

    num_params_billion: float = data["total_params_b"]

    activated_params_billion: float | None = None
    if "activated_params_per_token_b" in data and data["activated_params_per_token_b"]:
        activated_params_billion = data["activated_params_per_token_b"]

    return ModelConfig(
        model_name=data.get("model_name", path.stem),
        num_params_billion=num_params_billion,
        num_layers=data["num_layers"],
        hidden_size=data["hidden_size"],
        arch_family=data.get("arch_family", "dense"),
        attention_type=data.get("attention_type", "mha"),
        activated_params_billion=activated_params_billion,
        num_heads=data.get("num_heads"),
        num_kv_heads=data.get("num_kv_heads"),
        head_dim=data.get("head_dim"),
        latent_cache_dim=data.get("latent_cache_dim"),
        cache_aux_bytes_per_token_per_layer=data.get("cache_aux_bytes_per_token_per_layer", 0.0),
        cache_bytes_per_token_per_layer=data.get("cache_bytes_per_token_per_layer"),
    )


def load_gpu_config(path: Path) -> GPUConfig:
    """Read a GPU JSON file and return a ``GPUConfig``."""
    data: dict = json.loads(path.read_text(encoding="utf-8"))

    return GPUConfig(
        gpu_name=data.get("gpu_name", path.stem),
        vram_gb=data["vram_gb"],
        memory_bandwidth_gb_per_sec=data.get("memory_bandwidth_gb_per_sec"),
        fp32_tflops=data.get("fp32_tflops"),
        fp16_tflops=data.get("fp16_tflops"),
        bf16_tflops=data.get("bf16_tflops"),
        fp8_tflops=data.get("fp8_tflops"),
        int8_tflops=data.get("int8_tflops"),
    )


def load_all_model_configs() -> dict[str, ModelConfig]:
    """Scan ``data/models/*.json`` and return a ``{stem: ModelConfig}`` dict."""
    configs: dict[str, ModelConfig] = {}
    if not MODELS_DIR.is_dir():
        return configs
    for path in sorted(MODELS_DIR.glob("*.json")):
        configs[path.stem] = load_model_config(path)
    return configs


def load_all_gpu_configs() -> dict[str, GPUConfig]:
    """Scan ``data/gpus/*.json`` and return a ``{stem: GPUConfig}`` dict."""
    configs: dict[str, GPUConfig] = {}
    if not GPUS_DIR.is_dir():
        return configs
    for path in sorted(GPUS_DIR.glob("*.json")):
        configs[path.stem] = load_gpu_config(path)
    return configs
