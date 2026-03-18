from __future__ import annotations

from html import escape

from presets import get_gpu_preset


def import_gradio():
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 gradio，请先执行 `uv sync`。") from exc
    if not hasattr(gr, "Blocks"):
        raise RuntimeError("当前环境中的 gradio 不可用，请先执行 `uv sync` 安装完整依赖。")
    return gr


def build_theme(gr):
    return gr.themes.Default(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")


def update_precision_choices(gpu_preset_key: str, current_precision: str):
    gr = import_gradio()
    gpu = get_gpu_preset(gpu_preset_key).config
    choices = []
    if gpu.fp8_tflops is not None:
        choices.append("fp8")
    if gpu.fp16_tflops is not None:
        choices.append("fp16")
    if gpu.bf16_tflops is not None:
        choices.append("bf16")
    if not choices:
        choices = ["fp16", "bf16"]
    value = current_precision if current_precision in choices else choices[-1]
    return gr.Dropdown(choices=choices, value=value)


def build_config_section_header_html(icon: str, title: str, description: str) -> str:
    return f"""
    <div class="config-section-header">
      <div class="config-section-title-row">
        <span class="config-section-icon">{escape(icon)}</span>
        <div>
          <div class="config-section-title">{escape(title)}</div>
          <div class="config-section-description">{escape(description)}</div>
        </div>
      </div>
    </div>
    """
