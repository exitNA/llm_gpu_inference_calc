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
    if gpu.int4_tflops is not None or gpu.int8_tflops is not None:
        choices.append("int4")
    if gpu.int8_tflops is not None:
        choices.append("int8")
    if gpu.fp8_tflops is not None:
        choices.append("fp8")
    if gpu.fp16_tflops is not None:
        choices.append("fp16")
    if gpu.bf16_tflops is not None:
        choices.append("bf16")
    if not choices:
        choices = ["fp16", "bf16"]
    value = current_precision if current_precision in choices else choices[-1]
    return gr.update(choices=choices, value=value)


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


def build_traffic_mode_hint_html(qps_estimation_mode: str, concurrency_estimation_mode: str) -> str:
    if qps_estimation_mode == "direct_peak_qps":
        qps_title = "峰值 QPS"
        qps_copy = "已有监控或压测值时使用。"
    else:
        qps_title = "Poisson 反推峰值 QPS"
        qps_copy = "只有日调用量时使用。推荐先用 10s / 99%。"

    if concurrency_estimation_mode == "little_law":
        conc_title = "Little 近似峰值在途"
        conc_copy = "没有在线并发监控时使用。"
    else:
        conc_title = "直接输入峰值在途"
        conc_copy = "已有活跃请求峰值监控时使用。"

    return f"""
    <div class="traffic-mode-hint">
      <div class="traffic-mode-hint-block">
        <div class="traffic-mode-hint-kicker">QPS 口径</div>
        <div class="traffic-mode-hint-title">{escape(qps_title)}</div>
        <div class="traffic-mode-hint-copy">{escape(qps_copy)}</div>
      </div>
      <div class="traffic-mode-hint-divider"></div>
      <div class="traffic-mode-hint-block">
        <div class="traffic-mode-hint-kicker">在途口径</div>
        <div class="traffic-mode-hint-title">{escape(conc_title)}</div>
        <div class="traffic-mode-hint-copy">{escape(conc_copy)}</div>
      </div>
    </div>
    """


def update_traffic_input_visibility(qps_estimation_mode: str, concurrency_estimation_mode: str):
    gr = import_gradio()
    return (
        build_traffic_mode_hint_html(qps_estimation_mode, concurrency_estimation_mode),
        gr.update(visible=qps_estimation_mode == "direct_peak_qps"),
        gr.update(visible=qps_estimation_mode == "poisson_from_daily_requests"),
        gr.update(visible=concurrency_estimation_mode == "little_law"),
        gr.update(visible=concurrency_estimation_mode == "direct_peak_concurrency"),
    )
