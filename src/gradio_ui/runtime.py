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


def build_traffic_playbook_html() -> str:
    return """
    <div class="traffic-playbook">
      <div class="traffic-playbook-title">填写顺序</div>
      <div class="traffic-playbook-grid">
        <div class="traffic-playbook-card">
          <span class="traffic-playbook-step">1</span>
          <div>
            <div class="traffic-playbook-label">先选流量来源</div>
            <div class="traffic-playbook-copy">已有峰值 QPS 选“直接输入”；只有日调用量选“泊松反推”。</div>
          </div>
        </div>
        <div class="traffic-playbook-card">
          <span class="traffic-playbook-step">2</span>
          <div>
            <div class="traffic-playbook-label">再选在途口径</div>
            <div class="traffic-playbook-copy">没有并发监控就用 Little；已有活跃请求监控再填峰值在途。</div>
          </div>
        </div>
        <div class="traffic-playbook-card">
          <span class="traffic-playbook-step">3</span>
          <div>
            <div class="traffic-playbook-label">最后填业务画像</div>
            <div class="traffic-playbook-copy">不知道时先用默认：输入 3000、输出 1000、TTFT 3s、E2E 120s。</div>
          </div>
        </div>
      </div>
    </div>
    """


def build_traffic_mode_hint_html(qps_estimation_mode: str, concurrency_estimation_mode: str) -> str:
    if qps_estimation_mode == "direct_peak_qps":
        qps_title = "当前按“直接输入峰值 QPS”工作"
        qps_copy = "适合你已经有监控或压测峰值 QPS 的情况。只需要填峰值 QPS，不需要日调用量。"
    else:
        qps_title = "当前按“泊松日均反推峰值 QPS”工作"
        qps_copy = "适合你只有日调用量时使用。没有更细日志时，建议先用：高峰放大 200%、时间窗 10s、分位数 99%。"

    if concurrency_estimation_mode == "little_law":
        conc_title = "当前按“Little 定律近似”估峰值在途"
        conc_copy = "适合没有在线活跃请求监控时使用。通常先用安全系数 110%。"
    else:
        conc_title = "当前按“直接输入峰值在途请求量”工作"
        conc_copy = "只有当你已经有网关、调度器或服务端的活跃请求峰值监控时，才建议选这个。"

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
