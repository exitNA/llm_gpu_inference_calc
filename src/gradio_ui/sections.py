from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from presets import get_gpu_choices, get_model_choices
from ui.views import (
    build_final_summary_html,
    build_kv_detail_rows,
    build_memory_analysis_html,
    build_overview_html,
    build_request_detail_rows,
    build_throughput_analysis_html,
)

from .constants import KV_TABLE_HEADERS, REQUEST_TABLE_HEADERS
from .config import UIInputs
from .runtime import build_config_section_header_html, build_traffic_mode_hint_html, build_traffic_playbook_html


@dataclass(frozen=True)
class SidebarComponents:
    model_dropdown: Any
    precision_dropdown: Any
    gpu_preset_key: Any
    traffic_playbook_html: Any
    traffic_mode_hint_html: Any
    qps_estimation_mode: Any
    qps_direct_group: Any
    lambda_peak_qps: Any
    qps_poisson_group: Any
    daily_request_count: Any
    qps_burst_factor_pct: Any
    poisson_time_window_sec: Any
    poisson_qps_quantile_pct: Any
    p95_input_tokens: Any
    p95_output_tokens: Any
    ttft_p95_sec: Any
    e2e_p95_sec: Any
    concurrency_estimation_mode: Any
    concurrency_little_group: Any
    concurrency_direct_group: Any
    direct_peak_concurrency: Any
    concurrency_safety_factor_pct: Any
    weight_overhead_ratio: Any
    runtime_overhead_ratio: Any
    usable_vram_ratio: Any
    bandwidth_efficiency: Any
    compute_efficiency: Any
    reset_button: Any


@dataclass(frozen=True)
class ResultComponents:
    overview_html: Any
    memory_html: Any
    throughput_html: Any
    final_html: Any
    request_table: Any
    kv_table: Any
    calc_text: Any
    raw_json: Any


def build_sidebar(gr, defaults: UIInputs) -> SidebarComponents:
    with gr.Column(scale=4, min_width=420, elem_classes=["workspace-sidebar"]):
        with gr.Group(elem_classes=["config-panel"]):
            with gr.Group(elem_classes=["config-section", "config-section-model"]):
                gr.HTML(build_config_section_header_html("🧠", "模型配置", "模型选择与显存附加系数"))
                model_dropdown = gr.Dropdown(
                    label="选择模型",
                    choices=get_model_choices(),
                    value=defaults.model_dropdown,
                )
                weight_overhead_ratio = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="权重附加系数 α_w (%)",
                    value=defaults.weight_overhead_ratio,
                )
                runtime_overhead_ratio = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="运行时固定显存系数 α_r (%)",
                    value=defaults.runtime_overhead_ratio,
                )

            with gr.Group(elem_classes=["config-section", "config-section-gpu"]):
                gr.HTML(build_config_section_header_html("🖥️", "显卡配置", "GPU 规格与效率系数"))
                gpu_preset_key = gr.Dropdown(
                    label="选择显卡",
                    choices=get_gpu_choices(),
                    value=defaults.gpu_preset_key,
                )
                precision_dropdown = gr.Radio(
                    label="推理精度",
                    choices=["fp8", "fp16", "bf16"],
                    value=defaults.precision_override,
                    elem_classes=["compact-radio"],
                )
                usable_vram_ratio = gr.Slider(
                    minimum=50,
                    maximum=100,
                    step=1,
                    label="可用显存比例 η_vram (%)",
                    value=defaults.usable_vram_ratio,
                )
                bandwidth_efficiency = gr.Slider(
                    minimum=10,
                    maximum=100,
                    step=1,
                    label="带宽利用率 η_bw (%)",
                    value=defaults.bandwidth_efficiency,
                )
                compute_efficiency = gr.Slider(
                    minimum=10,
                    maximum=100,
                    step=1,
                    label="算力利用率 η_cmp (%)",
                    value=defaults.compute_efficiency,
                )

            with gr.Group(elem_classes=["config-section", "config-section-traffic"]):
                gr.HTML(
                    build_config_section_header_html(
                        "🚦",
                        "业务目标",
                        "按“流量来源 → 在途口径 → 请求画像”分步骤填写，界面只显示当前模式真正需要的字段。",
                    )
                )
                traffic_playbook_html = gr.HTML(build_traffic_playbook_html())
                qps_estimation_mode = gr.Radio(
                    label="第 1 步：流量来源",
                    choices=[
                        ("我已经知道峰值 QPS", "direct_peak_qps"),
                        ("我只有日调用量", "poisson_from_daily_requests"),
                    ],
                    value=defaults.qps_estimation_mode,
                    elem_classes=["compact-radio", "compact-radio-dual"],
                )
                traffic_mode_hint_html = gr.HTML(
                    build_traffic_mode_hint_html(
                        defaults.qps_estimation_mode,
                        defaults.concurrency_estimation_mode,
                    )
                )
                with gr.Group(
                    visible=defaults.qps_estimation_mode == "direct_peak_qps",
                    elem_classes=["traffic-mode-panel", "traffic-mode-panel-direct"],
                ) as qps_direct_group:
                    gr.HTML(
                        """
                        <div class="traffic-subsection-head">
                          <div class="traffic-subsection-kicker">峰值请求率</div>
                          <div class="traffic-subsection-title">你已经有峰值监控或压测结果</div>
                          <div class="traffic-subsection-copy">直接填写模型调用层面的峰值 QPS。若这是用户请求 QPS，请先折算成模型调用 QPS。</div>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["config-field-grid"]):
                        lambda_peak_qps = gr.Number(
                            label="峰值 QPS",
                            value=defaults.lambda_peak_qps,
                            precision=2,
                            info="不知道时，不要硬填；切到“我只有日调用量”更合适。",
                        )
                with gr.Group(
                    visible=defaults.qps_estimation_mode == "poisson_from_daily_requests",
                    elem_classes=["traffic-mode-panel", "traffic-mode-panel-poisson"],
                ) as qps_poisson_group:
                    gr.HTML(
                        """
                        <div class="traffic-subsection-head">
                          <div class="traffic-subsection-kicker">日调用量反推峰值</div>
                          <div class="traffic-subsection-title">你没有峰值 QPS，但有日调用量</div>
                          <div class="traffic-subsection-copy">系统会用 Poisson 分位数从日调用量反推 sizing 使用的峰值 QPS。没有更细日志时，建议先试：高峰放大 200%、时间窗 10s、分位数 99%。</div>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["config-field-grid"]):
                        daily_request_count = gr.Number(
                            label="日调用量",
                            value=defaults.daily_request_count,
                            precision=0,
                            info="按模型调用次数填，不是用户会话数。",
                        )
                        qps_burst_factor_pct = gr.Number(
                            label="高峰放大系数 (%)",
                            value=defaults.qps_burst_factor_pct,
                            precision=0,
                            info="没有更细数据时，建议先用 200%。",
                        )
                    with gr.Row(elem_classes=["config-field-grid"]):
                        poisson_time_window_sec = gr.Number(
                            label="统计时间窗 (s)",
                            value=defaults.poisson_time_window_sec,
                            precision=2,
                            info="推荐先用 10s；越小越保守。",
                        )
                        poisson_qps_quantile_pct = gr.Number(
                            label="分位数 (%)",
                            value=defaults.poisson_qps_quantile_pct,
                            precision=0,
                            info="推荐先用 99%。",
                        )
                concurrency_estimation_mode = gr.Radio(
                    label="第 2 步：峰值在途口径",
                    choices=[
                        ("我没有并发监控，用 Little 近似", "little_law"),
                        ("我已经知道峰值在途请求量", "direct_peak_concurrency"),
                    ],
                    value=defaults.concurrency_estimation_mode,
                    elem_classes=["compact-radio", "compact-radio-dual"],
                )
                with gr.Group(
                    visible=defaults.concurrency_estimation_mode == "little_law",
                    elem_classes=["traffic-mode-panel", "traffic-mode-panel-little"],
                ) as concurrency_little_group:
                    gr.HTML(
                        """
                        <div class="traffic-subsection-head">
                          <div class="traffic-subsection-kicker">Little 定律近似</div>
                          <div class="traffic-subsection-title">适合大多数还没有在线并发监控的场景</div>
                          <div class="traffic-subsection-copy">系统会按“峰值 QPS × E2E × 安全系数”估算峰值在途请求量。默认先用 110% 安全系数即可。</div>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["config-field-grid"]):
                        concurrency_safety_factor_pct = gr.Number(
                            label="在途安全系数 (%)",
                            value=defaults.concurrency_safety_factor_pct,
                            precision=0,
                            info="没有额外排队风险时，建议先用 110%。",
                        )
                with gr.Group(
                    visible=defaults.concurrency_estimation_mode == "direct_peak_concurrency",
                    elem_classes=["traffic-mode-panel", "traffic-mode-panel-direct-concurrency"],
                ) as concurrency_direct_group:
                    gr.HTML(
                        """
                        <div class="traffic-subsection-head">
                          <div class="traffic-subsection-kicker">直接输入峰值在途</div>
                          <div class="traffic-subsection-title">适合你已经有服务端活跃请求监控</div>
                          <div class="traffic-subsection-copy">直接填写系统峰值时刻同时挂着的请求量。没有这类监控时，不建议选这个模式。</div>
                        </div>
                        """
                    )
                    with gr.Row(elem_classes=["config-field-grid"]):
                        direct_peak_concurrency = gr.Number(
                            label="峰值在途请求量",
                            value=defaults.direct_peak_concurrency,
                            precision=0,
                            info="来自网关、调度器或服务日志里的活跃请求峰值。",
                        )
                gr.HTML(
                    """
                    <div class="traffic-subsection-head traffic-subsection-head-profile">
                      <div class="traffic-subsection-kicker">第 3 步：请求画像与时延目标</div>
                      <div class="traffic-subsection-title">不知道怎么填时，先用默认业务画像</div>
                      <div class="traffic-subsection-copy">默认值对应“长上下文 + 长输出”的保守在线问答口径。若你的业务更轻，请把输入/输出长度调低，否则会明显高估卡数。</div>
                      <div class="traffic-profile-strip">
                        <span class="traffic-profile-pill">轻问答: 输入 500 / 输出 200</span>
                        <span class="traffic-profile-pill">中等分析: 输入 1000 / 输出 300</span>
                        <span class="traffic-profile-pill traffic-profile-pill-active">保守默认: 输入 3000 / 输出 1000</span>
                      </div>
                    </div>
                    """
                )
                with gr.Row(elem_classes=["config-field-grid"]):
                    p95_input_tokens = gr.Number(
                        label="P95 输入长度",
                        value=defaults.p95_input_tokens,
                        precision=0,
                        info="按 token 填。默认 3000 偏保守。",
                    )
                    p95_output_tokens = gr.Number(
                        label="P95 输出长度",
                        value=defaults.p95_output_tokens,
                        precision=0,
                        info="长输出场景最容易把 G_dec 拉高。",
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    ttft_p95_sec = gr.Number(
                        label="TTFT 目标 (s)",
                        value=defaults.ttft_p95_sec,
                        precision=2,
                        info="不知道时先用 3s。",
                    )
                    e2e_p95_sec = gr.Number(
                        label="E2E 目标 (s)",
                        value=defaults.e2e_p95_sec,
                        precision=2,
                        info="不知道时先用 120s，且必须大于 TTFT。",
                    )

            with gr.Row(elem_classes=["button-row"]):
                reset_button = gr.Button("恢复默认", variant="secondary")

    return SidebarComponents(
        model_dropdown=model_dropdown,
        precision_dropdown=precision_dropdown,
        gpu_preset_key=gpu_preset_key,
        traffic_playbook_html=traffic_playbook_html,
        traffic_mode_hint_html=traffic_mode_hint_html,
        qps_estimation_mode=qps_estimation_mode,
        qps_direct_group=qps_direct_group,
        lambda_peak_qps=lambda_peak_qps,
        qps_poisson_group=qps_poisson_group,
        daily_request_count=daily_request_count,
        qps_burst_factor_pct=qps_burst_factor_pct,
        poisson_time_window_sec=poisson_time_window_sec,
        poisson_qps_quantile_pct=poisson_qps_quantile_pct,
        p95_input_tokens=p95_input_tokens,
        p95_output_tokens=p95_output_tokens,
        ttft_p95_sec=ttft_p95_sec,
        e2e_p95_sec=e2e_p95_sec,
        concurrency_estimation_mode=concurrency_estimation_mode,
        concurrency_little_group=concurrency_little_group,
        concurrency_direct_group=concurrency_direct_group,
        direct_peak_concurrency=direct_peak_concurrency,
        concurrency_safety_factor_pct=concurrency_safety_factor_pct,
        weight_overhead_ratio=weight_overhead_ratio,
        runtime_overhead_ratio=runtime_overhead_ratio,
        usable_vram_ratio=usable_vram_ratio,
        bandwidth_efficiency=bandwidth_efficiency,
        compute_efficiency=compute_efficiency,
        reset_button=reset_button,
    )


def build_results_panel(gr, default_result: dict[str, Any] | None = None) -> ResultComponents:
    initial_result = default_result or {}
    with gr.Column(scale=8, min_width=860, elem_classes=["workspace-content"]):
        overview_html = gr.HTML(value="" if default_result is None else build_overview_html(initial_result))
        memory_html = gr.HTML(value="" if default_result is None else build_memory_analysis_html(initial_result))
        throughput_html = gr.HTML(value="" if default_result is None else build_throughput_analysis_html(initial_result))
        final_html = gr.HTML(value="" if default_result is None else build_final_summary_html(initial_result))

        with gr.Accordion("📋 详细数据与原始 JSON", open=False, elem_classes=["custom-accordion"]):
            with gr.Tabs():
                with gr.Tab("业务与 Cache"):
                    request_table = gr.Dataframe(
                        headers=REQUEST_TABLE_HEADERS,
                        datatype=["str"] * len(REQUEST_TABLE_HEADERS),
                        interactive=False,
                        label="业务口径",
                        value=[] if default_result is None else build_request_detail_rows(initial_result),
                    )
                    kv_table = gr.Dataframe(
                        headers=KV_TABLE_HEADERS,
                        datatype=["str"] * len(KV_TABLE_HEADERS),
                        interactive=False,
                        label="Cache 口径",
                        value=[] if default_result is None else build_kv_detail_rows(initial_result),
                    )
                with gr.Tab("计算过程文本"):
                    calc_text = gr.Textbox(
                        label="计算过程",
                        value="" if default_result is None else initial_result["calculation_process_text"],
                        lines=24,
                        interactive=False,
                    )
                with gr.Tab("原始 JSON"):
                    raw_json = gr.JSON(label="原始结果", value=initial_result)

    return ResultComponents(
        overview_html=overview_html,
        memory_html=memory_html,
        throughput_html=throughput_html,
        final_html=final_html,
        request_table=request_table,
        kv_table=kv_table,
        calc_text=calc_text,
        raw_json=raw_json,
    )
