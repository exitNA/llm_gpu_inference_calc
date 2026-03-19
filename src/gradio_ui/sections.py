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
from .runtime import build_config_section_header_html


@dataclass(frozen=True)
class SidebarComponents:
    model_dropdown: Any
    precision_dropdown: Any
    gpu_preset_key: Any
    qps_estimation_mode: Any
    lambda_peak_qps: Any
    daily_request_count: Any
    qps_burst_factor_pct: Any
    poisson_time_window_sec: Any
    poisson_qps_quantile_pct: Any
    p95_input_tokens: Any
    p95_output_tokens: Any
    ttft_p95_sec: Any
    e2e_p95_sec: Any
    concurrency_estimation_mode: Any
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
                        "支持直接峰值 QPS、泊松日均反推 QPS，以及 Little/直接在途两种显存口径。",
                    )
                )
                qps_estimation_mode = gr.Radio(
                    label="QPS 建模方式",
                    choices=[
                        ("直接输入峰值 QPS", "direct_peak_qps"),
                        ("泊松日均反推峰值 QPS", "poisson_from_daily_requests"),
                    ],
                    value=defaults.qps_estimation_mode,
                    elem_classes=["compact-radio"],
                )
                with gr.Row(elem_classes=["config-field-grid"]):
                    lambda_peak_qps = gr.Number(
                        label="峰值 QPS",
                        value=defaults.lambda_peak_qps,
                        precision=2,
                    )
                    daily_request_count = gr.Number(
                        label="日调用量",
                        value=defaults.daily_request_count,
                        precision=0,
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    qps_burst_factor_pct = gr.Number(
                        label="QPS 高峰放大系数 (%)",
                        value=defaults.qps_burst_factor_pct,
                        precision=0,
                    )
                    poisson_time_window_sec = gr.Number(
                        label="Poisson 时间窗 (s)",
                        value=defaults.poisson_time_window_sec,
                        precision=2,
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    poisson_qps_quantile_pct = gr.Number(
                        label="Poisson 分位数 (%)",
                        value=defaults.poisson_qps_quantile_pct,
                        precision=0,
                    )
                    concurrency_estimation_mode = gr.Radio(
                        label="在途建模方式",
                        choices=[
                            ("Little 定律近似", "little_law"),
                            ("直接输入峰值在途", "direct_peak_concurrency"),
                        ],
                        value=defaults.concurrency_estimation_mode,
                        elem_classes=["compact-radio"],
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    direct_peak_concurrency = gr.Number(
                        label="峰值在途请求量",
                        value=defaults.direct_peak_concurrency,
                        precision=0,
                    )
                    concurrency_safety_factor_pct = gr.Number(
                        label="在途安全系数 (%)",
                        value=defaults.concurrency_safety_factor_pct,
                        precision=0,
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    p95_input_tokens = gr.Number(
                        label="输入长度",
                        value=defaults.p95_input_tokens,
                        precision=0,
                    )
                    p95_output_tokens = gr.Number(
                        label="输出长度",
                        value=defaults.p95_output_tokens,
                        precision=0,
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    ttft_p95_sec = gr.Number(
                        label="TTFT 目标 (s)",
                        value=defaults.ttft_p95_sec,
                        precision=2,
                    )
                    e2e_p95_sec = gr.Number(
                        label="E2E 目标 (s)",
                        value=defaults.e2e_p95_sec,
                        precision=2,
                    )

            with gr.Row(elem_classes=["button-row"]):
                reset_button = gr.Button("恢复默认", variant="secondary")

    return SidebarComponents(
        model_dropdown=model_dropdown,
        precision_dropdown=precision_dropdown,
        gpu_preset_key=gpu_preset_key,
        qps_estimation_mode=qps_estimation_mode,
        lambda_peak_qps=lambda_peak_qps,
        daily_request_count=daily_request_count,
        qps_burst_factor_pct=qps_burst_factor_pct,
        poisson_time_window_sec=poisson_time_window_sec,
        poisson_qps_quantile_pct=poisson_qps_quantile_pct,
        p95_input_tokens=p95_input_tokens,
        p95_output_tokens=p95_output_tokens,
        ttft_p95_sec=ttft_p95_sec,
        e2e_p95_sec=e2e_p95_sec,
        concurrency_estimation_mode=concurrency_estimation_mode,
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
