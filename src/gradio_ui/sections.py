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
from .runtime import build_config_section_header_html


@dataclass(frozen=True)
class SidebarComponents:
    model_dropdown: Any
    precision_dropdown: Any
    gpu_preset_key: Any
    lambda_avg_qps: Any
    lambda_peak_qps: Any
    avg_input_tokens: Any
    avg_output_tokens: Any
    p95_input_tokens: Any
    p95_output_tokens: Any
    ttft_avg_sec: Any
    ttft_p95_sec: Any
    e2e_avg_sec: Any
    e2e_p95_sec: Any
    concurrency_safety_factor_pct: Any
    weight_overhead_ratio: Any
    runtime_overhead_ratio: Any
    usable_vram_ratio: Any
    bandwidth_efficiency: Any
    compute_efficiency: Any
    reset_button: Any

    def analysis_inputs(self) -> list[Any]:
        return [
            self.model_dropdown,
            self.precision_dropdown,
            self.gpu_preset_key,
            self.lambda_avg_qps,
            self.lambda_peak_qps,
            self.avg_input_tokens,
            self.avg_output_tokens,
            self.p95_input_tokens,
            self.p95_output_tokens,
            self.ttft_avg_sec,
            self.ttft_p95_sec,
            self.e2e_avg_sec,
            self.e2e_p95_sec,
            self.concurrency_safety_factor_pct,
            self.weight_overhead_ratio,
            self.runtime_overhead_ratio,
            self.usable_vram_ratio,
            self.bandwidth_efficiency,
            self.compute_efficiency,
        ]


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

    def analysis_outputs(self) -> list[Any]:
        return [
            self.overview_html,
            self.memory_html,
            self.throughput_html,
            self.final_html,
            self.request_table,
            self.kv_table,
            self.calc_text,
            self.raw_json,
        ]


def build_sidebar(gr, defaults: tuple[Any, ...]) -> SidebarComponents:
    with gr.Column(scale=4, min_width=420, elem_classes=["workspace-sidebar"]):
        with gr.Group(elem_classes=["config-panel"]):
            with gr.Group(elem_classes=["config-section", "config-section-model"]):
                gr.HTML(build_config_section_header_html("🧠", "模型配置", "模型选择与显存附加系数"))
                model_dropdown = gr.Dropdown(label="选择模型", choices=get_model_choices(), value=defaults[0])
                weight_overhead_ratio = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="权重附加系数 α_w (%)",
                    value=defaults[14],
                )
                runtime_overhead_ratio = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="运行时固定显存系数 α_r (%)",
                    value=defaults[15],
                )

            with gr.Group(elem_classes=["config-section", "config-section-gpu"]):
                gr.HTML(build_config_section_header_html("🖥️", "显卡配置", "GPU 规格与效率系数"))
                gpu_preset_key = gr.Dropdown(label="选择显卡", choices=get_gpu_choices(), value=defaults[2])
                precision_dropdown = gr.Radio(
                    label="推理精度",
                    choices=["fp8", "fp16", "bf16"],
                    value=defaults[1],
                    elem_classes=["compact-radio"],
                )
                usable_vram_ratio = gr.Slider(
                    minimum=50,
                    maximum=100,
                    step=1,
                    label="可用显存比例 η_vram (%)",
                    value=defaults[16],
                )
                bandwidth_efficiency = gr.Slider(
                    minimum=10,
                    maximum=100,
                    step=1,
                    label="带宽利用率 η_bw (%)",
                    value=defaults[17],
                )
                compute_efficiency = gr.Slider(
                    minimum=10,
                    maximum=100,
                    step=1,
                    label="算力利用率 η_cmp (%)",
                    value=defaults[18],
                )

            with gr.Group(elem_classes=["config-section", "config-section-traffic"]):
                gr.HTML(build_config_section_header_html("🚦", "业务目标", "QPS、长度画像与时延目标"))
                with gr.Row(elem_classes=["config-field-grid"]):
                    lambda_avg_qps = gr.Number(label="平均 QPS", value=defaults[3], precision=2)
                    lambda_peak_qps = gr.Number(label="峰值 QPS", value=defaults[4], precision=2)
                    concurrency_safety_factor_pct = gr.Number(
                        label="在途安全系数 (%)",
                        value=defaults[13],
                        precision=0,
                    )
                with gr.Row(elem_classes=["config-field-grid"]):
                    avg_input_tokens = gr.Number(label="平均输入", value=defaults[5], precision=0)
                    avg_output_tokens = gr.Number(label="平均输出", value=defaults[6], precision=0)
                with gr.Row(elem_classes=["config-field-grid"]):
                    p95_input_tokens = gr.Number(label="P95 输入", value=defaults[7], precision=0)
                    p95_output_tokens = gr.Number(label="P95 输出", value=defaults[8], precision=0)
                with gr.Row(elem_classes=["config-field-grid"]):
                    ttft_avg_sec = gr.Number(label="平均 TTFT (s)", value=defaults[9], precision=2)
                    ttft_p95_sec = gr.Number(label="P95 TTFT (s)", value=defaults[10], precision=2)
                with gr.Row(elem_classes=["config-field-grid"]):
                    e2e_avg_sec = gr.Number(label="平均 E2E (s)", value=defaults[11], precision=2)
                    e2e_p95_sec = gr.Number(label="P95 E2E (s)", value=defaults[12], precision=2)

            with gr.Row(elem_classes=["button-row"]):
                reset_button = gr.Button("恢复默认", variant="secondary")

    return SidebarComponents(
        model_dropdown=model_dropdown,
        precision_dropdown=precision_dropdown,
        gpu_preset_key=gpu_preset_key,
        lambda_avg_qps=lambda_avg_qps,
        lambda_peak_qps=lambda_peak_qps,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        p95_input_tokens=p95_input_tokens,
        p95_output_tokens=p95_output_tokens,
        ttft_avg_sec=ttft_avg_sec,
        ttft_p95_sec=ttft_p95_sec,
        e2e_avg_sec=e2e_avg_sec,
        e2e_p95_sec=e2e_p95_sec,
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
