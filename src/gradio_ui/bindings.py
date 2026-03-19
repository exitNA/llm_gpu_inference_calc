from __future__ import annotations

from typing import Any

from .analysis import reset_all, run_analysis
from .runtime import update_precision_choices
from .sections import ResultComponents, SidebarComponents


def bind_events(*, sidebar: SidebarComponents, results: ResultComponents, demo: Any) -> None:
    inputs = [
        sidebar.model_dropdown,
        sidebar.precision_dropdown,
        sidebar.gpu_preset_key,
        sidebar.qps_estimation_mode,
        sidebar.lambda_peak_qps,
        sidebar.daily_request_count,
        sidebar.qps_burst_factor_pct,
        sidebar.poisson_time_window_sec,
        sidebar.poisson_qps_quantile_pct,
        sidebar.p95_input_tokens,
        sidebar.p95_output_tokens,
        sidebar.ttft_p95_sec,
        sidebar.e2e_p95_sec,
        sidebar.concurrency_estimation_mode,
        sidebar.direct_peak_concurrency,
        sidebar.concurrency_safety_factor_pct,
        sidebar.weight_overhead_ratio,
        sidebar.runtime_overhead_ratio,
        sidebar.usable_vram_ratio,
        sidebar.bandwidth_efficiency,
        sidebar.compute_efficiency,
    ]
    outputs = [
        results.overview_html,
        results.memory_html,
        results.throughput_html,
        results.final_html,
        results.request_table,
        results.kv_table,
        results.calc_text,
        results.raw_json,
    ]

    def run_analysis_safe(*values: Any):
        try:
            return run_analysis(*values)
        except (ValueError, KeyError):
            import gradio as _gr

            raise _gr.Error("参数无效，请检查输入")

    sidebar.gpu_preset_key.change(
        fn=update_precision_choices,
        inputs=[sidebar.gpu_preset_key, sidebar.precision_dropdown],
        outputs=sidebar.precision_dropdown,
    )

    for component in inputs:
        component.change(
            fn=run_analysis_safe,
            inputs=inputs,
            outputs=outputs,
        )

    sidebar.reset_button.click(fn=reset_all, outputs=[*inputs, *outputs])
    demo.load(fn=run_analysis, inputs=inputs, outputs=outputs)
