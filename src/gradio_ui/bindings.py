from __future__ import annotations

from typing import Any

from .analysis import reset_all, run_analysis
from .runtime import update_precision_choices
from .sections import ResultComponents, SidebarComponents


def bind_events(*, sidebar: SidebarComponents, results: ResultComponents, demo: Any) -> None:
    inputs = sidebar.analysis_inputs()
    outputs = results.analysis_outputs()

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
