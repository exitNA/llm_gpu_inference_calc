from __future__ import annotations

import sys
from typing import Any

from gpu_sizing_core.service import evaluate_single_model
from ui.views import (
    build_final_summary_html,
    build_kv_detail_rows,
    build_memory_analysis_html,
    build_overview_html,
    build_request_detail_rows,
    build_throughput_analysis_html,
)

from .config import UIInputs


def _print_console_calc_trace(context: str, result: dict[str, Any]) -> None:
    print(
        f"\n---\n[calc-trace] {result['model_name']} | {result['gpu_name']} | {result['business_gpu_count']} 卡",
        file=sys.stderr,
        flush=True,
    )
    print(result["calculation_process_text"], file=sys.stderr, flush=True)


def build_default_result(*, emit_console_trace: bool = True) -> dict[str, Any]:
    model, traffic, gpu, runtime = UIInputs.default().build_configs()
    result = evaluate_single_model(
        model=model,
        traffic=traffic,
        gpu=gpu,
        runtime=runtime,
    )
    if emit_console_trace:
        _print_console_calc_trace("default-result", result)
    return result
def run_analysis(
    *raw_inputs: Any,
) -> tuple[str, str, str, str, list[list[str]], list[list[str]], str, dict[str, Any]]:
    input_values = UIInputs.from_raw_inputs(raw_inputs)
    model, traffic, gpu, runtime = input_values.build_configs()
    result = evaluate_single_model(
        model=model,
        traffic=traffic,
        gpu=gpu,
        runtime=runtime,
    )
    _print_console_calc_trace("interactive-analysis", result)
    return (
        build_overview_html(result),
        build_memory_analysis_html(result),
        build_throughput_analysis_html(result),
        build_final_summary_html(result),
        build_request_detail_rows(result),
        build_kv_detail_rows(result),
        result["calculation_process_text"],
        result,
    )


def reset_all() -> tuple[Any, ...]:
    defaults = UIInputs.default()
    default_result = build_default_result()
    return (
        *defaults.gradio_values(),
        build_overview_html(default_result),
        build_memory_analysis_html(default_result),
        build_throughput_analysis_html(default_result),
        build_final_summary_html(default_result),
        build_request_detail_rows(default_result),
        build_kv_detail_rows(default_result),
        default_result["calculation_process_text"],
        default_result,
    )
