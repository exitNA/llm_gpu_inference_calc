from __future__ import annotations

from typing import Any

from gpu_sizing_core.service import evaluate_single_model, format_result_text
from ui.views import (
    build_final_summary_html,
    build_kv_detail_rows,
    build_memory_analysis_html,
    build_overview_html,
    build_request_detail_rows,
    build_throughput_analysis_html,
)

from .config import build_configs, build_default_configs, default_component_values


def build_default_result() -> dict[str, Any]:
    model, traffic, gpu, runtime = build_default_configs()
    return evaluate_single_model(
        model=model,
        traffic=traffic,
        gpu=gpu,
        runtime=runtime,
    )


def build_default_result_text() -> str:
    return format_result_text(build_default_result())


def run_analysis(
    *raw_inputs: Any,
) -> tuple[str, str, str, str, list[list[str]], list[list[str]], str, dict[str, Any]]:
    model, traffic, gpu, runtime = build_configs(*raw_inputs)
    result = evaluate_single_model(
        model=model,
        traffic=traffic,
        gpu=gpu,
        runtime=runtime,
    )
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
    defaults = default_component_values()
    default_result = build_default_result()
    return (
        *defaults,
        build_overview_html(default_result),
        build_memory_analysis_html(default_result),
        build_throughput_analysis_html(default_result),
        build_final_summary_html(default_result),
        build_request_detail_rows(default_result),
        build_kv_detail_rows(default_result),
        default_result["calculation_process_text"],
        default_result,
    )
