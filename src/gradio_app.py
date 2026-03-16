from html import escape
from typing import Any
import warnings

from gpu_sizing import (
    GPUConfig,
    HAConfig,
    ModelConfig,
    RequestShape,
    RuntimeConfig,
    TrafficConfig,
    evaluate_single_model_with_ha,
    format_result_text,
)
from presets import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GPU_PRESET_KEY,
    TRAFFIC_PROFILE,
    build_default_traffic_targets,
    get_default_gpu_key,
    get_default_model_key,
    get_gpu_choices,
    get_gpu_preset,
    get_model_choices,
    get_model_preset,
)
from ui_formatters import (
    APP_CSS,
    build_calculation_process_html,
    build_final_summary_html,
    build_kv_detail_rows,
    build_memory_analysis_html,
    build_overview_html,
    build_request_detail_rows,
    build_shape_name_html,
    build_throughput_analysis_html,
    build_traffic_profile_header_html,
)

APP_TITLE = "GPU Sizing Studio"
HA_MODE_LABELS = {
    "不启用": "none",
    "N+1 备份": "n_plus_1",
    "主备": "active_standby",
    "双活": "active_active"
}
HA_MODE_CHOICES = list(HA_MODE_LABELS.keys())
REQUEST_TABLE_HEADERS = ["请求类型", "占比", "输入", "输出", "总长度"]
KV_TABLE_HEADERS = ["请求类型", "总长度", "单请求 KV Cache(GB)"]

# Default request shape values from preset
_DEFAULT_SHAPES = TRAFFIC_PROFILE.request_shapes


def default_component_values() -> tuple[Any, ...]:
    default_prefill_tps_total, default_decode_tps_total = build_default_traffic_targets(
        DEFAULT_CONCURRENCY,
    )
    return (
        get_default_model_key(),
        "bf16",  # Default precision
        get_default_gpu_key(),
        DEFAULT_CONCURRENCY,
        default_prefill_tps_total,
        default_decode_tps_total,
        "不启用",
        0,
        25,
        True,
        15,  # default weight_overhead_ratio (%)
        8,   # default runtime_overhead_ratio (%)
        2,   # default runtime_overhead_gb
        90,  # default usable_vram_ratio (%)
    )

def default_shape_values() -> tuple[Any, ...]:
    """Return flat tuple of all request shape fields for UI default values."""
    vals: list[Any] = []
    for s in _DEFAULT_SHAPES:
        vals.extend([s.ratio, s.avg_input_tokens, s.avg_output_tokens])
    return tuple(vals)


def to_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"{field_name} 不能为空")
    int_value = int(float(value))
    if int_value != float(value):
        raise ValueError(f"{field_name} 必须是整数")
    return int_value


def to_float(value: Any, field_name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{field_name} 不能为空")
    return float(value)


def ensure_positive(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} 必须大于 0")


def ensure_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} 不能小于 0")


def _build_traffic_config_from_inputs(
    concurrency: int,
    target_prefill_tps_total: float,
    target_decode_tps_total: float,
    shape_ratios: list[float],
    shape_inputs: list[int],
    shape_outputs: list[int],
) -> TrafficConfig:
    """Build TrafficConfig from user-editable request shape values."""
    shapes = []
    for i, base_shape in enumerate(_DEFAULT_SHAPES):
        shapes.append(
            RequestShape(
                name=base_shape.name,
                ratio=shape_ratios[i],
                avg_input_tokens=shape_inputs[i],
                avg_output_tokens=shape_outputs[i],
            )
        )
    return TrafficConfig(
        concurrency=concurrency,
        target_decode_tps_total=target_decode_tps_total,
        batch_size_per_request=TRAFFIC_PROFILE.batch_size_per_request,
        target_prefill_tps_total=target_prefill_tps_total,
        request_shapes=shapes,
    )


def build_configs(
    model_dropdown: Any,
    precision_override: Any,
    gpu_preset_key: Any,
    concurrency: Any,
    target_prefill_tps_total: Any,
    target_decode_tps_total: Any,
    ha_mode: Any,
    replica_count: Any,
    failover_reserve_ratio: Any,
    use_p95_for_memory_sizing: Any,
    weight_overhead_ratio: Any,
    runtime_overhead_ratio: Any,
    runtime_overhead_gb: Any,
    usable_vram_ratio: Any,
    *shape_values: Any,
):
    concurrency_value = to_int(concurrency, "并发量")
    target_prefill_tps_total_value = to_float(target_prefill_tps_total, "目标 Prefill TPS")
    target_decode_tps_total_value = to_float(target_decode_tps_total, "目标 Decode TPS")
    
    ha_mode_val = HA_MODE_LABELS.get(str(ha_mode), "none")
    if ha_mode_val == "none":
        replica_count_value = 0
    else:
        replica_count_value = to_int(replica_count, "副本数")
        if ha_mode_val == "active_active" or ha_mode_val == "active_standby":
            ensure_positive(replica_count_value, "副本数")

    failover_ratio = to_float(failover_reserve_ratio, "故障冗余比例") / 100.0
    w_overhead_ratio_val = to_float(weight_overhead_ratio, "模型参数额外开销") / 100.0
    rt_overhead_ratio_val = to_float(runtime_overhead_ratio, "运行时显存占比") / 100.0
    rt_overhead_gb_val = to_float(runtime_overhead_gb, "基础运行时显存")
    u_vram_ratio_val = to_float(usable_vram_ratio, "安全可用显存比例") / 100.0

    ensure_positive(concurrency_value, "并发量")
    ensure_positive(target_prefill_tps_total_value, "目标 Prefill TPS")
    ensure_positive(target_decode_tps_total_value, "目标 Decode TPS")
    ensure_non_negative(failover_ratio, "故障冗余比例")
    ensure_non_negative(w_overhead_ratio_val, "模型参数额外开销")
    ensure_non_negative(rt_overhead_ratio_val, "运行时显存占比")
    ensure_non_negative(rt_overhead_gb_val, "基础运行时显存")
    if u_vram_ratio_val <= 0 or u_vram_ratio_val > 1.0:
        raise ValueError("安全可用显存比例必须在 0 到 1 之间")

    base_model = get_model_preset(str(model_dropdown)).config
    base_gpu = get_gpu_preset(str(gpu_preset_key)).config
    
    # 覆盖精度、推导 KV 精度、以及高级参数
    target_prec = str(precision_override).lower()
    target_kv = "fp8" if target_prec == "fp8" else "fp16"
    
    runtime = RuntimeConfig(
        precision=target_prec,
        kv_cache_dtype=target_kv,
        weight_overhead_ratio=w_overhead_ratio_val,
        runtime_overhead_ratio=rt_overhead_ratio_val,
        runtime_overhead_gb=rt_overhead_gb_val,
        usable_vram_ratio=u_vram_ratio_val,
        # fallback to defaults for efficiencies for now, as they are not exposed in UI
        decode_efficiency=0.40,
        prefill_efficiency=0.55,
        compute_efficiency=0.60,
        prefill_memory_reuse_factor=24.0,
    )

    # Parse shape values: 3 values per shape (ratio, input, output)
    n_shapes = len(_DEFAULT_SHAPES)
    if len(shape_values) >= n_shapes * 3:
        shape_ratios = [to_float(shape_values[i * 3], f"占比 {i+1}") for i in range(n_shapes)]
        shape_inputs = [to_int(shape_values[i * 3 + 1], f"输入 {i+1}") for i in range(n_shapes)]
        shape_outputs = [to_int(shape_values[i * 3 + 2], f"输出 {i+1}") for i in range(n_shapes)]
    else:
        # Fallback to default shapes
        shape_ratios = [s.ratio for s in _DEFAULT_SHAPES]
        shape_inputs = [s.avg_input_tokens for s in _DEFAULT_SHAPES]
        shape_outputs = [s.avg_output_tokens for s in _DEFAULT_SHAPES]

    traffic = _build_traffic_config_from_inputs(
        concurrency_value,
        target_prefill_tps_total_value,
        target_decode_tps_total_value,
        shape_ratios,
        shape_inputs,
        shape_outputs,
    )
    ha = HAConfig(
        ha_mode=ha_mode_val,
        replica_count=replica_count_value,
        failover_reserve_ratio=failover_ratio,
    )

    if ha.ha_mode not in HA_MODE_LABELS.values():
        raise ValueError(f"不支持的高可用模式: {ha.ha_mode}")

    return base_model, traffic, base_gpu, ha, runtime, bool(use_p95_for_memory_sizing)


def build_default_configs():
    values = default_component_values()
    shape_vals = default_shape_values()
    return build_configs(*values, *shape_vals)


def build_default_result_text() -> str:
    result = build_default_result()
    return format_result_text(result)


def build_default_result() -> dict[str, Any]:
    model, traffic, gpu, ha, runtime, use_p95 = build_default_configs()
    return evaluate_single_model_with_ha(
        model=model,
        traffic=traffic,
        gpu=gpu,
        ha=ha,
        runtime=runtime,
        use_p95_for_memory_sizing=use_p95,
    )


def run_analysis(
    *raw_inputs: Any,
) -> tuple[str, str, str, str, list[list[str]], list[list[str]], str, dict[str, Any]]:
    model, traffic, gpu, ha, runtime, use_p95_for_memory_sizing = build_configs(*raw_inputs)
    result = evaluate_single_model_with_ha(
        model=model,
        traffic=traffic,
        gpu=gpu,
        ha=ha,
        runtime=runtime,
        use_p95_for_memory_sizing=use_p95_for_memory_sizing,
    )
    return (
        build_overview_html(result),
        build_memory_analysis_html(result),
        build_throughput_analysis_html(result),
        build_final_summary_html(result),
        build_request_detail_rows(result),
        build_kv_detail_rows(result),
        build_calculation_process_html(result),
        result,
    )


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
        
    new_value = current_precision if current_precision in choices else choices[-1]
    return gr.Dropdown(choices=choices, value=new_value)


def gradio_dropdown_update(choices: list[tuple[str, str]], value: str):
    gr = import_gradio()
    return gr.Dropdown(choices=choices, value=value)


def build_config_panel_intro_html() -> str:
    return """
    <div class="config-panel-hero">
      <p class="config-panel-eyebrow">Control Panel</p>
      <div class="config-panel-title-row">
        <h2>参数配置</h2>
        <span class="config-panel-status">实时计算</span>
      </div>
      <p class="config-panel-copy">调整左侧参数，右侧结果会自动更新。</p>
    </div>
    """


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


def reset_all():
    defaults = default_component_values()
    shape_defaults = default_shape_values()
    default_result = build_default_result()
    return (
        *defaults,
        *shape_defaults,
        build_overview_html(default_result),
        build_memory_analysis_html(default_result),
        build_throughput_analysis_html(default_result),
        build_final_summary_html(default_result),
        build_request_detail_rows(default_result),
        build_kv_detail_rows(default_result),
        build_calculation_process_html(default_result),
        default_result,
    )


def build_app():
    gr = import_gradio()
    default_values = default_component_values()
    shape_vals = default_shape_values()
    default_result = build_default_result()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The parameters have been moved from the Blocks constructor to the launch\(\) method in Gradio 6\.0: theme, css\..*",
            category=UserWarning,
        )
        blocks = gr.Blocks(title=APP_TITLE, theme=build_theme(gr), css=APP_CSS)

    with blocks as demo:
        with gr.Row(elem_classes=["workspace-main"]):
            # ── Left: Configuration Panel ──
            with gr.Column(scale=4, min_width=420, elem_classes=["workspace-sidebar"]):
                with gr.Group(elem_classes=["config-panel"]):
                    gr.HTML(build_config_panel_intro_html())

                    with gr.Group(elem_classes=["config-section", "config-section-model"]):
                        gr.HTML(build_config_section_header_html("🧠", "模型配置", "模型选择与运行时冗余预算"))
                        model_dropdown = gr.Dropdown(
                            label="选择模型",
                            choices=get_model_choices(),
                            value=default_values[0],
                        )
                        with gr.Group(elem_classes=["config-slider-stack"]):
                            weight_overhead_ratio = gr.Slider(
                                minimum=0, maximum=100, step=1,
                                label="权重冗余 (%)",
                                value=default_values[10],
                                info="模型加载时权重膨胀的预留空间",
                                elem_classes=["compact-slider"],
                            )
                            runtime_overhead_ratio = gr.Slider(
                                minimum=0, maximum=100, step=1,
                                label="运行时开销 (%)",
                                value=default_values[11],
                                info="激活值和 CUDA 上下文占权重的比例",
                                elem_classes=["compact-slider"],
                            )
                            runtime_overhead_gb = gr.Slider(
                                minimum=0, maximum=10, step=1,
                                label="运行时保底 (GB)",
                                value=default_values[12],
                                info="最低保障的运行时开销绝对值",
                                elem_classes=["compact-slider"],
                            )

                    with gr.Group(elem_classes=["config-section", "config-section-gpu"]):
                        gr.HTML(build_config_section_header_html("🖥️", "显卡配置", "GPU 规格、精度与安全水位线"))
                        gpu_preset_key = gr.Dropdown(
                            label="选择显卡",
                            choices=get_gpu_choices(),
                            value=default_values[2],
                        )
                        precision_dropdown = gr.Radio(
                            label="推理精度",
                            choices=["fp8", "fp16", "bf16"],
                            value=default_values[1],
                            elem_classes=["compact-radio"],
                        )
                        usable_vram_ratio = gr.Slider(
                            minimum=50, maximum=100, step=1,
                            label="安全水位线 (%)",
                            value=default_values[13],
                            info="避免 OOM 的显存水位线上限",
                            elem_classes=["compact-slider"],
                        )

                    with gr.Group(elem_classes=["config-section", "config-section-traffic"]):
                        gr.HTML(build_config_section_header_html("🚦", "流量配置", "并发与集群吞吐目标"))
                        with gr.Row(elem_classes=["config-field-grid"]):
                            with gr.Column(min_width=96):
                                concurrency = gr.Number(
                                    label="并发量",
                                    value=default_values[3],
                                    precision=0,
                                )
                            with gr.Column(min_width=144):
                                target_prefill_tps_total = gr.Number(
                                    label="Prefill TPS",
                                    value=default_values[4],
                                    precision=2,
                                    info="系统总 prefill 吞吐目标（tok/s）",
                                )
                            with gr.Column(min_width=144):
                                target_decode_tps_total = gr.Number(
                                    label="Decode TPS",
                                    value=default_values[5],
                                    precision=2,
                                    info="系统总 decode 吞吐目标（tok/s）",
                                )
                        use_p95_for_memory_sizing = gr.Checkbox(
                            label="显存 sizing 使用 P95 序列长度",
                            value=default_values[9],
                            info="预留 95% 请求不 OOM 的极限长度显存；不勾选则按平均长度预算",
                            elem_classes=["config-toggle"],
                        )

                        with gr.Accordion("业务请求画像", open=False, elem_classes=["custom-accordion", "profile-accordion"]):
                            gr.HTML(build_traffic_profile_header_html())
                            shape_components: list = []
                            for idx, s in enumerate(_DEFAULT_SHAPES):
                                sv_offset = idx * 3
                                gr.HTML(build_shape_name_html(s.name))
                                with gr.Row(elem_classes=["shape-grid"]):
                                    ratio_input = gr.Number(
                                        label="占比",
                                        value=shape_vals[sv_offset],
                                        precision=2,
                                        minimum=0.0,
                                        maximum=1.0,
                                    )
                                    input_tokens = gr.Number(
                                        label="输入 tokens",
                                        value=shape_vals[sv_offset + 1],
                                        precision=0,
                                    )
                                    output_tokens = gr.Number(
                                        label="输出 tokens",
                                        value=shape_vals[sv_offset + 2],
                                        precision=0,
                                    )
                                shape_components.extend([ratio_input, input_tokens, output_tokens])

                    with gr.Group(elem_classes=["config-section", "config-section-ha"]):
                        gr.HTML(build_config_section_header_html("🛡️", "高可用", "副本策略与故障冗余"))
                        ha_mode = gr.Dropdown(
                            label="HA 模式",
                            choices=HA_MODE_CHOICES,
                            value=default_values[6],
                        )
                        with gr.Row(elem_classes=["config-ha-grid"]):
                            with gr.Column(min_width=110):
                                replica_count = gr.Number(
                                    label="副本数",
                                    value=default_values[7],
                                    precision=0,
                                    visible=(default_values[6] != "不启用"),
                                )
                            with gr.Column(min_width=200):
                                failover_reserve_ratio = gr.Slider(
                                    minimum=0, maximum=100, step=1,
                                    label="故障冗余比例 (%)",
                                    value=default_values[8],
                                    info="计划外故障时预留的额外容量占比",
                                    visible=(default_values[6] != "不启用"),
                                    elem_classes=["compact-slider"],
                                )

                    with gr.Row(elem_classes=["button-row"]):
                        reset_button = gr.Button("恢复默认", variant="secondary")

            # ── Right: Four-Step Narrative ──
            with gr.Column(scale=8, min_width=860, elem_classes=["workspace-content"]):
                # Step 0: Overview / Conclusion Hero
                overview_html = gr.HTML(value=build_overview_html(default_result))
                # Step 1: Memory Analysis
                memory_html = gr.HTML(value=build_memory_analysis_html(default_result))
                # Step 2: Throughput Analysis
                throughput_html = gr.HTML(value=build_throughput_analysis_html(default_result))
                # Step 3: Final Procurement
                final_html = gr.HTML(value=build_final_summary_html(default_result))

                # Debug / Detail Accordion at the bottom
                with gr.Accordion("📋 详细数据与原始 JSON", open=False, elem_classes=["custom-accordion"]):
                    with gr.Tabs():
                        with gr.Tab("请求与 KV"):
                            request_table = gr.Dataframe(
                                headers=REQUEST_TABLE_HEADERS,
                                datatype=["str", "str", "str", "str", "str"],
                                interactive=False,
                                label="请求分布明细",
                                value=build_request_detail_rows(default_result),
                            )
                            kv_table = gr.Dataframe(
                                headers=KV_TABLE_HEADERS,
                                datatype=["str", "str", "str"],
                                interactive=False,
                                label="KV Cache 明细",
                                value=build_kv_detail_rows(default_result),
                            )
                        with gr.Tab("计算过程"):
                            calculation_markdown = gr.HTML(
                                value=build_calculation_process_html(default_result),
                            )
                        with gr.Tab("原始 JSON"):
                            raw_json = gr.JSON(label="原始结果", value=default_result)

        # ── All inputs (base config + request shapes) ──
        base_inputs = [
            model_dropdown,
            precision_dropdown,
            gpu_preset_key,
            concurrency,
            target_prefill_tps_total,
            target_decode_tps_total,
            ha_mode,
            replica_count,
            failover_reserve_ratio,
            use_p95_for_memory_sizing,
            weight_overhead_ratio,
            runtime_overhead_ratio,
            runtime_overhead_gb,
            usable_vram_ratio,
        ]
        all_inputs = base_inputs + shape_components

        outputs = [
            overview_html,
            memory_html,
            throughput_html,
            final_html,
            request_table,
            kv_table,
            calculation_markdown,
            raw_json,
        ]

        def run_analysis_safe(*values: Any):
            try:
                return run_analysis(*values)
            except (ValueError, KeyError):
                import gradio as _gr
                raise _gr.Error("参数无效，请检查输入")

        # ── Cascading & Interactive Elements ──
        gpu_preset_key.change(
            fn=update_precision_choices,
            inputs=[gpu_preset_key, precision_dropdown],
            outputs=precision_dropdown,
        )

        def _update_ha_visibles(m):
            import gradio as gr
            is_enabled = m != "不启用"
            return (
                gr.update(visible=is_enabled),
                gr.update(visible=is_enabled),
            )

        ha_mode.change(
            fn=_update_ha_visibles,
            inputs=ha_mode,
            outputs=[replica_count, failover_reserve_ratio]
        )

        # ── Auto-calculate: wire .change() on every input ──
        auto_calc_inputs = [
            model_dropdown, precision_dropdown, gpu_preset_key,
            concurrency, target_prefill_tps_total, target_decode_tps_total,
            ha_mode, replica_count, failover_reserve_ratio,
            use_p95_for_memory_sizing, weight_overhead_ratio,
            runtime_overhead_ratio, runtime_overhead_gb, usable_vram_ratio,
        ] + shape_components

        for component in auto_calc_inputs:
            component.change(
                fn=run_analysis_safe,
                inputs=all_inputs,
                outputs=outputs,
            )

        reset_outputs = [
            model_dropdown,
            precision_dropdown,
            gpu_preset_key,
            concurrency,
            target_prefill_tps_total,
            target_decode_tps_total,
            ha_mode,
            replica_count,
            failover_reserve_ratio,
            use_p95_for_memory_sizing,
            weight_overhead_ratio,
            runtime_overhead_ratio,
            runtime_overhead_gb,
            usable_vram_ratio,
            *shape_components,
            overview_html,
            memory_html,
            throughput_html,
            final_html,
            request_table,
            kv_table,
            calculation_markdown,
            raw_json,
        ]
        reset_button.click(fn=reset_all, outputs=reset_outputs)

        # ── Initial load ──
        demo.load(fn=run_analysis, inputs=all_inputs, outputs=outputs)

    return demo


try:
    demo = build_app()
except RuntimeError:
    demo = None


def launch_app(host: str, port: int, share: bool) -> None:
    if demo is None:
        raise RuntimeError("未安装 gradio，请先执行 `uv sync`。")
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch_app(host="0.0.0.0", port=7860, share=False)
