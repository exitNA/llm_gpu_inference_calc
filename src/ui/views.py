from __future__ import annotations

from html import escape
from typing import Any

from .common import (
    fmt_compact,
    fmt_value,
    render_calc_accordion,
    render_math_text,
)


def _format_ratio_percent(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def _format_memory_basis(value_gib: float) -> str:
    if value_gib >= 1:
        return f"{value_gib:.1f} GiB"
    return f"{value_gib * 1024:.1f} MiB"


def _precision_unit_label(precision: str) -> str:
    mapping = {
        "fp32": "4 bytes",
        "fp16": "2 bytes",
        "bf16": "2 bytes",
        "fp8": "1 byte",
        "int8": "1 byte",
        "int4": "0.5 byte",
    }
    return mapping.get(precision.lower(), precision)


def _find_calc_step(section: dict[str, Any] | None, label: str) -> dict[str, Any] | None:
    if not section:
        return None
    for step in section.get("steps", []):
        if step.get("label") == label:
            return step
    return None


def _find_calc_step_in_sections(sections: list[dict[str, Any] | None], label: str) -> dict[str, Any] | None:
    for section in sections:
        step = _find_calc_step(section, label)
        if step:
            return step
    return None


def _render_tp_basis_item(step: dict[str, Any] | None) -> str:
    if not step:
        return ""

    note_html = ""
    if step.get("note"):
        note_html = f"""
        <div class="tp-basis-note">
          <span class="tp-basis-kicker">说明</span>
          <span class="tp-basis-text">{render_math_text(step["note"])}</span>
        </div>
        """

    return f"""
    <div class="tp-basis-item">
      <div class="tp-basis-topline">
        <span class="tp-basis-label">{render_math_text(step["label"])}</span>
        <span class="tp-basis-result">{escape(step["result"])}</span>
      </div>
      <div class="tp-basis-line">
        <span class="tp-basis-kicker">公式</span>
        <span class="tp-basis-text">{render_math_text(step["formula"])}</span>
      </div>
      <div class="tp-basis-line">
        <span class="tp-basis-kicker">代入</span>
        <span class="tp-basis-text">{render_math_text(step["substitution"])}</span>
      </div>
      {note_html}
    </div>
    """


def _render_tp_basis_block(sections: list[dict[str, Any] | None], labels: list[str]) -> str:
    items_html = "".join(_render_tp_basis_item(_find_calc_step_in_sections(sections, label)) for label in labels)
    if not items_html:
        return ""
    return f"""
    <div class="tp-basis-block">
      <div class="tp-basis-heading">数据计算依据</div>
      {items_html}
    </div>
    """


def _format_stage_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "-"
    return f"{numerator / denominator:.0%}"


def _format_signed_tps_gap(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{fmt_compact(abs(value))} tok/s"


def _render_tp_proof_card(step: dict[str, Any] | None) -> str:
    if not step:
        return ""
    human_formula = step.get("formula_note") or step["formula"]
    return f"""
    <div class="tp-proof-card">
      <div class="tp-proof-topline">
        <span class="tp-proof-label">{render_math_text(step["label"])}</span>
        <span class="tp-proof-value">{escape(step["result"])}</span>
      </div>
      <div class="tp-proof-formula">{render_math_text(human_formula)}</div>
      <div class="tp-proof-sub">{render_math_text(step["substitution"])}</div>
    </div>
    """


def _render_tp_stage_shell(
    *,
    stage_key: str,
    stage_title: str,
    badge_text: str,
    badge_class: str,
    result: dict[str, Any],
    sections: list[dict[str, Any] | None],
    workload_label: str,
    service_label: str,
    reference_label: str,
    latency_label: str,
    workload_key: str,
    cluster_capacity_key: str,
    reference_count_key: str,
    necessity_ok_key: str,
    necessity_gap_key: str,
    latency_ok_key: str,
) -> str:
    workload = result.get(workload_key)
    cluster_capacity = result.get(cluster_capacity_key)
    reference_count = result.get(reference_count_key)
    baseline_count = result["business_gpu_count"]
    necessity_ok = bool(result[necessity_ok_key])
    latency_ok = bool(result[latency_ok_key])
    gap_gpus = result[necessity_gap_key]
    capacity_ratio = _format_stage_ratio(cluster_capacity, workload)
    capacity_gap_text = _format_signed_tps_gap(None if workload is None or cluster_capacity is None else cluster_capacity - workload)
    latency_text = "满足" if latency_ok else "不满足"
    necessity_text = "满足" if necessity_ok else "不满足"
    status_class = "is-ok" if necessity_ok else "is-risk"

    if necessity_ok:
        summary = (
            f"当前建议采购 {baseline_count} 张卡时，该阶段的理论总服务率约为 "
            f"{fmt_compact(cluster_capacity)} tok/s，已经覆盖目标工作量 {fmt_compact(workload)} tok/s。"
        )
    else:
        summary = (
            f"当前建议采购 {baseline_count} 张卡时，该阶段的理论总服务率约为 "
            f"{fmt_compact(cluster_capacity)} tok/s，仍低于目标工作量 {fmt_compact(workload)} tok/s。"
        )

    summary += " 这里的卡数只作必要条件参考，不会直接抬高采购主结果。"

    proof_html = "".join(
        [
            _render_tp_proof_card(_find_calc_step_in_sections(sections, workload_label)),
            _render_tp_proof_card(_find_calc_step_in_sections(sections, service_label)),
            _render_tp_proof_card(_find_calc_step_in_sections(sections, reference_label)),
            _render_tp_proof_card(_find_calc_step_in_sections(sections, latency_label)),
        ]
    )

    return f"""
    <div class="throughput-shell throughput-shell-{stage_key}">
      <div class="throughput-spotlight throughput-spotlight-{stage_key}">
        <div class="tp-card-header">
          <span class="tp-card-title">{stage_title}</span>
          <span class="tp-badge {badge_class}">{badge_text}</span>
        </div>
        <span class="memory-result-kicker">理论参考卡数</span>
        <div class="memory-result-main">
          <span class="memory-result-value">{reference_count}</span>
          <span class="memory-result-unit">GPUs</span>
        </div>
        <span class="memory-result-caption">只用于必要条件参考，不直接并入采购主结果</span>
        <div class="memory-result-proof">
          <span class="memory-result-formula">当前显存基线 {baseline_count} 张，{'已覆盖' if necessity_ok else f'仍差 {gap_gpus} 张'}</span>
          <span class="memory-result-note">目标工作量与单卡理论服务率反推得到该阶段的参考卡数。</span>
        </div>
        <div class="memory-result-stats">
          <div class="memory-result-stat">
            <span class="memory-result-stat-label">当前显存基线</span>
            <span class="memory-result-stat-value">{baseline_count} 卡</span>
          </div>
          <div class="memory-result-stat">
            <span class="memory-result-stat-label">当前总理论服务率</span>
            <span class="memory-result-stat-value">{fmt_compact(cluster_capacity)} tok/s</span>
          </div>
          <div class="memory-result-stat">
            <span class="memory-result-stat-label">相对目标</span>
            <span class="memory-result-stat-value">{capacity_ratio} · {capacity_gap_text}</span>
          </div>
          <div class="memory-result-stat">
            <span class="memory-result-stat-label">时延必要条件</span>
            <span class="memory-result-stat-value">{latency_text}</span>
          </div>
        </div>
      </div>
      <div class="throughput-cause-cluster">
        <div class="memory-cause-heading">
          <span class="memory-cause-kicker">{stage_title} 必要条件结论</span>
          <span class="memory-cause-summary">{summary}</span>
        </div>
        <div class="throughput-status-strip {status_class}">
          <div class="throughput-status-item">
            <span class="throughput-status-label">必要条件结果</span>
            <span class="throughput-status-value">{necessity_text}</span>
          </div>
          <div class="throughput-status-item">
            <span class="throughput-status-label">目标工作量</span>
            <span class="throughput-status-value">{fmt_compact(workload)} tok/s</span>
          </div>
          <div class="throughput-status-item">
            <span class="throughput-status-label">理论总服务率</span>
            <span class="throughput-status-value">{fmt_compact(cluster_capacity)} tok/s</span>
          </div>
        </div>
        <div class="throughput-proof-grid">
          {proof_html}
        </div>
      </div>
    </div>
    """

def build_request_detail_rows(result: dict[str, Any]) -> list[list[str]]:
    rows = []
    for item in result.get("request_profile_rows", []):
        rows.append(
            [
                item["name"],
                fmt_value(item["qps"], 2, " req/s"),
                str(item["input_tokens"]),
                str(item["output_tokens"]),
                str(item["total_tokens"]),
                fmt_value(item["ttft_target_sec"], 2, " s"),
                fmt_value(item["e2e_target_sec"], 2, " s"),
            ]
        )
    return rows


def build_kv_detail_rows(result: dict[str, Any]) -> list[list[str]]:
    rows = []
    for item in result.get("kv_profile_rows", []):
        rows.append(
            [
                item["name"],
                str(item["seq_len_total"]),
                fmt_value(item["kv_cache_gib_per_request"], 2, " GiB"),
                "-" if item["max_concurrency_by_memory"] is None else str(item["max_concurrency_by_memory"]),
            ]
        )
    return rows

def build_overview_html(result: dict[str, Any]) -> str:
    dominant_constraints = list(result.get("dominant_constraints", [])) or ["显存"]
    dominant = " · ".join(dominant_constraints)
    traffic = result["traffic_config"]
    gpu = result["gpu_config"]
    runtime = result["runtime_config"]
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("业务目标折算与输入口径", calc_sections[0] if len(calc_sections) > 0 else None)
    return f"""
    <div class="result-card-unified">
      <div class="result-card-header">
        <div class="hero-gpu-count-box">
          <span class="hero-gpu-number">{result['business_gpu_count']}</span>
          <span class="hero-gpu-unit">GPUs</span>
        </div>
        <div class="result-card-title-area">
          <div class="hero-model-name-large">{escape(result['model_name'])}</div>
          <div class="hero-tag-row">
            <span class="hero-constraint constraint-throughput">{escape(dominant)}</span>
            <span class="hero-strategy-tag">主结果按显存给出；吞吐与时延只做必要条件检查</span>
          </div>
        </div>
      </div>
      <div class="result-card-body-grid">
        <div class="result-group-card">
          <div class="result-group-header">
            <span class="group-icon">🧠</span>
            <span class="result-group-title">模型与硬件</span>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">模型</span>
            <div class="result-item-content">
              <span class="result-item-value">{result['model_config'].num_params_billion:.0f}B</span>
              <span class="result-item-sub">{escape(result['arch_family'])} · {escape(result['attention_type'])}</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">GPU</span>
            <div class="result-item-content">
              <span class="result-item-value">{gpu.vram_gb:.0f}GB</span>
              <span class="result-item-sub">{escape(gpu.gpu_name)} ({runtime.precision}/{runtime.kv_cache_dtype})</span>
            </div>
          </div>
        </div>
        <div class="result-group-card">
          <div class="result-group-header">
            <span class="group-icon">🚦</span>
            <span class="result-group-title">业务目标</span>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">峰值 QPS</span>
            <div class="result-item-content">
              <span class="result-item-value">{result['lambda_peak_qps_effective']:.2f}</span>
              <span class="result-item-sub">{escape(result['qps_model_label'])}</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">长度</span>
            <div class="result-item-content">
              <span class="result-item-value">{traffic.p95_total_tokens}</span>
              <span class="result-item-sub">总长度 tokens</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">峰值在途</span>
            <div class="result-item-content">
              <span class="result-item-value">{fmt_compact(result['c_peak_budget'])}</span>
              <span class="result-item-sub">{escape(result['concurrency_model_label'])}</span>
            </div>
          </div>
        </div>
        <div class="result-group-card highlight-group">
          <div class="result-group-header">
            <span class="group-icon">⚡</span>
            <span class="result-group-title">结论与风险</span>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">主结果</span>
            <div class="result-item-content">
              <span class="result-item-value-compact">{result['business_gpu_count']} GPUs</span>
              <span class="result-item-sub">当前版本按显存约束给采购基线</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">时延风险</span>
            <div class="result-item-content">
              <span class="result-item-value">{escape(str(result['latency_risk_level']).upper())}</span>
              <span class="result-item-sub">{escape(result['latency_risk_note'])}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    {section_html}
    """


def build_memory_analysis_html(result: dict[str, Any]) -> str:
    model = result["model_config"]
    runtime = result["runtime_config"]
    weight_gib = result["weight_with_overhead_gib"]
    runtime_gib = result["runtime_overhead_gib"]
    total_memory_gib = result["memory_for_sizing_gib"]
    usable_vram_gib = result["usable_vram_gib_per_gpu"]
    gpu_count_by_memory = result["gpu_count_by_memory"]
    peak_cache_gib = max(total_memory_gib - weight_gib - runtime_gib, 0.0)
    per_request_cache_gib = result["p95_kv_gib_per_request"]
    c_peak_budget = result["c_peak_budget"]
    required_gpu_ratio = total_memory_gib / usable_vram_gib if usable_vram_gib > 0 else 0.0
    previous_gpu_count = max(gpu_count_by_memory - 1, 0)
    previous_capacity_gib = previous_gpu_count * usable_vram_gib
    current_capacity_gib = gpu_count_by_memory * usable_vram_gib
    free_capacity_gib = max(current_capacity_gib - total_memory_gib, 0.0)
    used_ratio = total_memory_gib / current_capacity_gib if current_capacity_gib > 0 else 0.0
    free_ratio = free_capacity_gib / current_capacity_gib if current_capacity_gib > 0 else 0.0
    weight_formula = (
        f"{model.num_params_billion:.0f}B × {_precision_unit_label(runtime.precision)} × "
        f"(1 + {_format_ratio_percent(runtime.weight_overhead_ratio)})"
    )
    runtime_formula = f"Mw × α_r = {weight_gib:.1f} × {_format_ratio_percent(runtime.runtime_overhead_ratio)}"
    cache_formula = f"{fmt_compact(c_peak_budget)} req × {_format_memory_basis(per_request_cache_gib)} / req"
    free_formula = f"{gpu_count_by_memory} × {usable_vram_gib:.1f} - {total_memory_gib:.1f}"

    segments = [
        ("权重", weight_gib, "bg-weight", "dot-weight", "模型权重", weight_formula),
        ("固定", runtime_gib, "bg-runtime", "dot-runtime", "框架常驻", runtime_formula),
        ("Cache", peak_cache_gib, "bg-kv", "dot-kv", "请求缓存", cache_formula),
        ("空闲", free_capacity_gib, "bg-free", "dot-free", "剩余余量", free_formula),
    ]
    disk_segments_html = "".join(
        f'<div class="mac-disk-segment {segment_class}" style="width:{(value / current_capacity_gib * 100) if current_capacity_gib > 0 else 0:.4f}%"></div>'
        for _label, value, segment_class, _dot_class, _detail, _formula in segments
        if value > 0
    )
    legend_html = "".join(
        f"""
        <div class="mac-disk-legend-item">
          <span class="mem-legend-dot {dot_class}"></span>
          <div class="mac-disk-legend-text">
            <div class="mac-disk-legend-top">
              <span class="mem-legend-name">{label}</span>
              <span class="mem-legend-value">{value:.1f} GiB</span>
            </div>
            <span class="mac-disk-detail-text">{detail}</span>
            <span class="mac-disk-formula-text">{formula}</span>
          </div>
        </div>
        """
        for label, value, _segment_class, dot_class, detail, formula in segments
        if value > 0
    )
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("显存约束计算细节", calc_sections[1] if len(calc_sections) > 1 else None)
    result_stats_html = f"""
          <div class="memory-result-stats">
            <div class="memory-result-stat">
              <span class="memory-result-stat-label">单卡有效显存</span>
              <span class="memory-result-stat-value">{usable_vram_gib:.1f} GiB</span>
            </div>
            <div class="memory-result-stat">
              <span class="memory-result-stat-label">当前余量</span>
              <span class="memory-result-stat-value">{free_capacity_gib:.1f} GiB · {free_ratio:.0%}</span>
            </div>
    """
    if previous_gpu_count > 0:
        result_stats_html += f"""
            <div class="memory-result-stat">
              <span class="memory-result-stat-label">{previous_gpu_count} 张卡仍不够</span>
              <span class="memory-result-stat-value">{previous_capacity_gib:.1f} GiB</span>
            </div>
        """
    result_stats_html += f"""
            <div class="memory-result-stat">
              <span class="memory-result-stat-label">并发余量系数</span>
              <span class="memory-result-stat-value">{fmt_value(result['concurrency_margin_ratio_p95'], 2)}</span>
            </div>
          </div>
    """
    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">1</span>
        <div>
          <div class="step-title">显存约束</div>
          <div class="step-subtitle">{render_math_text("M_w + M_r + M_cache 对应的总显存下界")}</div>
        </div>
      </div>
      <div class="memory-causality-shell">
        <div class="memory-result-spotlight">
          <span class="memory-result-kicker">显存约束结果</span>
          <div class="memory-result-main">
            <span class="memory-result-value">{gpu_count_by_memory}</span>
            <span class="memory-result-unit">GPUs</span>
          </div>
          <span class="memory-result-caption">显存口径下的最小卡数</span>
          <div class="memory-result-proof">
            <span class="memory-result-formula">{total_memory_gib:.1f} / {usable_vram_gib:.1f} = {required_gpu_ratio:.2f}</span>
            <span class="memory-result-note">单卡 {usable_vram_gib:.1f} GiB，向上取整得 {gpu_count_by_memory} 张。</span>
          </div>
          {result_stats_html}
        </div>
        <div class="memory-cause-cluster">
          <div class="memory-cause-heading">
            <span class="memory-cause-kicker">显存分布</span>
            <span class="memory-cause-summary">{gpu_count_by_memory} 张卡拼出的有效显存池为 {current_capacity_gib:.1f} GiB。下方分段条同时展示权重、固定开销、请求 Cache 和剩余空闲。</span>
          </div>
          <div class="mac-disk-layout">
            <div class="mac-disk-header">
              <span class="mac-disk-title">{gpu_count_by_memory} 张卡的有效显存池</span>
              <span class="mac-disk-capacity">{current_capacity_gib:.1f} GiB</span>
            </div>
            <div class="mac-disk-bar" aria-hidden="true">
              {disk_segments_html}
            </div>
            <div class="mac-disk-legend">
              {legend_html}
            </div>
          </div>
          <div class="memory-rollup-strip memory-rollup-strip-compact">
            <div class="memory-rollup-item">
              <span class="memory-rollup-label">已占用</span>
              <span class="memory-rollup-value">{total_memory_gib:.1f} GiB</span>
            </div>
            <div class="memory-rollup-item">
              <span class="memory-rollup-label">空闲</span>
              <span class="memory-rollup-value">{free_capacity_gib:.1f} GiB</span>
            </div>
            <div class="memory-rollup-item memory-rollup-emphasis">
              <span class="memory-rollup-label">利用率</span>
              <span class="memory-rollup-value">{used_ratio:.0%}</span>
            </div>
          </div>
        </div>
      </div>
      {section_html}
    </div>
    """


def build_throughput_analysis_html(result: dict[str, Any]) -> str:
    calc_sections = result.get("calculation_process_sections", [])
    workload_section = calc_sections[0] if len(calc_sections) > 0 else None
    throughput_section = calc_sections[2] if len(calc_sections) > 2 else None
    section_html = render_calc_accordion("吞吐必要条件与时延必要条件细节", throughput_section)
    prefill_stage_html = _render_tp_stage_shell(
        stage_key="prefill",
        stage_title="Prefill",
        badge_text="输入阶段",
        badge_class="tp-badge-prefill",
        result=result,
        sections=[workload_section, throughput_section],
        workload_label="峰值 Prefill 工作量",
        service_label="单卡 Prefill 理论服务率",
        reference_label="Prefill 理论参考卡数 G_pre",
        latency_label="Prefill 时延必要条件",
        workload_key="tps_pre_target_peak",
        cluster_capacity_key="cluster_prefill_tps_capacity_p95",
        reference_count_key="prefill_gpu_count_by_throughput",
        necessity_ok_key="prefill_necessity_met_at_baseline",
        necessity_gap_key="prefill_necessity_gpu_gap",
        latency_ok_key="prefill_latency_ok",
    )
    decode_stage_html = _render_tp_stage_shell(
        stage_key="decode",
        stage_title="Decode",
        badge_text="输出阶段",
        badge_class="tp-badge-decode",
        result=result,
        sections=[workload_section, throughput_section],
        workload_label="峰值 Decode 工作量",
        service_label="单卡 Decode 理论服务率",
        reference_label="Decode 理论参考卡数 G_dec",
        latency_label="Decode 时延必要条件",
        workload_key="tps_dec_target_peak",
        cluster_capacity_key="cluster_decode_tps_capacity",
        reference_count_key="decode_gpu_count_by_throughput",
        necessity_ok_key="decode_necessity_met_at_baseline",
        necessity_gap_key="decode_necessity_gpu_gap",
        latency_ok_key="decode_latency_ok",
    )
    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">2</span>
        <div>
          <div class="step-title">吞吐必要条件与时延必要条件</div>
          <div class="step-subtitle">{render_math_text("当前建议采购卡数只按显存约束给出；下面检查它是否与 Prefill / Decode 理论必要条件发生冲突")}</div>
        </div>
      </div>
      <div class="throughput-stage-grid">
        {prefill_stage_html}
        {decode_stage_html}
      </div>
      {section_html}
    </div>
    """


def build_final_summary_html(result: dict[str, Any]) -> str:
    dominant_constraints = list(result.get("dominant_constraints", [])) or ["显存"]
    dominant = " · ".join(dominant_constraints)
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("最终卡数与近似能力回推细节", calc_sections[3] if len(calc_sections) > 3 else None)
    cost_html = ""
    if result.get("estimated_total_cost") is not None:
        cost_html = f"""
        <div class="ha-detail-card">
          <span class="ha-detail-label">估算总成本</span>
          <span class="ha-detail-value">¥{result['estimated_total_cost']:,.0f}</span>
          <span class="ha-detail-meta">单卡 ¥{result['unit_price']:,.0f}</span>
        </div>
        """
    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">3</span>
        <div>
          <div class="step-title">最终卡数与近似能力回推</div>
          <div class="step-subtitle">{render_math_text("先给出 G_req，再给出基于必要条件模型的总量近似回推")}</div>
        </div>
      </div>
      <div class="decision-label">① 当前版本只按显存约束给采购基线</div>
      <div class="constraint-grid">
        <div class="constraint-card"><span class="constraint-title">采购基线</span><span class="constraint-value">{result['business_gpu_count']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">只由显存约束给出</span></div>
        <div class="constraint-card"><span class="constraint-title">Prefill 参考</span><span class="constraint-value">{result['prefill_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">{'当前基线满足' if result['prefill_necessity_met_at_baseline'] else f'当前基线仍差 {result["prefill_necessity_gpu_gap"]} 卡'}</span></div>
        <div class="constraint-card"><span class="constraint-title">Decode 参考</span><span class="constraint-value">{result['decode_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">{'当前基线满足' if result['decode_necessity_met_at_baseline'] else f'当前基线仍差 {result["decode_necessity_gpu_gap"]} 卡'}</span></div>
      </div>
      <div class="final-formula final-summary-card">
        <div class="formula-term formula-result">
          <span class="formula-label">建议采购</span>
          <span class="formula-value">{result['business_gpu_count']} GPUs</span>
        </div>
      </div>
      <div class="ha-detail-grid">
        <div class="ha-detail-card"><span class="ha-detail-label">主导约束</span><span class="ha-detail-value">{escape(dominant)}</span><span class="ha-detail-meta">业务基线由谁决定</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">理论可持续 QPS</span><span class="ha-detail-value">{fmt_value(result['sustainable_qps_p95'], 2)} req/s</span><span class="ha-detail-meta">基于必要条件模型的总量近似，不等于实测 SLA</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">最大在途请求量</span><span class="ha-detail-value">{result['max_concurrency_by_memory_p95']}</span><span class="ha-detail-meta">显存口径</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">理论每日输出 token</span><span class="ha-detail-value">{fmt_compact(result['daily_decode_token_capacity_p95'])}</span><span class="ha-detail-meta">按理论 Decode 服务率线性回推</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">理论每日输入 token</span><span class="ha-detail-value">{fmt_compact(result['daily_prefill_token_capacity_p95'])}</span><span class="ha-detail-meta">按理论 Prefill 服务率线性回推</span></div>
        {cost_html}
      </div>
      {section_html}
    </div>
    """
