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
            <span class="hero-strategy-tag">显存、输入吞吐、输出吞吐三类约束取最大值</span>
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
              <span class="result-item-sub">建议采购卡数下界</span>
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
    previous_capacity_html = ""
    if previous_gpu_count > 0:
        previous_capacity_html = f"""
        <div class="ha-detail-card memory-threshold-card">
          <span class="ha-detail-label">{previous_gpu_count} 张卡仍不够</span>
          <span class="ha-detail-value">{previous_capacity_gib:.1f} GiB</span>
          <span class="ha-detail-meta">仍低于 {total_memory_gib:.1f} GiB，因此不够装下</span>
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
            <span class="memory-result-note">单卡有效显存 {usable_vram_gib:.1f} GiB，向上取整后得到 {gpu_count_by_memory} 张。</span>
          </div>
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
      <div class="ha-detail-grid memory-support-grid">
        <div class="ha-detail-card">
          <span class="ha-detail-label">单卡有效显存</span>
          <span class="ha-detail-value">{usable_vram_gib:.1f} GiB</span>
          <span class="ha-detail-meta">厂商标称显存折算后再扣除安全水位</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">{gpu_count_by_memory} 张卡空闲显存</span>
          <span class="ha-detail-value">{free_capacity_gib:.1f} GiB</span>
          <span class="ha-detail-meta">当前最小可行卡数下还剩 {free_ratio:.0%} 显存余量</span>
        </div>
        {previous_capacity_html}
        <div class="ha-detail-card">
          <span class="ha-detail-label">并发余量系数</span>
          <span class="ha-detail-value">{fmt_value(result['concurrency_margin_ratio_p95'], 2)}</span>
          <span class="ha-detail-meta">显存层面还能承接多少在途请求</span>
        </div>
      </div>
      {section_html}
    </div>
    """


def build_throughput_analysis_html(result: dict[str, Any]) -> str:
    prefill_ok = "满足" if result["prefill_latency_ok"] else "不满足"
    decode_ok = "满足" if result["decode_latency_ok"] else "不满足"
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("吞吐与时延必要条件细节", calc_sections[2] if len(calc_sections) > 2 else None)
    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">2</span>
        <div>
          <div class="step-title">吞吐与时延必要条件</div>
          <div class="step-subtitle">{render_math_text("目标 token 工作量决定 G_pre / G_dec，时延只做单卡必要条件检查")}</div>
        </div>
      </div>
      <div class="throughput-grid">
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Prefill</span>
            <span class="tp-badge tp-badge-prefill">输入阶段</span>
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row"><span class="tp-metric-label">目标工作量</span><span class="tp-metric-value">{fmt_compact(result['tps_pre_target_peak'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">带宽上界</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_bw_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">算力上界</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_compute_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">单卡吞吐</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_card'])} tok/s</span></div>
          </div>
          <div class="tp-result"><span class="tp-result-label">卡数下界</span><span class="tp-result-value">{result['prefill_gpu_count_by_throughput']}</span></div>
          <div class="tp-result" style="margin-top:10px;"><span class="tp-result-label">TTFT 检查</span><span class="tp-result-value">{prefill_ok}</span></div>
        </div>
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Decode</span>
            <span class="tp-badge tp-badge-decode">输出阶段</span>
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row"><span class="tp-metric-label">目标工作量</span><span class="tp-metric-value">{fmt_compact(result['tps_dec_target_peak'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">带宽上界</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_bw_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">算力上界</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_compute_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">单卡吞吐</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_card'])} tok/s</span></div>
          </div>
          <div class="tp-result"><span class="tp-result-label">卡数下界</span><span class="tp-result-value">{result['decode_gpu_count_by_throughput']}</span></div>
          <div class="tp-result" style="margin-top:10px;"><span class="tp-result-label">Decode 时延检查</span><span class="tp-result-value">{decode_ok}</span></div>
        </div>
      </div>
      {section_html}
    </div>
    """


def build_final_summary_html(result: dict[str, Any]) -> str:
    dominant_constraints = list(result.get("dominant_constraints", [])) or ["显存"]
    dominant = " · ".join(dominant_constraints)
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("最终卡数与能力回推细节", calc_sections[3] if len(calc_sections) > 3 else None)
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
          <div class="step-title">最终卡数与能力回推</div>
          <div class="step-subtitle">{render_math_text("先给出 G_req，再回推总吞吐、可持续 QPS 与显存可承载在途量")}</div>
        </div>
      </div>
      <div class="decision-label">① 三类约束中取需求最高者</div>
      <div class="constraint-grid">
        <div class="constraint-card"><span class="constraint-title">显存压力</span><span class="constraint-value">{result['gpu_count_by_memory']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">由显存占用决定</span></div>
        <div class="constraint-card"><span class="constraint-title">输入处理压力</span><span class="constraint-value">{result['prefill_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">由输入吞吐决定</span></div>
        <div class="constraint-card"><span class="constraint-title">输出生成压力</span><span class="constraint-value">{result['decode_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">由输出吞吐决定</span></div>
      </div>
      <div class="final-formula final-summary-card">
        <div class="formula-term formula-result">
          <span class="formula-label">建议采购</span>
          <span class="formula-value">{result['business_gpu_count']} GPUs</span>
        </div>
      </div>
      <div class="ha-detail-grid">
        <div class="ha-detail-card"><span class="ha-detail-label">主导约束</span><span class="ha-detail-value">{escape(dominant)}</span><span class="ha-detail-meta">业务基线由谁决定</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">可持续 QPS</span><span class="ha-detail-value">{fmt_value(result['sustainable_qps_p95'], 2)} req/s</span><span class="ha-detail-meta">按当前配置可长期承接的请求速率</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">最大在途请求量</span><span class="ha-detail-value">{result['max_concurrency_by_memory_p95']}</span><span class="ha-detail-meta">显存口径</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">每日输出 token</span><span class="ha-detail-value">{fmt_compact(result['daily_decode_token_capacity_p95'])}</span><span class="ha-detail-meta">对应输出生成能力</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">每日输入 token</span><span class="ha-detail-value">{fmt_compact(result['daily_prefill_token_capacity_p95'])}</span><span class="ha-detail-meta">对应输入处理能力</span></div>
        {cost_html}
      </div>
      {section_html}
    </div>
    """
