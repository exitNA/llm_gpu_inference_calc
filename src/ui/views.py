from __future__ import annotations

from html import escape
from typing import Any

from presets import get_gpu_preset, get_model_preset

from .common import (
    APP_CSS,
    fmt_compact,
    fmt_value,
    get_dominant_constraints,
    render_calc_accordion,
    render_calc_panel,
)


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
                fmt_value(item["kv_cache_gb_per_request"], 2, " GB"),
                "-" if item["max_concurrency_by_memory"] is None else str(item["max_concurrency_by_memory"]),
            ]
        )
    return rows


def build_traffic_profile_header_html() -> str:
    return """
    <div class="selection-card">
      <p class="selection-eyebrow">business inputs</p>
      <p class="selection-copy">
        按 sizing 主口径输入峰值 QPS、P95 输入输出长度，以及 P95 TTFT/E2E 目标。
      </p>
    </div>
    """


def build_shape_name_html(name: str) -> str:
    return f'<div class="selection-card" style="margin-top:6px"><strong>{escape(name)}</strong></div>'


def build_calculation_process_html(result: dict[str, Any]) -> str:
    sections = result.get("calculation_process_sections", [])
    cards = [render_calc_panel(section, idx) for idx, section in enumerate(sections, start=1)]
    return f"""
    <div class="calc-process-shell">
      <div class="calc-process-hero">
        <div>
          <p class="calc-process-eyebrow">Calculation Detail View</p>
          <h2>计算细节总览</h2>
        </div>
        <p class="calc-process-copy">每一步先看结论，再看公式与代入过程，便于人工快速复核。</p>
      </div>
      <div class="calc-process-grid">
        {''.join(cards)}
      </div>
    </div>
    """


def build_overview_html(result: dict[str, Any]) -> str:
    dominant = " · ".join(get_dominant_constraints(result))
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
            <span class="hero-strategy-tag">G_req = max(G_mem, G_pre, G_dec)</span>
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
            <span class="result-item-label">QPS</span>
            <div class="result-item-content">
              <span class="result-item-value">{traffic.lambda_peak_qps:.2f}</span>
              <span class="result-item-sub">峰值 req/s</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">长度</span>
            <div class="result-item-content">
              <span class="result-item-value">{traffic.p95_total_tokens}</span>
              <span class="result-item-sub">P95 总长度 tokens</span>
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
              <span class="result-item-value-compact">G_req {result['business_gpu_count']}</span>
              <span class="result-item-sub">文档主口径卡数下界</span>
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
    gpu = result["gpu_config"]
    peak_cache_gb = max(result["memory_for_sizing_gb"] - result["weight_with_overhead_gb"] - result["runtime_overhead_gb"], 0.0)
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("显存约束计算细节", calc_sections[1] if len(calc_sections) > 1 else None)
    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">1</span>
        <div>
          <div class="step-title">显存约束</div>
          <div class="step-subtitle">Mw + Mr + M_cache 对应的总显存下界</div>
        </div>
      </div>
      <div class="constraint-grid">
        <div class="constraint-card constraint-winner">
          <span class="constraint-title">权重显存 Mw</span>
          <span class="constraint-value">{result['weight_with_overhead_gb']:.1f}</span>
          <span class="constraint-unit">GB</span>
          <span class="constraint-reason">含 α_w 后的模型权重显存</span>
        </div>
        <div class="constraint-card">
          <span class="constraint-title">固定显存 Mr</span>
          <span class="constraint-value">{result['runtime_overhead_gb']:.1f}</span>
          <span class="constraint-unit">GB</span>
          <span class="constraint-reason">按 α_r × Mw 估算</span>
        </div>
        <div class="constraint-card">
          <span class="constraint-title">峰值 Cache</span>
          <span class="constraint-value">{peak_cache_gb:.1f}</span>
          <span class="constraint-unit">GB</span>
          <span class="constraint-reason">C_peak^budget × M_cache^req(S_p95)</span>
        </div>
      </div>
      <div class="ha-detail-grid">
        <div class="ha-detail-card">
          <span class="ha-detail-label">总显存需求</span>
          <span class="ha-detail-value">{result['memory_for_sizing_gb']:.1f} GB</span>
          <span class="ha-detail-meta">峰值 QPS + P95 长度口径</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">单卡有效显存</span>
          <span class="ha-detail-value">{result['usable_vram_gb_per_gpu']:.1f} GB</span>
          <span class="ha-detail-meta">{gpu.vram_gb:.0f} GB × η_vram</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">显存约束卡数</span>
          <span class="ha-detail-value">{result['gpu_count_by_memory']}</span>
          <span class="ha-detail-meta">G_mem</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">并发余量系数</span>
          <span class="ha-detail-value">{fmt_value(result['concurrency_margin_ratio_p95'], 2)}</span>
          <span class="ha-detail-meta">ρ_conc,p95</span>
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
          <div class="step-subtitle">峰值 token 工作量决定 G_pre/G_dec，时延只做单卡必要条件检查</div>
        </div>
      </div>
      <div class="throughput-grid">
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Prefill</span>
            <span class="tp-badge tp-badge-prefill">G_pre</span>
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row"><span class="tp-metric-label">峰值工作量</span><span class="tp-metric-value">{fmt_compact(result['tps_pre_target_peak'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">带宽上界</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_bw_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">算力上界</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_compute_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">单卡吞吐</span><span class="tp-metric-value">{fmt_compact(result['prefill_tps_p95_card'])} tok/s</span></div>
          </div>
          <div class="tp-result"><span class="tp-result-label">卡数下界</span><span class="tp-result-value">{result['prefill_gpu_count_by_throughput']}</span></div>
          <div class="tp-result" style="margin-top:10px;"><span class="tp-result-label">P95 TTFT 检查</span><span class="tp-result-value">{prefill_ok}</span></div>
        </div>
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Decode</span>
            <span class="tp-badge tp-badge-decode">G_dec</span>
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row"><span class="tp-metric-label">峰值工作量</span><span class="tp-metric-value">{fmt_compact(result['tps_dec_target_peak'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">带宽上界</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_bw_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">算力上界</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_compute_limited'])} tok/s</span></div>
            <div class="tp-metric-row"><span class="tp-metric-label">单卡吞吐</span><span class="tp-metric-value">{fmt_compact(result['decode_tps_card'])} tok/s</span></div>
          </div>
          <div class="tp-result"><span class="tp-result-label">卡数下界</span><span class="tp-result-value">{result['decode_gpu_count_by_throughput']}</span></div>
          <div class="tp-result" style="margin-top:10px;"><span class="tp-result-label">P95 Decode 检查</span><span class="tp-result-value">{decode_ok}</span></div>
        </div>
      </div>
      {section_html}
    </div>
    """


def build_final_summary_html(result: dict[str, Any]) -> str:
    dominant = " · ".join(get_dominant_constraints(result))
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
          <div class="step-subtitle">先给出 G_req，再回推总吞吐、可持续 QPS 与显存可承载在途量</div>
        </div>
      </div>
      <div class="decision-label">① 三类约束取最大值</div>
      <div class="constraint-grid">
        <div class="constraint-card"><span class="constraint-title">G_mem</span><span class="constraint-value">{result['gpu_count_by_memory']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">显存下界</span></div>
        <div class="constraint-card"><span class="constraint-title">G_pre</span><span class="constraint-value">{result['prefill_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">Prefill 吞吐下界</span></div>
        <div class="constraint-card"><span class="constraint-title">G_dec</span><span class="constraint-value">{result['decode_gpu_count_by_throughput']}</span><span class="constraint-unit">卡</span><span class="constraint-reason">Decode 吞吐下界</span></div>
      </div>
      <div class="final-formula">
        <div class="formula-term formula-result">
          <span class="formula-label">G_req</span>
          <span class="formula-value">{result['business_gpu_count']}</span>
        </div>
      </div>
      <div class="ha-detail-grid">
        <div class="ha-detail-card"><span class="ha-detail-label">主导约束</span><span class="ha-detail-value">{escape(dominant)}</span><span class="ha-detail-meta">业务基线由谁决定</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">保守可持续 QPS</span><span class="ha-detail-value">{fmt_value(result['sustainable_qps_p95'], 2)} req/s</span><span class="ha-detail-meta">P95 长度口径</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">最大在途请求量</span><span class="ha-detail-value">{result['max_concurrency_by_memory_p95']}</span><span class="ha-detail-meta">显存口径 P95</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">每日 Decode token</span><span class="ha-detail-value">{fmt_compact(result['daily_decode_token_capacity_p95'])}</span><span class="ha-detail-meta">保守口径</span></div>
        <div class="ha-detail-card"><span class="ha-detail-label">每日 Prefill token</span><span class="ha-detail-value">{fmt_compact(result['daily_prefill_token_capacity_p95'])}</span><span class="ha-detail-meta">保守口径</span></div>
        {cost_html}
      </div>
      {section_html}
    </div>
    """


def build_model_preset_html(preset_key: str) -> str:
    preset = get_model_preset(preset_key)
    cfg = preset.config
    return f"""
    <div class="selection-card">
      <p class="selection-eyebrow">model preset</p>
      <div class="metric-stack">
        <div class="metric-line"><span>模型名称</span><strong>{escape(cfg.model_name)}</strong></div>
        <div class="metric-line"><span>总参数量</span><strong>{cfg.num_params_billion:.1f}B</strong></div>
        <div class="metric-line"><span>层数</span><strong>{cfg.num_layers}</strong></div>
      </div>
    </div>
    """


def build_gpu_preset_html(preset_key: str) -> str:
    preset = get_gpu_preset(preset_key)
    cfg = preset.config
    return f"""
    <div class="selection-card">
      <p class="selection-eyebrow">gpu preset</p>
      <div class="metric-stack">
        <div class="metric-line"><span>显卡名称</span><strong>{escape(cfg.gpu_name)}</strong></div>
        <div class="metric-line"><span>显存容量</span><strong>{cfg.vram_gb:.0f} GB</strong></div>
        <div class="metric-line"><span>显存带宽</span><strong>{fmt_value(cfg.memory_bandwidth_gb_per_sec, 0, ' GB/s')}</strong></div>
      </div>
    </div>
    """
