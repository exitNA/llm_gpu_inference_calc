from __future__ import annotations

from html import escape
from typing import Any

from .common import (
    fmt_compact,
    fmt_value,
    render_calc_accordion,
    render_math_text,
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
            <span class="result-item-label">QPS</span>
            <div class="result-item-content">
              <span class="result-item-value">{traffic.lambda_peak_qps:.2f}</span>
              <span class="result-item-sub">业务请求速率</span>
            </div>
          </div>
          <div class="result-item-row">
            <span class="result-item-label">长度</span>
            <div class="result-item-content">
              <span class="result-item-value">{traffic.p95_total_tokens}</span>
              <span class="result-item-sub">总长度 tokens</span>
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
    weight_gib = result["weight_with_overhead_gib"]
    runtime_gib = result["runtime_overhead_gib"]
    total_memory_gib = result["memory_for_sizing_gib"]
    usable_vram_gib = result["usable_vram_gib_per_gpu"]
    gpu_count_by_memory = result["gpu_count_by_memory"]
    peak_cache_gib = max(total_memory_gib - weight_gib - runtime_gib, 0.0)
    required_gpu_ratio = total_memory_gib / usable_vram_gib if usable_vram_gib > 0 else 0.0
    previous_gpu_count = max(gpu_count_by_memory - 1, 0)
    previous_capacity_gib = previous_gpu_count * usable_vram_gib
    current_capacity_gib = gpu_count_by_memory * usable_vram_gib

    causes = [
        ("权重显存", weight_gib, "模型加载后的权重占用"),
        ("固定显存", runtime_gib, "框架与运行时常驻开销"),
        ("Cache 需求", peak_cache_gib, "按并发预算估算的请求缓存占用"),
    ]
    dominant_cause_title = max(causes, key=lambda item: item[1])[0]
    cause_cards = []
    for title, value, reason in causes:
        share = value / total_memory_gib if total_memory_gib > 0 else 0.0
        cause_cards.append(
            f"""
        <div class="constraint-card memory-cause-card{' cause-dominant' if title == dominant_cause_title else ''}">
          <span class="constraint-title">{title}</span>
          <span class="constraint-value">{value:.1f}</span>
          <span class="constraint-unit">GiB</span>
          <span class="constraint-reason">{reason}</span>
          <span class="memory-cause-share">占总需求 {share:.0%}</span>
        </div>
        """
        )
    cause_cards_html = "".join(cause_cards)
    calc_sections = result.get("calculation_process_sections", [])
    section_html = render_calc_accordion("显存约束计算细节", calc_sections[1] if len(calc_sections) > 1 else None)
    previous_capacity_html = ""
    if previous_gpu_count > 0:
        previous_capacity_html = f"""
        <div class="ha-detail-card memory-threshold-card">
          <span class="ha-detail-label">{previous_gpu_count} 张卡总容量</span>
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
        <div class="memory-cause-cluster">
          <div class="memory-cause-heading">
            <span class="memory-cause-kicker">原因：显存被这三部分占用</span>
            <span class="memory-cause-summary">先把显存用量加总，再除以单卡有效显存，最后向上取整得到卡数。</span>
          </div>
          <div class="constraint-grid memory-cause-grid">
            {cause_cards_html}
          </div>
          <div class="memory-rollup-strip">
            <div class="memory-rollup-item">
              <span class="memory-rollup-label">总显存需求</span>
              <span class="memory-rollup-value">{total_memory_gib:.1f} GiB</span>
            </div>
            <span class="memory-rollup-operator">/</span>
            <div class="memory-rollup-item">
              <span class="memory-rollup-label">单卡有效显存</span>
              <span class="memory-rollup-value">{usable_vram_gib:.1f} GiB</span>
            </div>
            <span class="memory-rollup-operator">=</span>
            <div class="memory-rollup-item memory-rollup-emphasis">
              <span class="memory-rollup-label">理论需求</span>
              <span class="memory-rollup-value">{required_gpu_ratio:.2f} 张</span>
            </div>
          </div>
        </div>
        <div class="memory-causal-arrow" aria-hidden="true">→</div>
        <div class="memory-result-spotlight">
          <span class="memory-result-kicker">结果：显存约束卡数</span>
          <div class="memory-result-main">
            <span class="memory-result-value">{gpu_count_by_memory}</span>
            <span class="memory-result-unit">GPUs</span>
          </div>
          <span class="memory-result-caption">显存口径下至少需要这么多卡</span>
          <div class="memory-result-proof">
            <span class="memory-result-formula">{total_memory_gib:.1f} / {usable_vram_gib:.1f} = {required_gpu_ratio:.2f}</span>
            <span class="memory-result-note">卡数必须取整数，所以向上取整后得到 {gpu_count_by_memory} 张。</span>
          </div>
        </div>
      </div>
      <div class="ha-detail-grid memory-support-grid">
        <div class="ha-detail-card">
          <span class="ha-detail-label">总显存需求</span>
          <span class="ha-detail-value">{total_memory_gib:.1f} GiB</span>
          <span class="ha-detail-meta">按字节换算后的二进制容量</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">单卡有效显存</span>
          <span class="ha-detail-value">{usable_vram_gib:.1f} GiB</span>
          <span class="ha-detail-meta">由厂商标称显存折算并扣除安全水位</span>
        </div>
        {previous_capacity_html}
        <div class="ha-detail-card memory-threshold-card memory-threshold-card-current">
          <span class="ha-detail-label">{gpu_count_by_memory} 张卡总容量</span>
          <span class="ha-detail-value">{current_capacity_gib:.1f} GiB</span>
          <span class="ha-detail-meta">首次覆盖 {total_memory_gib:.1f} GiB 需求，因此这是最小可行值</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">并发余量系数</span>
          <span class="ha-detail-value">{fmt_value(result['concurrency_margin_ratio_p95'], 2)}</span>
          <span class="ha-detail-meta">显存上还能承接多少在途请求</span>
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
