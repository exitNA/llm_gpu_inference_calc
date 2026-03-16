from html import escape
from typing import Any

# HA 模式内部键 → 中文显示名
HA_MODE_CN = {
    "none": "不启用",
    "n_plus_1": "N+1 备份",
    "active_standby": "主备模式",
    "active_active": "双活",
}
from pathlib import Path
from presets import get_gpu_preset, get_model_preset

APP_CSS = Path(__file__).parent.parent.joinpath("static", "app.css").read_text(encoding="utf-8")


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def format_metric_value(value: Any, suffix: str = "") -> str:
    return _fmt(value, suffix)


def _fmt_tps_compact(value: Any) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )
    for threshold, suffix in units:
        if abs(amount) >= threshold:
            return f"{amount / threshold:.1f}{suffix}"
    return f"{amount:.0f}"


def _fmt_token_volume_compact(value: Any) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )
    for threshold, suffix in units:
        if abs(amount) >= threshold:
            return f"{amount / threshold:.1f}{suffix}"
    return f"{amount:.1f}"


# ── Helpers ──────────────────────────────────────────────────────────


def get_dominant_constraints(result: dict[str, Any]) -> list[str]:
    constraints: list[str] = []
    if result["business_gpu_count"] == result["gpu_count_by_memory"]:
        constraints.append("显存")
    if result["decode_gpu_count_by_throughput"] == result["business_gpu_count"]:
        constraints.append("Decode 吞吐")
    if result["prefill_gpu_count_by_throughput"] == result["business_gpu_count"]:
        constraints.append("Prefill 吞吐")
    return constraints or ["显存"]


def _constraint_cls(constraints: list[str]) -> str:
    if "Decode 吞吐" in constraints or "Prefill 吞吐" in constraints:
        return "constraint-throughput"
    return "constraint-memory"


# ── Kept for Gradio Dataframe tables ─────────────────────────────────


def build_request_detail_rows(result: dict[str, Any]) -> list[list[str]]:
    return [
        [
            item["name"],
            f"{item['ratio']:.0%}",
            str(item["avg_input_tokens"]),
            str(item["avg_output_tokens"]),
            str(item["seq_len_total"]),
        ]
        for item in result["request_shape_details"]
    ]


def build_kv_detail_rows(result: dict[str, Any]) -> list[list[str]]:
    return [
        [item["name"], str(item["seq_len_total"]), f"{item['kv_cache_gb_per_request']:.2f}"]
        for item in result["kv_distribution_details"]
    ]


# ── Kept for config panel ────────────────────────────────────────────


def build_traffic_profile_header_html() -> str:
    return """
    <div class="selection-card">
      <p class="selection-eyebrow">traffic profile</p>
      <p class="selection-copy">
        调整各请求类型的占比、输入/输出 token 数。占比之和应为 1.0。
      </p>
    </div>
    """


def build_shape_name_html(name: str) -> str:
    return f'<div class="selection-card" style="margin-top:6px"><strong>{escape(name)}</strong></div>'


def _render_calc_steps_compact(steps: list[dict[str, Any]]) -> str:
    """Render calculation steps as a compact derivation chain."""
    rows: list[str] = []
    for step in steps:
        if step.get("is_group"):
            # Render group label as a heading, then each variant indented
            variant_rows: list[str] = []
            for var in step.get("variants", []):
                note_html = ""
                if var.get("note"):
                    note_html = f'<div class="cd-note">{escape(var["note"])}</div>'
                variant_rows.append(f"""
                <div class="cd-row cd-variant">
                  <div class="cd-head">
                    <span class="cd-branch">{escape(var["name"])}</span>
                    <span class="cd-eq">=</span>
                    <span class="cd-result">{escape(var["result"])}</span>
                  </div>
                  <div class="cd-body">
                    <span class="cd-formula">{escape(var["formula"])}</span>
                    <span class="cd-subst">{escape(var["substitution"])}</span>
                  </div>
                  {note_html}
                </div>
                """)
            rows.append(f"""
            <div class="cd-group">
              <div class="cd-group-label">{escape(step["label"])}</div>
              {"".join(variant_rows)}
            </div>
            """)
        else:
            note_html = ""
            if step.get("note"):
                note_html = f'<div class="cd-note">{escape(step["note"])}</div>'
            rows.append(f"""
            <div class="cd-row">
              <div class="cd-head">
                <span class="cd-label">{escape(step["label"])}</span>
                <span class="cd-eq">=</span>
                <span class="cd-result">{escape(step["result"])}</span>
              </div>
              <div class="cd-body">
                <span class="cd-formula">{escape(step["formula"])}</span>
                <span class="cd-subst">{escape(step["substitution"])}</span>
              </div>
              {note_html}
            </div>
            """)

    return f"""<div class="cd-chain">{"".join(rows)}</div>"""



def _render_calc_accordion(title: str, steps: list[dict[str, Any]]) -> str:
    """Wrap calculation steps in a collapsible details section."""
    return f"""
    <div class="in-card-calc">
      <details>
        <summary><span>{escape(title)}</span></summary>
        <div class="calc-step-list">
          {_render_calc_steps_compact(steps)}
        </div>
      </details>
    </div>
    """


def build_calculation_process_html(result: dict[str, Any]) -> str:
    sections = result.get("calculation_process_sections", [])
    section_cards: list[str] = []
    for idx, section in enumerate(sections, start=1):
        step_cards: list[str] = []
        for step in section.get("steps", []):
            note_html = ""
            if step.get("note"):
                note_html = (
                    f'<div class="calc-step-note">{escape(step["note"])}</div>'
                )
            step_cards.append(
                f"""
                <div class="calc-step-card">
                  <div class="calc-step-main">
                    <div class="calc-step-topline">
                      <div class="calc-step-metric">{escape(step["label"])}</div>
                      <div class="calc-step-result-inline">{escape(step["result"])}</div>
                    </div>
                    <div class="calc-step-formula">
                      <span class="calc-step-k">公式</span>
                      <span class="calc-step-v">{escape(step["formula"])}</span>
                    </div>
                    <div class="calc-step-detail">
                      <span class="calc-step-k">代入</span>
                      <span class="calc-step-v">{escape(step["substitution"])}</span>
                    </div>
                    {note_html}
                  </div>
                </div>
                """
            )
        summary_html = ""
        if section.get("summary"):
            summary_html = (
                f'<p class="calc-section-summary">{escape(section["summary"])}</p>'
            )
        section_cards.append(
            f"""
            <section class="calc-section-card">
              <div class="calc-section-header">
                <span class="calc-section-index">{idx:02d}</span>
                <div class="calc-section-copy">
                  <h3>{escape(section['title'])}</h3>
                  {summary_html}
                </div>
              </div>
              <div class="calc-step-list">
                {''.join(step_cards)}
              </div>
            </section>
            """
        )
    return f"""
    <div class="calc-process-shell">
      <div class="calc-process-hero">
        <div>
          <p class="calc-process-eyebrow">Validation View</p>
          <h2>计算过程校验面板</h2>
        </div>
        <p class="calc-process-copy">按“先看结论、再看公式、最后看代入”的顺序组织，每张卡只表达一个结论，更接近人工复核时的阅读节奏。</p>
      </div>
      <div class="calc-process-grid">
        {''.join(section_cards)}
      </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
#  Step ① — 概览：最终 GPU 数量 + 输入上下文
# ═══════════════════════════════════════════════════════════════════════


def build_overview_html(result: dict[str, Any]) -> str:
    dominant = get_dominant_constraints(result)
    dominant_text = " · ".join(dominant)
    badge_cls = _constraint_cls(dominant)

    constraint_badges = "".join(
        f'<span class="hero-constraint {badge_cls}">{escape(c)}</span>'
        for c in dominant
    )

    gpu_name = escape(result["gpu_name"])
    prec = escape(result["precision"])
    kv_dtype = escape(result["kv_cache_dtype"])
    model_name = escape(result["model_name"])
    concurrency = result["traffic_config"].concurrency
    target_decode_tps = result["traffic_config"].target_decode_tps_total
    target_prefill_tps = result["traffic_config"].prefill_tokens_per_sec_total
    daily_decode = result.get("daily_decode_token_capacity")
    daily_prefill = result.get("daily_prefill_token_capacity")
    avg_conversation_duration = result.get("avg_conversation_duration_sec")
    avg_prefill_duration = result.get("avg_prefill_duration_sec")
    avg_decode_duration = result.get("avg_decode_duration_sec")
    params_b = result["model_config"].num_params_billion
    arch = escape(result.get("arch_family", ""))
    attn = escape(result.get("attention_type", ""))
    sizing_basis = result["memory_sizing_basis"]

    ha_mode_cn = HA_MODE_CN.get(result['ha_mode'], result['ha_mode'])
    runtime_gb = result.get('usable_vram_gb_per_gpu', 0)
    
    # Calculation process for Section 0: Request Profile Stats
    calc_sections = result.get("calculation_process_sections", [])
    section_0_html = ""
    if len(calc_sections) > 0:
        section_0_html = _render_calc_accordion("🔍 长度分布计算细节", calc_sections[0].get("steps", []))

    return f"""
    <div class="result-card-unified">
      <!-- 头部核心结论 -->
      <div class="result-card-header">
        <div class="hero-gpu-count-box">
          <span class="hero-gpu-number">{result['total_gpu_count_after_ha']}</span>
          <span class="hero-gpu-unit">GPUs</span>
        </div>
        <div class="result-card-title-area">
          <div class="hero-model-name-large">
            {model_name}
          </div>
          <div class="hero-tag-row">
            {constraint_badges}
            <span class="hero-strategy-tag">策略: {sizing_basis}</span>
          </div>
        </div>
      </div>

      <!-- 核心指标网格 -->
      <div class="result-card-body-grid">
        
        <!-- Column 1: 模型与算力 -->
        <div class="result-group-card">
          <div class="result-group-header">
            <span class="group-icon">🧠</span>
            <span class="result-group-title">模型参数</span>
          </div>
          
          <div class="result-item-row">
            <span class="result-item-label">规模</span>
            <div class="result-item-content">
              <span class="result-item-value">{params_b:.0f}B</span>
              <span class="result-item-sub">{arch.upper()} · {attn.upper()}</span>
            </div>
          </div>

          <div class="result-item-row">
            <span class="result-item-label">显卡</span>
            <div class="result-item-content">
              <span class="result-item-value">{result['gpu_config'].vram_gb:.0f}GB</span>
              <span class="result-item-sub">{gpu_name} ({prec}/{kv_dtype})</span>
            </div>
          </div>
        </div>

        <!-- Column 2: 流量与部署 -->
        <div class="result-group-card">
          <div class="result-group-header">
            <span class="group-icon">🚦</span>
            <span class="result-group-title">流量配置</span>
          </div>
          
          <div class="result-item-row">
            <span class="result-item-label">并发</span>
            <div class="result-item-content">
              <span class="result-item-value">{concurrency} Req</span>
              <span class="result-item-sub">In {result['avg_input_tokens']:.0f} / Out {result['avg_output_tokens']:.0f} tok</span>
            </div>
          </div>

          <div class="result-item-row">
            <span class="result-item-label">高可用</span>
            <div class="result-item-content">
              <span class="result-item-value">{ha_mode_cn}</span>
              <span class="result-item-sub">{result['replica_count']} 副本 · 冗余 +{result['ha_extra_gpu_count']} 卡</span>
            </div>
          </div>
        </div>

        <!-- Column 3: 服务能力 (SLAs) -->
        <div class="result-group-card highlight-group">
          <div class="result-group-header">
            <span class="group-icon">⚡</span>
            <span class="result-group-title">性能估算</span>
          </div>
          
          <div class="result-item-row">
            <span class="result-item-label">吞吐</span>
            <div class="result-item-content">
              <span class="result-item-value-compact">D {_fmt_tps_compact(target_decode_tps)} / P {_fmt_tps_compact(target_prefill_tps)} tok/s</span>
              <span class="result-item-sub">日上限 {_fmt_token_volume_compact(daily_decode)} / {_fmt_token_volume_compact(daily_prefill)} tok</span>
            </div>
          </div>

          <div class="result-item-row">
            <span class="result-item-label">响应</span>
            <div class="result-item-content">
              <span class="result-item-value">{_fmt(avg_conversation_duration, ' s')}</span>
              <span class="result-item-sub">P {_fmt(avg_prefill_duration, 's')} + D {_fmt(avg_decode_duration, 's')}</span>
            </div>
          </div>
        </div>

      </div>
    </div>
    {section_0_html}
    """


# ═══════════════════════════════════════════════════════════════════════
#  Step ② — 显存分析：为什么需要这么多卡
# ═══════════════════════════════════════════════════════════════════════


def build_memory_analysis_html(result: dict[str, Any]) -> str:
    weight_gb = result["weight_with_overhead_gb"]
    kv_total_gb = max(result["memory_for_sizing_gb"] - weight_gb - result["runtime_overhead_gb"], 0)
    runtime_gb = result["runtime_overhead_gb"]
    total_need = result["memory_for_sizing_gb"]
    usable_per_gpu = result["usable_vram_gb_per_gpu"]
    gpu_count_mem = result["gpu_count_by_memory"]

    capacity = usable_per_gpu * gpu_count_mem if gpu_count_mem > 0 else 1
    free_gb = max(capacity - total_need, 0)
    
    total_for_ring = total_need + free_gb
    if total_for_ring <= 0:
        total_for_ring = 1

    # Calculate percentages for the horizontal bar
    w_pct = (weight_gb / total_for_ring) * 100
    kv_pct = (kv_total_gb / total_for_ring) * 100
    rt_pct = (runtime_gb / total_for_ring) * 100
    free_pct = (free_gb / total_for_ring) * 100
    
    util_pct = min((total_need / capacity) * 100, 100) if capacity > 0 else 0
    
    raw_weight_gb = result["raw_weight_gb"]
    overhead_ratio = result["runtime_config"].weight_overhead_ratio
    runtime_overhead_ratio = result["runtime_config"].runtime_overhead_ratio
    kv_per_req = result.get("p95_kv_gb_per_request", result.get("avg_kv_gb_per_request", 0))
    concurrency = result["traffic_config"].concurrency

    # Calculation process for Section 1: Memory Estimation
    calc_sections = result.get("calculation_process_sections", [])
    section_1_html = ""
    if len(calc_sections) > 1:
        section_1_html = _render_calc_accordion("🔍 显存计算细节", calc_sections[1].get("steps", []))

    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">1</span>
        <div>
          <div class="step-title">显存分析</div>
          <div class="step-subtitle">模型权重、KV Cache 和运行时开销如何占满 GPU 显存</div>
        </div>
      </div>

      <div class="mac-disk-layout">
        <div class="mac-disk-header">
          <span class="mac-disk-title">已占用 {util_pct:.0f}%</span>
          <span class="mac-disk-capacity">共 {capacity:.0f} GB</span>
        </div>
        
        <div class="mac-disk-bar">
          <div class="mac-disk-segment bg-weight" style="width: {w_pct}%;"></div>
          <div class="mac-disk-segment bg-kv" style="width: {kv_pct}%;"></div>
          <div class="mac-disk-segment bg-runtime" style="width: {rt_pct}%;"></div>
          <div class="mac-disk-segment bg-free" style="width: {free_pct}%;"></div>
        </div>

        <div class="mac-disk-legend">
          <div class="mac-disk-legend-item">
            <span class="mem-legend-dot dot-weight"></span>
            <div class="mac-disk-legend-text">
              <div class="mac-disk-legend-top">
                <span class="mem-legend-name">模型权重</span>
                <span class="mem-legend-value">{weight_gb:.0f} GB</span>
              </div>
              <span class="mac-disk-detail-text">包含 {raw_weight_gb:.0f} GB 原始权重 + {overhead_ratio:.0%} 参数开销</span>
            </div>
          </div>
          <div class="mac-disk-legend-item">
            <span class="mem-legend-dot dot-kv"></span>
            <div class="mac-disk-legend-text">
              <div class="mac-disk-legend-top">
                <span class="mem-legend-name">KV Cache</span>
                <span class="mem-legend-value">{kv_total_gb:.0f} GB</span>
              </div>
              <span class="mac-disk-detail-text">基于 {result['memory_sizing_basis']}：单并发 {kv_per_req:.2f} GB × {concurrency} 并发</span>
            </div>
          </div>
          <div class="mac-disk-legend-item">
            <span class="mem-legend-dot dot-runtime"></span>
            <div class="mac-disk-legend-text">
              <div class="mac-disk-legend-top">
                <span class="mem-legend-name">运行时开销</span>
                <span class="mem-legend-value">{runtime_gb:.0f} GB</span>
              </div>
              <span class="mac-disk-detail-text">包含激活内存及 CUDA 上下文，权重 × {runtime_overhead_ratio:.0%}</span>
            </div>
          </div>
          <div class="mac-disk-legend-item">
            <span class="mem-legend-dot dot-free"></span>
            <div class="mac-disk-legend-text">
              <div class="mac-disk-legend-top">
                <span class="mem-legend-name">可用空闲</span>
                <span class="mem-legend-value">{free_gb:.0f} GB</span>
              </div>
              <span class="mac-disk-detail-text">剩余安全可用显存</span>
            </div>
          </div>
        </div>
      </div>

      <div class="mem-summary-strip">
        <div class="mem-summary-card">
          <span class="mem-summary-label">总显存需求</span>
          <span class="mem-summary-value">{total_need:.0f}</span>
          <span class="mem-summary-meta">GB ({result['memory_sizing_basis']})</span>
        </div>
        <div class="mem-summary-card">
          <span class="mem-summary-label">单卡可用</span>
          <span class="mem-summary-value">{usable_per_gpu:.0f}</span>
          <span class="mem-summary-meta">GB / 总 {result['gpu_config'].vram_gb:.0f} GB</span>
        </div>
        <div class="mem-summary-card">
          <span class="mem-summary-label">显存约束卡数</span>
          <span class="mem-summary-value">{gpu_count_mem}</span>
          <span class="mem-summary-meta">⌈ {total_need:.0f} ÷ {usable_per_gpu:.0f} ⌉</span>
        </div>
      </div>
      {section_1_html}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
#  Step ③ — 吞吐分析：Decode / Prefill 能力
# ═══════════════════════════════════════════════════════════════════════


def build_throughput_analysis_html(result: dict[str, Any]) -> str:
    decode_spec = result["decode_tps_per_gpu_spec"]
    prefill_spec = result["prefill_tps_per_gpu_spec"]
    decode_mem = result.get("decode_tps_per_gpu_memory_limited")
    decode_comp = result.get("decode_tps_per_gpu_compute_limited")
    prefill_mem = result.get("prefill_tps_per_gpu_memory_limited")
    prefill_comp = result.get("prefill_tps_per_gpu_compute_limited")
    decode_gpu = result["decode_gpu_count_by_throughput"]
    prefill_gpu = result["prefill_gpu_count_by_throughput"]
    bw_limited = result.get("theoretical_tokens_per_sec_bandwidth_limited")
    target_decode = result["traffic_config"].target_decode_tps_total
    target_prefill = result["traffic_config"].prefill_tokens_per_sec_total
    cluster_decode_capacity = result.get("cluster_decode_tps_capacity")
    cluster_prefill_capacity = result.get("cluster_prefill_tps_capacity")
    daily_decode = result.get("daily_decode_token_capacity")
    daily_prefill = result.get("daily_prefill_token_capacity")

    dominant = get_dominant_constraints(result)
    decode_is_bottleneck = "Decode 吞吐" in dominant
    prefill_is_bottleneck = "Prefill 吞吐" in dominant

    decode_badge = (
        '<span class="tp-badge tp-badge-bottleneck">瓶颈</span>'
        if decode_is_bottleneck
        else '<span class="tp-badge tp-badge-decode">Decode</span>'
    )
    prefill_badge = (
        '<span class="tp-badge tp-badge-bottleneck">瓶颈</span>'
        if prefill_is_bottleneck
        else '<span class="tp-badge tp-badge-prefill">Prefill</span>'
    )

    # Calculation process for Section 2: Throughput Estimation
    calc_sections = result.get("calculation_process_sections", [])
    section_2_html = ""
    if len(calc_sections) > 2:
        section_2_html = _render_calc_accordion("🔍 吞吐计算细节", calc_sections[2].get("steps", []))

    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">2</span>
        <div>
          <div class="step-title">吞吐分析</div>
          <div class="step-subtitle">Decode 和 Prefill 阶段的需求 vs 单卡处理能力</div>
        </div>
      </div>

      <div class="throughput-grid">
        <!-- Decode Card -->
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Decode 阶段</span>
            {decode_badge}
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row">
              <span class="tp-metric-label">集群需求 TPS</span>
              <span class="tp-metric-value">{_fmt_tps_compact(target_decode)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">单卡能力（综合）</span>
              <span class="tp-metric-value">{_fmt_tps_compact(decode_spec)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">├ 带宽受限</span>
              <span class="tp-metric-value">{_fmt_tps_compact(decode_mem)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">└ 算力受限</span>
              <span class="tp-metric-value">{_fmt_tps_compact(decode_comp)} tok/s</span>
            </div>
          </div>
          <div class="tp-result">
            <span class="tp-result-label">需要 GPU</span>
            <span class="tp-result-value">{decode_gpu if decode_gpu is not None else '-'}</span>
          </div>
          <div class="tp-result" style="margin-top: 10px;">
            <span class="tp-result-label">当前集群上限</span>
            <span class="tp-result-value">{_fmt_tps_compact(cluster_decode_capacity)} tok/s</span>
          </div>
          <div class="tp-result" style="margin-top: 10px;">
            <span class="tp-result-label">日供给上限</span>
            <span class="tp-result-value">{_fmt_token_volume_compact(daily_decode)} tok</span>
          </div>
        </div>

        <!-- Prefill Card -->
        <div class="tp-card">
          <div class="tp-card-header">
            <span class="tp-card-title">Prefill 阶段</span>
            {prefill_badge}
          </div>
          <div class="tp-metric-stack">
            <div class="tp-metric-row">
              <span class="tp-metric-label">集群需求 TPS</span>
              <span class="tp-metric-value">{_fmt_tps_compact(target_prefill)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">单卡能力（综合）</span>
              <span class="tp-metric-value">{_fmt_tps_compact(prefill_spec)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">├ 带宽受限</span>
              <span class="tp-metric-value">{_fmt_tps_compact(prefill_mem)} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">└ 算力受限</span>
              <span class="tp-metric-value">{_fmt_tps_compact(prefill_comp)} tok/s</span>
            </div>
          </div>
          <div class="tp-result">
            <span class="tp-result-label">需要 GPU</span>
            <span class="tp-result-value">{prefill_gpu if prefill_gpu is not None else '-'}</span>
          </div>
          <div class="tp-result" style="margin-top: 10px;">
            <span class="tp-result-label">当前集群上限</span>
            <span class="tp-result-value">{_fmt_tps_compact(cluster_prefill_capacity)} tok/s</span>
          </div>
          <div class="tp-result" style="margin-top: 10px;">
            <span class="tp-result-label">日供给上限</span>
            <span class="tp-result-value">{_fmt_token_volume_compact(daily_prefill)} tok</span>
          </div>
        </div>
      </div>

      <div class="tp-bw-strip">
        <span class="tp-bw-label">理论带宽上限（单卡）</span>
        <span class="tp-bw-value">{_fmt_tps_compact(bw_limited)} tok/s</span>
      </div>
      {section_2_html}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
#  Step ④ — 最终采购：HA 冗余 + 汇总
# ═══════════════════════════════════════════════════════════════════════


def build_final_summary_html(result: dict[str, Any]) -> str:
    biz = result["business_gpu_count"]
    ha_extra = result["ha_extra_gpu_count"]
    total = result["total_gpu_count_after_ha"]
    ha_mode = HA_MODE_CN.get(result["ha_mode"], result["ha_mode"])
    replicas = result["replica_count"]
    cost = result.get("estimated_total_cost")
    unit_price = result.get("unit_price")

    cost_html = ""
    if cost is not None:
        cost_html = f"""
        <div class="ha-detail-card">
          <span class="ha-detail-label">估算总成本</span>
          <span class="ha-detail-value">¥{cost:,.0f}</span>
          <span class="ha-detail-meta">单卡 ¥{unit_price:,.0f} × {total} 卡</span>
        </div>
        """

    dominant = get_dominant_constraints(result)
    dominant_text = " · ".join(dominant)
    daily_decode = result.get("daily_decode_token_capacity")
    daily_prefill = result.get("daily_prefill_token_capacity")
    avg_conversation_duration = result.get("avg_conversation_duration_sec")

    mem_gpu = result["gpu_count_by_memory"]
    decode_gpu = result["decode_gpu_count_by_throughput"]
    prefill_gpu = result["prefill_gpu_count_by_throughput"]

    # Determine which constraints are the bottleneck (could be multiple ties)
    is_mem_dominant = mem_gpu == biz
    is_decode_dominant = (decode_gpu is not None) and decode_gpu == biz
    is_prefill_dominant = (prefill_gpu is not None) and prefill_gpu == biz

    def _constraint_class(is_dominant: bool) -> str:
        return "constraint-card constraint-winner" if is_dominant else "constraint-card"

    def _winner_badge(is_dominant: bool) -> str:
        return '<span class="constraint-badge">⬆ 瓶颈</span>' if is_dominant else ''

    decode_val = decode_gpu if decode_gpu is not None else '-'
    prefill_val = prefill_gpu if prefill_gpu is not None else '-'

    # Calculation process for Section 3: Final Results & Derived Metrics
    calc_sections = result.get("calculation_process_sections", [])
    section_3_html = ""
    if len(calc_sections) > 3:
        section_3_html = _render_calc_accordion("🔍 最终采购与指标计算细节", calc_sections[3].get("steps", []))

    return f"""
    <div class="step-section">
      <div class="step-header">
        <span class="step-number">3</span>
        <div>
          <div class="step-title">最终采购汇总</div>
          <div class="step-subtitle">对比三大约束维度，取最大值作为业务基线，再叠加 HA 冗余</div>
        </div>
      </div>

      <!-- Phase 1: 三大约束对比 -->
      <div class="decision-label">① 三大约束对比 — 取最大值决定业务基线</div>
      <div class="constraint-grid">
        <div class="{_constraint_class(is_mem_dominant)}">
          {_winner_badge(is_mem_dominant)}
          <span class="constraint-title">显存约束</span>
          <span class="constraint-value">{mem_gpu}</span>
          <span class="constraint-unit">卡</span>
          <span class="constraint-reason">模型权重 + KV Cache 所需最少卡数</span>
        </div>
        <div class="{_constraint_class(is_decode_dominant)}">
          {_winner_badge(is_decode_dominant)}
          <span class="constraint-title">Decode 吞吐</span>
          <span class="constraint-value">{decode_val}</span>
          <span class="constraint-unit">卡</span>
          <span class="constraint-reason">满足解码阶段 TPS 需求的卡数</span>
        </div>
        <div class="{_constraint_class(is_prefill_dominant)}">
          {_winner_badge(is_prefill_dominant)}
          <span class="constraint-title">Prefill 吞吐</span>
          <span class="constraint-value">{prefill_val}</span>
          <span class="constraint-unit">卡</span>
          <span class="constraint-reason">满足预填充阶段 TPS 需求的卡数</span>
        </div>
      </div>

      <!-- Phase 2: 最终公式 -->
      <div class="decision-label">② 叠加高可用冗余 — 得出最终采购量</div>
      <div class="final-formula">
        <div class="formula-term">
          <span class="formula-label">业务基线</span>
          <span class="formula-value">{biz}</span>
        </div>
        <span class="formula-op">+</span>
        <div class="formula-term">
          <span class="formula-label">HA 冗余 ({ha_mode})</span>
          <span class="formula-value">{ha_extra}</span>
        </div>
        <span class="formula-op">=</span>
        <div class="formula-term formula-result">
          <span class="formula-label">最终采购</span>
          <span class="formula-value">{total}</span>
        </div>
      </div>

      <div class="ha-detail-grid">
        <div class="ha-detail-card">
          <span class="ha-detail-label">主导约束</span>
          <span class="ha-detail-value">{escape(dominant_text)}</span>
          <span class="ha-detail-meta">决定业务基线的瓶颈因素</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">HA 模式</span>
          <span class="ha-detail-value">{ha_mode}</span>
          <span class="ha-detail-meta">{replicas} 副本</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">每日 token 上限</span>
          <span class="ha-detail-value">D {_fmt_token_volume_compact(daily_decode)} / P {_fmt_token_volume_compact(daily_prefill)}</span>
          <span class="ha-detail-meta">按业务基线卡数估算，不重复计入 HA 冗余</span>
        </div>
        <div class="ha-detail-card">
          <span class="ha-detail-label">平均一次对话耗时</span>
          <span class="ha-detail-value">{_fmt(avg_conversation_duration, ' s')}</span>
          <span class="ha-detail-meta">按 G_biz 与当前并发近似估算</span>
        </div>
        {cost_html}
      </div>
      {section_3_html}
    </div>
    """


# ── Legacy aliases (removed old functions, keep backward compat) ──

def build_model_preset_html(preset_key: str) -> str:
    preset = get_model_preset(preset_key)
    config = preset.config
    rows = [
        f"<span>模型名称</span><strong>{escape(config.model_name)}</strong>",
        f"<span>总参数量</span><strong>{config.num_params_billion:.1f}B</strong>",
    ]
    if config.activated_params_billion:
        rows.append(f"<span>激活参数量</span><strong>{config.activated_params_billion:.1f}B</strong>")
    rows.append(f"<span>层数</span><strong>{config.num_layers}</strong>")
    rows.append(f"<span>Hidden Size</span><strong>{config.hidden_size}</strong>")
    metric_lines = "".join(f'<div class="metric-line">{r}</div>' for r in rows)
    return f"""
    <div class="selection-card">
      <p class="selection-eyebrow">model preset</p>
      <div class="metric-stack">{metric_lines}</div>
    </div>
    """


def build_gpu_preset_html(preset_key: str) -> str:
    preset = get_gpu_preset(preset_key)
    config = preset.config
    rows = [
        f"<span>显卡名称</span><strong>{escape(config.gpu_name)}</strong>",
        f"<span>显存容量</span><strong>{config.vram_gb:.0f} GB</strong>",
    ]
    if config.memory_bandwidth_gb_per_sec is not None:
        rows.append(f"<span>显存带宽</span><strong>{config.memory_bandwidth_gb_per_sec:.0f} GB/s</strong>")
    metric_lines = "".join(f'<div class="metric-line">{r}</div>' for r in rows)
    return f"""
    <div class="selection-card">
      <p class="selection-eyebrow">gpu preset</p>
      <div class="metric-stack">{metric_lines}</div>
    </div>
    """
