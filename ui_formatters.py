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

APP_CSS = Path(__file__).parent.joinpath("app.css").read_text(encoding="utf-8")


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
    return f"{float(value):.0f}"


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
    params_b = result["model_config"].num_params_billion
    arch = escape(result.get("arch_family", ""))
    attn = escape(result.get("attention_type", ""))
    sizing_basis = result["memory_sizing_basis"]

    return f"""
    <div class="step-section" style="padding: 0; border: none; background: transparent; box-shadow: none;">
      <!-- Conclusion Hero -->
      <div class="conclusion-hero">
        <div class="hero-gpu-count">
          <span class="hero-gpu-number">{result['total_gpu_count_after_ha']}</span>
          <span class="hero-gpu-unit">GPUs 总采购</span>
        </div>
        <div class="hero-details">
          <div class="hero-model-name">{model_name}</div>
          <div class="hero-chips">
            <span class="hero-chip">{gpu_name}</span>
            <span class="hero-chip">{prec} / {kv_dtype}</span>
            <span class="hero-chip">{sizing_basis} 策略</span>
            <span class="hero-chip">并发 {concurrency}</span>
            <span class="hero-chip">速度 D {_fmt_tps_compact(target_decode_tps)} / P {_fmt_tps_compact(target_prefill_tps)} tok/s</span>
            {constraint_badges}
          </div>
        </div>
      </div>

      <!-- Input Context -->
      <div class="context-grid">
        <div class="context-card">
          <span class="context-label">模型</span>
          <span class="context-value">{params_b:.0f}B</span>
          <span class="context-meta">{arch.upper()} · {attn.upper()}</span>
        </div>
        <div class="context-card">
          <span class="context-label">GPU 显卡</span>
          <span class="context-value">{result['gpu_config'].vram_gb:.0f} GB</span>
          <span class="context-meta">{gpu_name} · 可用 {result['usable_vram_gb_per_gpu']:.0f} GB</span>
        </div>
        <div class="context-card">
          <span class="context-label">网关并发</span>
          <span class="context-value">{concurrency} 请求</span>
          <span class="context-meta">输入 {result['avg_input_tokens']:.0f} / 输出 {result['avg_output_tokens']:.0f} 词元 · D {_fmt_tps_compact(target_decode_tps)} / P {_fmt_tps_compact(target_prefill_tps)} tok/s</span>
        </div>
        <div class="context-card">
          <span class="context-label">高可用</span>
          <span class="context-value">{HA_MODE_CN.get(result['ha_mode'], result['ha_mode'])}</span>
          <span class="context-meta">{result['replica_count']} 副本 · 冗余 +{result['ha_extra_gpu_count']} 卡</span>
        </div>
      </div>
    </div>
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
              <span class="tp-metric-value">{target_decode:.0f} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">单卡能力（综合）</span>
              <span class="tp-metric-value">{_fmt(decode_spec, ' tok/s')}</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">├ 带宽受限</span>
              <span class="tp-metric-value">{_fmt(decode_mem, ' tok/s')}</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">└ 算力受限</span>
              <span class="tp-metric-value">{_fmt(decode_comp, ' tok/s')}</span>
            </div>
          </div>
          <div class="tp-result">
            <span class="tp-result-label">需要 GPU</span>
            <span class="tp-result-value">{decode_gpu if decode_gpu is not None else '-'}</span>
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
              <span class="tp-metric-value">{target_prefill:.0f} tok/s</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">单卡能力（综合）</span>
              <span class="tp-metric-value">{_fmt(prefill_spec, ' tok/s')}</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">├ 带宽受限</span>
              <span class="tp-metric-value">{_fmt(prefill_mem, ' tok/s')}</span>
            </div>
            <div class="tp-metric-row">
              <span class="tp-metric-label">└ 算力受限</span>
              <span class="tp-metric-value">{_fmt(prefill_comp, ' tok/s')}</span>
            </div>
          </div>
          <div class="tp-result">
            <span class="tp-result-label">需要 GPU</span>
            <span class="tp-result-value">{prefill_gpu if prefill_gpu is not None else '-'}</span>
          </div>
        </div>
      </div>

      <div class="tp-bw-strip">
        <span class="tp-bw-label">理论带宽上限（单卡）</span>
        <span class="tp-bw-value">{_fmt(bw_limited, ' tok/s')}</span>
      </div>
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
        {cost_html}
      </div>
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
