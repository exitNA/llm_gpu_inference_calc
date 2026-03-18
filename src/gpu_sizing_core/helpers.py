from __future__ import annotations

from math import ceil, floor

from .constants import PRECISION_BYTES

FORMULA_TERM_LABELS: list[tuple[str, str]] = [
    ("TPS_pre,target^avg", "平均 Prefill 工作量"),
    ("TPS_dec,target^avg", "平均 Decode 工作量"),
    ("TPS_pre,target^peak", "峰值 Prefill 工作量"),
    ("TPS_dec,target^peak", "峰值 Decode 工作量"),
    ("TPS_pre,p95^cap", "P95 总 Prefill 吞吐能力"),
    ("TPS_pre,bw^card", "Prefill 带宽上界"),
    ("TPS_pre,cmp^card", "Prefill 算力上界"),
    ("TPS_pre^card", "单卡 Prefill 吞吐"),
    ("TPS_dec^cap", "总 Decode 吞吐能力"),
    ("TPS_dec,bw^card", "Decode 带宽上界"),
    ("TPS_dec,cmp^card", "Decode 算力上界"),
    ("TPS_dec^card", "单卡 Decode 吞吐"),
    ("C_max,p95^mem", "显存口径最大在途请求量"),
    ("C_peak^budget", "峰值在途预算"),
    ("C_avg^budget", "平均在途预算"),
    ("M_cache^req(S_p95)", "单请求 P95 Cache 显存"),
    ("M_cache^req(S_avg)", "平均单请求 Cache 显存"),
    ("M_cache", "总 Cache 显存"),
    ("V_cache^avail", "可用于 Cache 的总显存"),
    ("V_gpu^eff", "单卡有效显存"),
    ("T_dec,p95", "P95 Decode 时间预算"),
    ("T_dec,avg", "平均 Decode 时间预算"),
    ("Req_day^p95", "保守日请求量"),
    ("λ_p95^sus", "保守可持续 QPS"),
    ("λ_avg^sus", "常态可持续 QPS"),
    ("λ_peak", "峰值 QPS"),
    ("λ_avg", "平均 QPS"),
    ("S_in,p95", "P95 输入长度"),
    ("S_out,p95", "P95 输出长度"),
    ("S_in,avg", "平均输入长度"),
    ("S_out,avg", "平均输出长度"),
    ("S_p95", "P95 总长度"),
    ("S_avg", "平均总长度"),
    ("TTFT_p95,target", "P95 TTFT 目标"),
    ("TTFT_p95", "P95 TTFT"),
    ("TTFT_avg", "平均 TTFT"),
    ("E2E_p95", "P95 E2E 目标"),
    ("E2E_avg", "平均 E2E 目标"),
    ("G_req", "业务基线卡数"),
    ("G_mem", "显存约束卡数"),
    ("G_pre", "Prefill 吞吐卡数"),
    ("G_dec", "Decode 吞吐卡数"),
    ("Mw", "权重显存"),
    ("Mr", "运行时固定显存"),
    ("P_total", "总参数量"),
    ("P_act", "每 token 激活参数量"),
    ("B_mem", "显存带宽"),
    ("F_peak", "峰值算力"),
    ("b_pre", "Prefill 单 token 字节开销"),
    ("b_dec", "Decode 单 token 字节开销"),
    ("α_attn", "注意力额外算力系数"),
    ("α_w", "权重附加系数"),
    ("α_r", "运行时固定显存系数"),
    ("η_bw", "带宽利用率"),
    ("η_cmp", "算力利用率"),
    ("η_vram", "可用显存比例"),
    ("ρ_conc,p95", "并发余量系数"),
    ("TPOT", "单 token 输出时延"),
    ("L", "模型层数"),
]


def precision_to_bytes(precision: str) -> float:
    key = precision.lower()
    if key not in PRECISION_BYTES:
        raise ValueError(f"Unsupported precision: {precision}")
    return PRECISION_BYTES[key]


def format_calc_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    rounded = round(float(value), digits)
    if float(rounded).is_integer():
        return f"{int(rounded):,}"
    return f"{rounded:,.{digits}f}".rstrip("0").rstrip(".")


def bytes_to_gb(num_bytes: float) -> float:
    return num_bytes / 1e9


def gb_to_bytes(num_gb: float) -> float:
    return num_gb * 1e9


def ceil_div(a: float, b: float) -> int:
    return ceil(a / b)


def floor_div(a: float, b: float) -> int:
    return floor(a / b)


def round_optional(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def multiply_optional(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def divide_optional(value: float, divisor: float | None) -> float | None:
    if divisor is None or divisor <= 0:
        return None
    return value / divisor


def format_adaptive_memory(bytes_val: float | None, digits: int = 2) -> str:
    if bytes_val is None:
        return "-"
    if bytes_val >= 1e9:
        return f"{format_calc_number(bytes_val / 1e9, digits)} GB"
    if bytes_val >= 1e6:
        return f"{format_calc_number(bytes_val / 1e6, digits)} MB"
    if bytes_val >= 1e3:
        return f"{format_calc_number(bytes_val / 1e3, digits)} KB"
    return f"{format_calc_number(bytes_val, 0)} bytes"


def format_adaptive_tps(value: float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))
    for threshold, suffix in units:
        if abs(amount) >= threshold:
            return f"{amount / threshold:.1f}{suffix} tok/s"
    return f"{amount:.2f} tok/s"


def format_adaptive_token_volume(value: float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))
    for threshold, suffix in units:
        if abs(amount) >= threshold:
            return f"{amount / threshold:.1f}{suffix} tok"
    return f"{amount:.1f} tok"


def format_ratio_percent(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def humanize_formula(formula: str) -> str:
    humanized = formula
    for source, target in FORMULA_TERM_LABELS:
        humanized = humanized.replace(source, target)
    humanized = humanized.replace("ceil", "向上取整")
    humanized = humanized.replace("floor", "向下取整")
    humanized = humanized.replace("min", "取较小值")
    humanized = humanized.replace("max", "取较大值")
    return humanized


def build_calc_step(
    label: str,
    formula: str,
    substitution: str,
    result: str,
    note: str | None = None,
    formula_note: str | None = None,
) -> dict[str, str]:
    step = {
        "label": label,
        "formula": formula,
        "substitution": substitution,
        "result": result,
    }
    resolved_formula_note = formula_note or humanize_formula(formula)
    if resolved_formula_note and resolved_formula_note != formula:
        step["formula_note"] = resolved_formula_note
    if note:
        step["note"] = note
    return step
