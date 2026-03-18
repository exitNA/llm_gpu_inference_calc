from __future__ import annotations

from math import ceil, floor

from .constants import PRECISION_BYTES


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


def build_calc_step(
    label: str,
    formula: str,
    substitution: str,
    result: str,
    note: str | None = None,
) -> dict[str, str]:
    step = {
        "label": label,
        "formula": formula,
        "substitution": substitution,
        "result": result,
    }
    if note:
        step["note"] = note
    return step
