from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

APP_CSS = Path(__file__).resolve().parent.parent.parent.joinpath("static", "app.css").read_text(encoding="utf-8")


def fmt_value(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def fmt_compact(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))
    for threshold, suffix in units:
        if abs(amount) >= threshold:
            return f"{amount / threshold:.{digits}f}{suffix}"
    return f"{amount:.{digits}f}".rstrip("0").rstrip(".")


def get_dominant_constraints(result: dict[str, Any]) -> list[str]:
    return list(result.get("dominant_constraints", [])) or ["显存"]


def render_calc_section_steps(section: dict[str, Any]) -> str:
    step_cards = []
    for step in section.get("steps", []):
        note_html = ""
        if step.get("note"):
            note_html = f'<div class="calc-step-note">{escape(step["note"])}</div>'
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
    return "".join(step_cards)


def render_calc_accordion(title: str, section: dict[str, Any] | None) -> str:
    if not section:
        return ""
    step_cards = render_calc_section_steps(section)
    summary_html = ""
    if section.get("summary"):
        summary_html = f'<div class="calc-step-note">{escape(section["summary"])}</div>'
    return f"""
    <div class="in-card-calc">
      <details>
        <summary><span>{escape(title)}</span></summary>
        <div class="calc-step-list">
          {summary_html}
          {step_cards}
        </div>
      </details>
    </div>
    """


def render_calc_panel(section: dict[str, Any], idx: int) -> str:
    step_cards = render_calc_section_steps(section)
    summary_html = ""
    if section.get("summary"):
        summary_html = f'<p class="calc-section-summary">{escape(section["summary"])}</p>'
    return f"""
    <section class="calc-section-card">
      <div class="calc-section-header">
        <span class="calc-section-index">{idx:02d}</span>
        <div class="calc-section-copy">
          <h3>{escape(section["title"])}</h3>
          {summary_html}
        </div>
      </div>
      <div class="calc-step-list">
        {step_cards}
      </div>
    </section>
    """
