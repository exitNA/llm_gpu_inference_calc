from __future__ import annotations

from html import escape
from pathlib import Path
import re
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


_MATH_REPLACEMENTS: list[tuple[str, str]] = [
    ("TPS_pre,target^peak", '<span class="math-inline">TPS<sub>pre</sub></span>'),
    ("TPS_dec,target^peak", '<span class="math-inline">TPS<sub>dec</sub></span>'),
    ("TPS_pre,p95^cap", '<span class="math-inline">TPS<sub>pre,cap</sub></span>'),
    ("TPS_dec^cap", '<span class="math-inline">TPS<sub>dec,cap</sub></span>'),
    ("TPS_pre,bw^card", '<span class="math-inline">TPS<sub>pre,bw</sub></span>'),
    ("TPS_pre,cmp^card", '<span class="math-inline">TPS<sub>pre,cmp</sub></span>'),
    ("TPS_pre^card", '<span class="math-inline">TPS<sub>pre,card</sub></span>'),
    ("TPS_dec,bw^card", '<span class="math-inline">TPS<sub>dec,bw</sub></span>'),
    ("TPS_dec,cmp^card", '<span class="math-inline">TPS<sub>dec,cmp</sub></span>'),
    ("TPS_dec^card", '<span class="math-inline">TPS<sub>dec,card</sub></span>'),
    ("C_peak^budget", '<span class="math-inline">C<sub>peak</sub></span>'),
    ("C_max,p95^mem", '<span class="math-inline">C<sub>max</sub>(P95)</span>'),
    ("M_cache^req(S_p95)", '<span class="math-inline">M<sub>req</sub>(P95)</span>'),
    ("M_cache", '<span class="math-inline">M<sub>cache</sub></span>'),
    ("V_cache^avail", '<span class="math-inline">V<sub>cache</sub></span>'),
    ("V_gpu^eff", '<span class="math-inline">V<sub>gpu</sub></span>'),
    ("T_dec,p95", '<span class="math-inline">T<sub>dec</sub>(P95)</span>'),
    ("Req_day^p95", '<span class="math-inline">Req<sub>day</sub>(P95)</span>'),
    ("λ_p95^sus", '<span class="math-inline">&lambda;<sub>sus</sub></span>'),
    ("λ_peak", '<span class="math-inline">&lambda;<sub>peak</sub></span>'),
    ("S_in,p95", '<span class="math-inline">S<sub>in</sub>(P95)</span>'),
    ("S_out,p95", '<span class="math-inline">S<sub>out</sub>(P95)</span>'),
    ("S_p95", '<span class="math-inline">S(P95)</span>'),
    ("TTFT_p95,target", '<span class="math-inline">TTFT<sub>p95</sub></span>'),
    ("TTFT_p95", '<span class="math-inline">TTFT<sub>p95</sub></span>'),
    ("E2E_p95", '<span class="math-inline">E2E<sub>p95</sub></span>'),
    ("G_req", '<span class="math-inline">G<sub>req</sub></span>'),
    ("G_mem", '<span class="math-inline">G<sub>mem</sub></span>'),
    ("G_pre", '<span class="math-inline">G<sub>pre</sub></span>'),
    ("G_dec", '<span class="math-inline">G<sub>dec</sub></span>'),
    ("Mw", '<span class="math-inline">M<sub>w</sub></span>'),
    ("Mr", '<span class="math-inline">M<sub>r</sub></span>'),
    ("P_total", '<span class="math-inline">P<sub>total</sub></span>'),
    ("P_act", '<span class="math-inline">P<sub>act</sub></span>'),
    ("B_mem", '<span class="math-inline">B<sub>mem</sub></span>'),
    ("F_peak", '<span class="math-inline">F<sub>peak</sub></span>'),
    ("b_pre", '<span class="math-inline">b<sub>pre</sub></span>'),
    ("b_dec", '<span class="math-inline">b<sub>dec</sub></span>'),
    ("α_attn", '<span class="math-inline">&alpha;<sub>attn</sub></span>'),
    ("α_w", '<span class="math-inline">&alpha;<sub>w</sub></span>'),
    ("α_r", '<span class="math-inline">&alpha;<sub>r</sub></span>'),
    ("η_bw", '<span class="math-inline">&eta;<sub>bw</sub></span>'),
    ("η_cmp", '<span class="math-inline">&eta;<sub>cmp</sub></span>'),
    ("η_vram", '<span class="math-inline">&eta;<sub>vram</sub></span>'),
    ("ρ_conc,p95", '<span class="math-inline">&rho;<sub>p95</sub></span>'),
    ("TPOT", '<span class="math-inline">TPOT</span>'),
]


def render_math_text(text: str) -> str:
    rendered = escape(text)
    for source, target in _MATH_REPLACEMENTS:
        rendered = rendered.replace(source, target)
    rendered = re.sub(r"\bceil\(", '<span class="math-inline">ceil</span>(', rendered)
    rendered = re.sub(r"\bmin\(", '<span class="math-inline">min</span>(', rendered)
    rendered = re.sub(r"\bfloor\(", '<span class="math-inline">floor</span>(', rendered)
    rendered = re.sub(r"\bmax\(", '<span class="math-inline">max</span>(', rendered)
    return rendered


def render_calc_section_steps(section: dict[str, Any]) -> str:
    step_cards = []
    for idx, step in enumerate(section.get("steps", []), start=1):
        note_html = ""
        if step.get("note"):
            note_html = f'<div class="calc-step-note">{render_math_text(step["note"])}</div>'
        label_html = render_math_text(step["label"])
        human_formula = step.get("formula_note") or step["formula"]
        full_formula = human_formula if "=" in human_formula else f'{step["label"]} = {human_formula}'
        formula_html = render_math_text(full_formula)
        substitution_html = render_math_text(step["substitution"])
        step_cards.append(
            f"""
            <div class="calc-step-card">
              <div class="calc-step-topline">
                <span class="calc-step-index">{idx:02d}</span>
                <div class="calc-step-highlight">
                  <span class="calc-step-highlight-label">{label_html}</span>
                  <span class="calc-step-highlight-equals">=</span>
                  <span class="calc-step-highlight-result">{escape(step["result"])}</span>
                </div>
              </div>
              <div class="calc-step-main">
                <div class="calc-step-derivation">
                  <span class="calc-step-derivation-formula">{formula_html}</span>
                  <span class="calc-step-derivation-arrow">→</span>
                  <span class="calc-step-derivation-sub">{substitution_html}</span>
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
        summary_html = f'<div class="calc-section-summary-inline">{render_math_text(section["summary"])}</div>'
    step_count = len(section.get("steps", []))
    return f"""
    <div class="in-card-calc">
      <details>
        <summary>
          <span>{escape(title)}</span>
          <span class="calc-summary-meta">{step_count} steps</span>
        </summary>
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
        summary_html = f'<p class="calc-section-summary">{render_math_text(section["summary"])}</p>'
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
