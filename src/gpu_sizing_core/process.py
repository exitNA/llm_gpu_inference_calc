from __future__ import annotations

import re
from typing import Any

from .calculations import get_peak_compute_tflops, resolve_head_dim
from .helpers import build_calc_step, format_adaptive_memory, format_adaptive_tps, format_calc_number, precision_to_bytes
from .models import GPUConfig, ModelConfig, RuntimeConfig, TrafficConfig


def _describe_cache_formula(model: ModelConfig, runtime: RuntimeConfig) -> str:
    kv_bytes = precision_to_bytes(runtime.kv_cache_dtype)
    head_dim = resolve_head_dim(model)
    attention_type = model.attention_type.lower()
    if model.cache_bytes_per_token_per_layer is not None:
        return (
            "显式输入 cache_bytes_per_token_per_layer = "
            f"{format_calc_number(model.cache_bytes_per_token_per_layer)} bytes"
        )
    if attention_type == "mha":
        return (
            "2 × hidden_size × kv_bytes = "
            f"2 × {model.hidden_size} × {format_calc_number(kv_bytes)}"
        )
    if attention_type == "gqa":
        return (
            "2 × num_kv_heads × head_dim × kv_bytes = "
            f"2 × {model.num_kv_heads} × {head_dim} × {format_calc_number(kv_bytes)}"
        )
    if attention_type == "mqa":
        return f"2 × head_dim × kv_bytes = 2 × {head_dim} × {format_calc_number(kv_bytes)}"
    if attention_type == "mla":
        return (
            "latent_cache_dim × kv_bytes + aux = "
            f"{model.latent_cache_dim} × {format_calc_number(kv_bytes)} + "
            f"{format_calc_number(model.cache_aux_bytes_per_token_per_layer)}"
        )
    return "自定义 attention cache 公式"


def build_calculation_process_sections(result: dict[str, Any]) -> list[dict[str, Any]]:
    traffic: TrafficConfig = result["traffic_config"]
    runtime: RuntimeConfig = result["runtime_config"]
    gpu: GPUConfig = result["gpu_config"]
    model: ModelConfig = result["model_config"]

    return [
        {
            "title": "业务目标折算",
            "summary": "把峰值 QPS、P95 长度和 P95 时延目标折算成 sizing 所需的吞吐与在途约束。",
            "steps": [
                build_calc_step("P95 总长度", "S_p95 = S_in,p95 + S_out,p95", f"{traffic.p95_input_tokens} + {traffic.p95_output_tokens}", f"{traffic.p95_total_tokens} tokens"),
                build_calc_step("峰值 Prefill 工作量", "TPS_pre,target^peak = λ_peak × S_in,p95", f"{format_calc_number(traffic.lambda_peak_qps)} × {traffic.p95_input_tokens}", f"{format_adaptive_tps(result['tps_pre_target_peak'])}"),
                build_calc_step("峰值 Decode 工作量", "TPS_dec,target^peak = λ_peak × S_out,p95", f"{format_calc_number(traffic.lambda_peak_qps)} × {traffic.p95_output_tokens}", f"{format_adaptive_tps(result['tps_dec_target_peak'])}"),
                build_calc_step("峰值在途预算", "C_peak^budget = λ_peak × E2E_p95 × 安全系数", f"{format_calc_number(traffic.lambda_peak_qps)} × {format_calc_number(traffic.e2e_p95_target_sec)} × {format_calc_number(traffic.concurrency_safety_factor)}", f"{format_calc_number(result['c_peak_budget'])} req"),
                build_calc_step("P95 Decode 时间预算", "T_dec,p95 = E2E_p95 - TTFT_p95", f"{format_calc_number(traffic.e2e_p95_target_sec)} - {format_calc_number(traffic.ttft_p95_target_sec)}", f"{format_calc_number(result['decode_p95_budget_sec'])} s"),
            ],
        },
        {
            "title": "显存约束",
            "summary": "按权重、固定运行时开销和峰值在途请求的 cache 显存估算总显存，再折算卡数。",
            "steps": [
                build_calc_step("原始权重显存", "P_total × b_w", f"{format_calc_number(model.num_params_billion)}e9 × {format_calc_number(precision_to_bytes(runtime.precision))}", format_adaptive_memory(result["raw_weight_bytes"])),
                build_calc_step("权重显存 Mw", "Mw = P_total × b_w × (1 + α_w)", f"{format_calc_number(model.num_params_billion)}e9 × {format_calc_number(precision_to_bytes(runtime.precision))} × (1 + {format_calc_number(runtime.weight_overhead_ratio, 2)})", format_adaptive_memory(result["weight_bytes"])),
                build_calc_step("运行时固定显存 Mr", "Mr = α_r × Mw", f"{format_calc_number(runtime.runtime_overhead_ratio, 2)} × {format_adaptive_memory(result['weight_bytes'])}", format_adaptive_memory(result["runtime_fixed_bytes"])),
                build_calc_step("单请求 P95 Cache", "M_cache^req(S_p95) = L × S_p95 × 每 token 每层 cache 字节数", f"{model.num_layers} × {traffic.p95_total_tokens} × ({_describe_cache_formula(model, runtime)})", format_adaptive_memory(result["p95_cache_bytes_per_request"])),
                build_calc_step("总 Cache 显存", "M_cache = C_peak^budget × M_cache^req(S_p95)", f"{format_calc_number(result['c_peak_budget'])} × {format_adaptive_memory(result['p95_cache_bytes_per_request'])}", format_adaptive_memory(result["cache_total_peak_bytes"])),
                build_calc_step("单卡有效显存", "V_gpu^eff = V_gpu × η_vram", f"{format_calc_number(gpu.vram_gb)} GB × {format_calc_number(runtime.usable_vram_ratio, 2)}", format_adaptive_memory(result["usable_vram_bytes_per_gpu"])),
                build_calc_step("显存约束卡数 G_mem", "G_mem = ceil((Mw + Mr + M_cache) / (V_gpu × η_vram))", f"ceil({format_adaptive_memory(result['total_memory_bytes'])} / {format_adaptive_memory(result['usable_vram_bytes_per_gpu'])})", f"{result['gpu_count_by_memory']} 卡"),
            ],
        },
        {
            "title": "吞吐与时延必要条件",
            "summary": "分别估算单卡 Prefill/Decode 吞吐，再检查 P95 TTFT 与 Decode 预算是否在单卡能力层面成立。",
            "steps": [
                build_calc_step("Prefill 带宽上界", "TPS_pre,bw^card = B_mem × η_bw / b_pre(S_in,p95)", f"{format_calc_number(gpu.memory_bandwidth_gb_per_sec)} GB/s × {format_calc_number(runtime.bandwidth_efficiency, 2)} / {format_adaptive_memory(result['prefill_bytes_per_token_p95'])}", format_adaptive_tps(result["prefill_tps_p95_bw_limited"])),
                build_calc_step("Prefill 算力上界", "TPS_pre,cmp^card = F_peak × η_cmp / (2P_act + α_attn × S_in,p95)", f"{format_calc_number(get_peak_compute_tflops(gpu, runtime.precision))} TFLOPS × {format_calc_number(runtime.compute_efficiency, 2)} / (2P_act + {format_calc_number(result['prefill_flops_attention_coeff'])} × {traffic.p95_input_tokens})", format_adaptive_tps(result["prefill_tps_p95_compute_limited"])),
                build_calc_step("单卡 Prefill 吞吐", "TPS_pre^card = min(TPS_pre,bw^card, TPS_pre,cmp^card)", f"min({format_adaptive_tps(result['prefill_tps_p95_bw_limited'])}, {format_adaptive_tps(result['prefill_tps_p95_compute_limited'])})", format_adaptive_tps(result["prefill_tps_p95_card"]), note="Prefill 使用 S_in,p95 作为保守长度口径。"),
                build_calc_step("Decode 带宽上界", "TPS_dec,bw^card = B_mem × η_bw / b_dec", f"{format_calc_number(gpu.memory_bandwidth_gb_per_sec)} GB/s × {format_calc_number(runtime.bandwidth_efficiency, 2)} / {format_adaptive_memory(result['decode_bytes_per_token'])}", format_adaptive_tps(result["decode_tps_bw_limited"])),
                build_calc_step("Decode 算力上界", "TPS_dec,cmp^card = F_peak × η_cmp / (2P_act)", f"{format_calc_number(get_peak_compute_tflops(gpu, runtime.precision))} TFLOPS × {format_calc_number(runtime.compute_efficiency, 2)} / {format_calc_number(result['decode_flops_per_token'])} FLOPs", format_adaptive_tps(result["decode_tps_compute_limited"])),
                build_calc_step("单卡 Decode 吞吐", "TPS_dec^card = min(TPS_dec,bw^card, TPS_dec,cmp^card)", f"min({format_adaptive_tps(result['decode_tps_bw_limited'])}, {format_adaptive_tps(result['decode_tps_compute_limited'])})", format_adaptive_tps(result["decode_tps_card"])),
                build_calc_step("Prefill 吞吐卡数 G_pre", "G_pre = ceil(TPS_pre,target^peak / TPS_pre^card)", f"ceil({format_adaptive_tps(result['tps_pre_target_peak'])} / {format_adaptive_tps(result['prefill_tps_p95_card'])})", f"{result['prefill_gpu_count_by_throughput']} 卡"),
                build_calc_step("Decode 吞吐卡数 G_dec", "G_dec = ceil(TPS_dec,target^peak / TPS_dec^card)", f"ceil({format_adaptive_tps(result['tps_dec_target_peak'])} / {format_adaptive_tps(result['decode_tps_card'])})", f"{result['decode_gpu_count_by_throughput']} 卡"),
                build_calc_step("Prefill 时延必要条件", "S_in,p95 / TPS_pre^card <= TTFT_p95,target", f"{traffic.p95_input_tokens} / {format_calc_number(result['prefill_tps_p95_card'])} <= {format_calc_number(traffic.ttft_p95_target_sec)}", f"{format_calc_number(result['prefill_latency_check_sec'])} s -> {'满足' if result['prefill_latency_ok'] else '不满足'}"),
                build_calc_step("Decode 时延必要条件", "S_out,p95 / TPS_dec^card <= E2E_p95 - TTFT_p95", f"{traffic.p95_output_tokens} / {format_calc_number(result['decode_tps_card'])} <= {format_calc_number(result['decode_p95_budget_sec'])}", f"{format_calc_number(result['decode_latency_check_sec'])} s -> {'满足' if result['decode_latency_ok'] else '不满足'}"),
                build_calc_step("理论 Decode TPOT", "TPOT = 1000 / TPS_dec^card", f"1000 / {format_calc_number(result['decode_tps_card'])}", f"{format_calc_number(result['decode_ms_per_token'])} ms/token"),
            ],
        },
        {
            "title": "最终卡数与能力回推",
            "summary": "取 G_mem、G_pre、G_dec 最大值作为业务基线，再回推保守可持续 QPS 与显存并发余量。",
            "steps": [
                build_calc_step("业务基线卡数 G_req", "G_req = max(G_mem, G_pre, G_dec)", f"max({result['gpu_count_by_memory']}, {result['prefill_gpu_count_by_throughput']}, {result['decode_gpu_count_by_throughput']})", f"{result['business_gpu_count']} 卡"),
                build_calc_step("P95 总 Prefill 吞吐能力", "TPS_pre,p95^cap = G_req × TPS_pre^card(S_in,p95)", f"{result['business_gpu_count']} × {format_adaptive_tps(result['prefill_tps_p95_card'])}", format_adaptive_tps(result["cluster_prefill_tps_capacity_p95"])),
                build_calc_step("总 Decode 吞吐能力", "TPS_dec^cap = G_req × TPS_dec^card", f"{result['business_gpu_count']} × {format_adaptive_tps(result['decode_tps_card'])}", format_adaptive_tps(result["cluster_decode_tps_capacity"])),
                build_calc_step("保守可持续 QPS", "λ_p95^sus = min(TPS_pre,p95^cap / S_in,p95, TPS_dec^cap / S_out,p95)", f"min({format_adaptive_tps(result['cluster_prefill_tps_capacity_p95'])} / {traffic.p95_input_tokens}, {format_adaptive_tps(result['cluster_decode_tps_capacity'])} / {traffic.p95_output_tokens})", f"{format_calc_number(result['sustainable_qps_p95'])} req/s"),
                build_calc_step("保守日请求量", "Req_day^p95 = λ_p95^sus × 86400", f"{format_calc_number(result['sustainable_qps_p95'])} × 86400", f"{format_calc_number(result['daily_request_capacity_p95'])} req/day"),
                build_calc_step("显存最大在途量", "C_max,p95^mem = floor(V_cache^avail / M_cache^req(S_p95))", f"floor({format_adaptive_memory(result['cache_available_bytes'])} / {format_adaptive_memory(result['p95_cache_bytes_per_request'])})", f"{result['max_concurrency_by_memory_p95']} req"),
                build_calc_step("并发余量系数", "ρ_conc,p95 = C_max,p95^mem / C_peak^budget", f"{format_calc_number(result['max_concurrency_by_memory_p95'])} / {format_calc_number(result['c_peak_budget'])}", f"{format_calc_number(result['concurrency_margin_ratio_p95'])}"),
            ],
        },
    ]


_COMPACT_FORMULA_REPLACEMENTS: list[tuple[str, str]] = [
    ("TPS_pre,target^peak", "Pre_peak"),
    ("TPS_dec,target^peak", "Dec_peak"),
    ("TPS_pre,p95^cap", "Pre_cap95"),
    ("TPS_dec^cap", "Dec_cap"),
    ("TPS_pre,bw^card", "Pre_bw"),
    ("TPS_pre,cmp^card", "Pre_cmp"),
    ("TPS_pre^card", "Pre_card"),
    ("TPS_dec,bw^card", "Dec_bw"),
    ("TPS_dec,cmp^card", "Dec_cmp"),
    ("TPS_dec^card", "Dec_card"),
    ("C_max,p95^mem", "C_mem95"),
    ("C_peak^budget", "C_peak"),
    ("M_cache^req(S_p95)", "M_req95"),
    ("M_cache", "M_cache"),
    ("V_cache^avail", "V_cache"),
    ("V_gpu^eff", "V_gpu_eff"),
    ("T_dec,p95", "T_dec95"),
    ("Req_day^p95", "Req_day95"),
    ("λ_p95^sus", "QPS95"),
    ("λ_peak", "QPSpeak"),
    ("S_in,p95", "Sin95"),
    ("S_out,p95", "Sout95"),
    ("S_p95", "S95"),
    ("TTFT_p95,target", "TTFT95"),
    ("TTFT_p95", "TTFT95"),
    ("E2E_p95", "E2E95"),
]


def _compact_formula(formula: str) -> str:
    compact = formula
    for source, target in _COMPACT_FORMULA_REPLACEMENTS:
        compact = compact.replace(source, target)
    compact = compact.replace("TPS_pre", "Pre")
    compact = compact.replace("TPS_dec", "Dec")
    compact = compact.replace("P_act", "Pact")
    compact = compact.replace("B_mem", "Bmem")
    compact = compact.replace("F_peak", "Fpeak")
    compact = compact.replace("α_attn", "a_attn")
    compact = compact.replace("η_bw", "eta_bw")
    compact = compact.replace("η_cmp", "eta_cmp")
    compact = compact.replace("η_vram", "eta_vram")
    compact = compact.replace("ρ_conc,p95", "rho95")
    return compact


def _strip_units(text: str) -> str:
    stripped = re.sub(r"\s+(tok/s|req/s|req/day|ms/token|tokens|req|GB/s|TFLOPS|FLOPs|GB|MB|KB|s)\b", "", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def format_calculation_process_text(sections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(f"[{section['title']}]")
        for step in section["steps"]:
            compact_formula = _compact_formula(step["formula"])
            compact_substitution = _strip_units(step["substitution"])
            line = (
                f"- {step['label']}: {compact_formula} -> {compact_substitution} -> {step['result']}"
            )
            if step.get("note"):
                line = f"{line} ({step['note']})"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).strip()
