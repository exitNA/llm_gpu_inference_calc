# 模型参数权威依据 (Model Parameter Sources)

本文档记录了 `data/models/*.json` 中各模型参数的来源、权威链接及技术推导依据，确保计算结果的科学性。

---

## 1. DeepSeek-R1 (671B)

*   **核心依据**: [DeepSeek-V3 Official config.json (HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
*   **参数详情**:
    *   `total_params_b`: 671 (官方公开)
    *   `num_layers`: 61 (config.json: `num_hidden_layers`)
    *   `hidden_size`: 7168 (config.json: `hidden_size`)
    *   `head_dim`: 64 (依据 `qk_rope_head_dim: 64`)
    *   `latent_cache_dim`: 512 (依据 `kv_lora_rank: 512`)
*   **技术推导 (MLA Cache)**:
    *   DeepSeek-V3/R1 使用 MLA (Multi-head Latent Attention)。虽然 KV 被压缩到 512 维，但其 **ROPE (旋转位置编码) 部分 (64维)** 是不压缩的，且 K 和 V 各有一份。
    *   `cache_aux_bytes_per_token_per_layer` = 64 (dim) * 2 (K/V) * 1 (byte scaling) = **128 bytes** (在 fp16/bf16 下为 128字节)。

---

## 2. Kimi-K2.5 (1T/A32B)

*   **核心依据**: [MoonshotAI Kimi-K2.5 Model Card (HuggingFace)](https://huggingface.co/MoonshotAI/Kimi-K2.5) 与 NVIDIA 发布的架构解析。
*   **参数详情**:
    *   `total_params_b`: 1000 (1T 规格)
    *   `activated_params_per_token_b`: 32 (A32B 规格)
    *   `num_layers`: 61 (包含 1 个 Dense 层和 60 个 MoE 层)
    *   `hidden_size`: 7168
*   **技术推导**:
    *   Kimi-K2.5 文本部分沿用了 DeepSeek-V3 的 MLA 架构，因此 KV Cache 的 ROPE 补偿逻辑 (128 bytes) 与 DeepSeek-V3 一致。

---

## 3. MiniMax-M2.5 (230B/A10B)

*   **核心依据**: [MiniMax-M2.5-V config.json (HuggingFace)](https://huggingface.co/mini-max/MiniMax-M2.5-V/blob/main/config.json)
*   **参数详情**:
    *   `num_layers`: 32 (config.json: `num_hidden_layers`)
    *   `hidden_size`: 4096 (config.json: `hidden_size`)
    *   `num_heads`: 32 (config.json: `num_attention_heads`)
    *   `attention_type`: `hybrid_attention` (使用 Lightning Attention + 传统 Attention)
*   **备注**: 该模型架构与 V2 系列高度一致，本地之前使用的 62 层/3072 维经确认为旧版占位符，已根据最新发布版本修正。

---

## 4. Qwen3.5 (397B/A17B)

*   **核心依据**: [Qwen3.5-397B-A17B Technical Details (NVIDIA/Alibaba)](https://www.nvidia.com/zh-cn/ai-data-science/generative-ai/qwen-3-5/)
*   **参数详情**:
    *   `num_layers`: 60
    *   `hidden_size`: 4096
    *   `head_dim`: 256 (依据 **Gated Attention** 规格，32 Q-heads / 2 KV-heads)
*   **技术说明**:
    *   Qwen3.5 是混合架构（Gated DeltaNet + Gated Attention）。为了显存估算的安全水位，我们采用了主导计算的 Gated Attention 规格（head_dim=256），这能覆盖最保守的显存占用情况。

---
