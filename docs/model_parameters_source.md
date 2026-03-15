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

*   **核心依据**: [MiniMaxAI MiniMax-M2.5 (HuggingFace)](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) 与官方 M2.5 系列架构解析。
*   **参数详情**:
    *   `num_layers`: 32 (M2.5 系列规范：32层/4096维)
    *   `hidden_size`: 4096
    *   `num_heads`: 32
    *   `attention_type`: `hybrid_attention` (使用 Lightning Attention + 传统 Attention)
*   **备注**: 之前曾将其与 M2 (62层) 混淆。经核实 M2.5/M1/Text-01 系列采用了更宽但更浅的 32层/4096维 架构。已同步修正 `json` 配置文件。

---

## 4. Qwen3.5 (397B/A17B)

*   **核心依据**: [Qwen3.5-397B-A17B (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) / [Qwen Official Blog](https://qwen.ai/blog/qwen3.5-towards-native-multimodal-agents/)
*   **参数详情**:
    *   `num_layers`: 60
    *   `hidden_size`: 4096
    *   `head_dim`: 256 (依据 **Gated Attention** 规格，32 Q-heads / 2 KV-heads)
*   **技术说明**:
    *   Qwen3.5 是混合架构（Gated DeltaNet + Gated Attention）。为了显存估算的安全水位，我们采用了主导计算的 Gated Attention 规格（head_dim=256），这能覆盖最保守的显存占用情况。

---
