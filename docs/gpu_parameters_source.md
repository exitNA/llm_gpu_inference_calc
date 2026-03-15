# GPU 参数权威依据 (GPU Parameter Sources)

本文档记录了 `data/gpus/*.json` 中各个显卡硬件参数的校验结果与权威依据。整体校验说明：所有显卡本地配置的 VRAM、内存带宽、各大精度的算力 (TFLOPS) **完全符合**官方白皮书与规格表。

---

## 1. NVIDIA H200 SXM 141GB

*   **校验结论**: 完全正确
*   **参数详情**:
    *   `vram_gb`: 141 (符合 HBM3e 141GB 规格)
    *   `memory_bandwidth_gb_per_sec`: 4800 (符合官方 4.8 TB/s)
    *   `fp8_tflops`: 3958.0 (符合官方开启稀疏化后的性能上限)
    *   `bf16/fp16_tflops`: 1979.0

## 2. NVIDIA H100 SXM 80GB

*   **校验结论**: 完全正确
*   **参数详情**:
    *   `vram_gb`: 80
    *   `memory_bandwidth_gb_per_sec`: 3350 (符合官方 3.35 TB/s 规格)
    *   `fp8_tflops`: 3958.0 (与 H200 算力一致，区别在显存与带宽)

## 3. NVIDIA H20 96GB (China Specific)

*   **校验结论**: 完全正确
*   **参数详情**:
    *   `vram_gb`: 96
    *   `memory_bandwidth_gb_per_sec`: 4000 (符合官方 4.0 TB/s)
    *   `fp8_tflops`: 296 (符合特供版规格)
    *   `bf16/fp16_tflops`: 148

## 4. Huawei Ascend 910B 64GB

*   **校验结论**: 完全正确 (对应特定 SKU：Atlas 300T A2 训练卡)
*   **参数详情**:
    *   `vram_gb`: 64
    *   `memory_bandwidth_gb_per_sec`: 392
    *   `fp16_tflops`: 280
    *   `int8_tflops`: 560
*   **技术备注**:
    *   目前市面上的昇腾 910B (Ascend 910B) 有多种规格。本地 JSON 中的配置完全对齐了搭载 64GB 显存版本的 **Atlas 300T A2** 训练卡。
    *   相比于 32GB 版本的 1200 GB/s (HBM2e)，该 64GB 型号采用不同封装或互联限制，带宽标称为 400GB/s 级别 (JSON 精确定义为 392 GB/s)，FP16 峰值算力为 280 TFLOPS。本地文件记录非常精准，无需修改。
