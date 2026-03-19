# DeepSeek-R1 实际部署与测试结果摘录

## 1. 文档目标

本文整理公开可查的 DeepSeek-R1 满血版（671B / 37B activated）部署与测试资料，作为本仓库容量估算的经验参考。

本文只收集两类信息：

- **部署建议**：某个推理框架或厂商给出的推荐硬件与并行配置；
- **实测结果**：带有明确硬件、软件栈、请求长度或并发条件的公开 benchmark / 生产实践数据。

为尽量保证可信度，本文优先采用以下来源：

- 官方文档；
- 官方博客；
- 官方 GitHub issue / benchmark；
- 框架核心团队或联合作者发布的技术博客。

## 2. 阅读方式

在阅读公开 benchmark 时，需要特别注意以下几点：

1. **吞吐指标必须结合长度口径一起看**。`tokens/s` 在 `760/460`、`1000/2000`、`4096/1536` 这几类工作负载下不可直接横向比较。
2. **栈不同，结果不可直接等价**。`SGLang`、`NVIDIA NIM`、`TensorRT-LLM`、`vLLM` 的 kernel、调度、并行策略不同，吞吐和时延会有显著差异。
3. **部署建议不等于实测峰值**。官方“推荐配置”通常是可运行且较稳妥的起点，不代表该配置已经给出了完整 benchmark。
4. **推理分阶段拆分时，要分别看 Prefill 和 Decode**。对于 DeepSeek-R1 这类 MoE 推理，很多生产实践已经把两阶段拆开部署，因此单个“总吞吐”数字往往掩盖了真实瓶颈。

## 3. 公开资料汇总

### 3.1 SGLang 官方文档：推荐部署配置

SGLang 当前文档把 DeepSeek V3/V3.1/R1 归为同一类推荐配置，给出的结论是：

- **Full precision FP8（推荐）**：`8 x H200`、`8 x MI300X`、`2 x 8 x H100/800/20`；
- **Full precision BF16**：`2 x 8 x H200`、`2 x 8 x MI300X`、`4 x 8 x H100/800/20`、`4 x 8 x A100/A800`；
- 文档同时明确给出了 **“Serving with two H20*8 nodes”** 的多机示例。

这组信息更像“当前官方推荐的可运行部署口径”，而不是严格 benchmark。

对本仓库的含义是：

- 若希望在 **FP8 满血版** 下较稳妥地运行 DeepSeek-R1，`16 x H20` 是主流公开推荐口径之一；
- 若只有 `8 x H20`，则通常需要更激进的量化、截断上下文、降低并发，或者接受更保守的 SLA。

来源：

- SGLang 文档，`DeepSeek V3/V3.1/R1 Usage`
  https://docs.sglang.ai/basic_usage/deepseek_v3.html

### 3.2 SGLang 官方 Issue #3956：2 节点 H20 实测

SGLang 官方仓库 issue `#3956` 给出了一组非常有价值的 H20 实测数据，属于公开的一手 benchmark。

测试环境摘要：

- 更新时间：**2025-02-28**
- 软件：`SGLang 0.4.3.post2 / 0.4.3 master`
- 硬件：**2 nodes of H20**，每节点 `8 x H20 96GiB`
- 模型：`DeepSeek-R1`
- `model_max_length`：修改为 `3200`
- Batched benchmark 口径：
  - 平均输入长度：`760 tokens`
  - 平均输出长度：`460 tokens`

文中可直接读到的代表性结果包括：

- 在一组开启 `Torch Compile + Cuda Graph + Radix Cache + Flashinfer-mla` 的配置下，客户端并发 `768` 时，**Avg Throughput = 3909.04 token/s**；
- 打开 `DP-Attn` 后，不同并发下给出了 **306.18 / 4329.32 / 5457.14 token/s** 的平均吞吐结果；
- 单请求 `TPS@1` 的有效结果中，可见 `22.4`、`24.4`、`40.0 token/s` 等数值；
- 但同一 issue 也明确记录了若干失败或不稳定情况，包括：
  - `gibberish`
  - `OOM`
  - `CUDA error at graph capture`

这组数据很重要，因为它同时说明了两件事：

1. `16 x H20` 跑满血版 DeepSeek-R1 是公开可复现的；
2. 该配置下的性能高度依赖 kernel、精度稳定性、`mem-fraction-static`、DP-Attention 和其他优化组合，不是“裸跑即可稳定达到”。

来源：

- SGLang Issue #3956
  https://github.com/sgl-project/sglang/issues/3956

### 3.3 NVIDIA 官方：HGX H200 上的 DeepSeek-R1 NIM

NVIDIA 官方博客给出了一条更偏“成品软件栈”的结果：

- 使用 `NVIDIA NIM microservice`
- 在 **单台 NVIDIA HGX H200（8 x H200）**
- 满血版 `DeepSeek-R1 671B`
- 可达到 **up to 3,872 tokens/s**

这条数据的价值在于：

- 它来自 NVIDIA 官方软件栈，说明 `8 x H200` 已可实现较高吞吐；
- 但文中没有给出与 SGLang issue 同等细粒度的输入输出长度、并发或 TTFT/TPOT 条件，因此它更适合作为**上层系统能力参考**，而非直接拿来校准本仓库中的 `Prefill/Decode` 细分模型。

来源：

- NVIDIA Blog, `DeepSeek-R1 Now Live With NVIDIA NIM`
  https://blogs.nvidia.com/blog/deepseek-r1-nim-microservice/

### 3.4 LMSYS + Ant Group：H20-96G 生产实践

LMSYS 在 **2025-09-26** 发布了与 Ant Group 合作的技术文章 `Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G`，这是当前公开资料里最接近“生产实践复盘”的一篇。

文章的关键信息包括：

- 目标：在 **H20 96G** 上满足真实在线 SLA；
- 部署策略：
  - **Prefill**：单节点 `8 x H20`，`TP-8`
  - **Decode**：双节点 `16 x H20`，`DP16 + EP16`
- 文章给出的总结性结果：
  - **每节点达到 16.5k input tokens/s**
  - **每节点达到 5.7k output tokens/s**
  - 工作负载口径写明为 `4096-token input sequences`
- 在线服务层面的 Decode 配置还给出分级 SLA 数字：
  - `InferX Base`：`714 tokens/s / GPU`
  - `InferX Pro`：`675 tokens/s / GPU`
  - `InferX Max`：`423 tokens/s / GPU`

这篇文章非常值得保留，因为它补上了一个常见盲点：

- 对 H20 这类“带宽较强、算力较弱”的卡，**Prefill 和 Decode 不一定适合使用同一种并行切分方式**；
- 真正的在线系统往往已经采用 **Prefill 单节点、Decode 双节点** 的拆分部署，而不是单一 `TP=16` 或单一 `EP=16` 一把梭。

来源：

- LMSYS Org, `Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G`
  https://lmsys.org/blog/2025-09-26-sglang-ant-group/

## 4. 一页表汇总

| 来源 | 类型 | 硬件 | 主要口径 | 公开结果 | 备注 |
| --- | --- | --- | --- | --- | --- |
| SGLang 文档 | 官方部署建议 | `2 x 8 x H100/800/20` 或 `8 x H200` | DeepSeek V3/V3.1/R1 FP8 推荐配置 | 给出推荐硬件与多机示例 | 不是严格 benchmark |
| SGLang Issue #3956 | 官方公开 benchmark | `2 x 8 x H20 96GiB` | Avg in/out=`760/460`，并发最高到 `768` | `3909.04 tok/s`，部分配置 `4329.32 / 5457.14 tok/s`，`TPS@1` 有 `22.4 / 24.4 / 40.0` | 同时暴露了 OOM、graph capture、精度异常等问题 |
| NVIDIA NIM 博客 | 官方系统级结果 | `8 x H200` | NIM microservice，长度口径未公开 | `up to 3,872 tok/s` | 更适合作为整栈能力参考 |
| LMSYS + Ant Group | 生产实践复盘 | Prefill=`8 x H20`；Decode=`16 x H20` | `4096` 输入、分阶段部署、按 SLA 调优 | 每节点 `16.5k input tok/s`、`5.7k output tok/s`；Decode `423~714 tok/s/GPU` | 最接近真实线上推理实践 |

## 5. 与本仓库 sizing 的关系

对于本仓库当前的估算逻辑，这些外部资料最有用的不是“抄一个固定卡数”，而是用于校验以下三点：

1. **显存下界是否离谱**
   如果你的估算结果明显低于公开部署建议，例如算出 DeepSeek-R1 FP8 只需 `8 x H20` 就能在较舒适口径下支撑在线服务，那通常需要重新检查：
   - 权重显存口径；
   - KV cache 口径；
   - 是否错误地把峰值在途估得过低。

2. **Decode 吞吐是否偏乐观**
   公开资料普遍显示，满血版 DeepSeek-R1 的线上部署里，**Decode 是核心瓶颈**。如果 sizing 结果里 `G_dec` 长期远低于 `G_mem`，而工作负载又是长输出场景，往往说明 `decode_tps_card` 假设过乐观。

3. **是否需要分阶段部署**
   LMSYS/Ant Group 的 H20 实践已经说明，在线系统可能需要把 Prefill 和 Decode 分别组织。当前仓库的模型仍以“统一卡池总量估算”为主，因此适合采购前容量规划，不应被直接解读为生产拓扑设计方案。

## 6. 使用建议

如果你准备把这些资料用于容量评审，建议按以下顺序使用：

1. 先用本仓库按业务画像算出 `G_mem / G_pre / G_dec`；
2. 再拿本文中的公开部署资料检查量级是否合理；
3. 若你的结果明显低于公开实践，优先复查：
   - 输出长度假设；
   - 峰值 QPS / 峰值在途建模方式；
   - `η_bw`、`η_cmp`、`η_vram` 是否取值过乐观；
4. 若你已经接近 `16 x H20` 这类公开实践量级，则更应关注：
   - 单机 / 双机 fault domain；
   - 是否拆分 Prefill 与 Decode；
   - TTFT / TPOT 是否满足真实 SLA。

## 7. 参考链接

- SGLang 文档
  https://docs.sglang.ai/basic_usage/deepseek_v3.html
- SGLang Issue #3956
  https://github.com/sgl-project/sglang/issues/3956
- NVIDIA Blog: DeepSeek-R1 NIM
  https://blogs.nvidia.com/blog/deepseek-r1-nim-microservice/
- LMSYS + Ant Group: H20 最佳实践
  https://lmsys.org/blog/2025-09-26-sglang-ant-group/
