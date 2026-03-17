# 大模型在线推理 GPU 资源测算方法

---

## 1. 文档目标

本文用于在线推理场景下的大模型 GPU 资源测算，给出一套从**输入条件**出发，经过**部署方案求解**，最终输出**GPU 数量与预期可达到效果**的完整方法。

本文目标不是复现精确 benchmark，而是提供一套：

- 逻辑自洽
- 公式口径统一
- 可工程实现
- 可用于前期 sizing / 方案比较 / 容量规划 / 采购前预算评估

的估算方法。

本文输出：

- 推荐部署方案
- 最小服务单元（实例）需要的 GPU 数 $instance\_gpus$
- 是否需要分片 $is\_sharded$
- 部署模式 $deployment\_mode$
- 业务所需 GPU 数 $G_{biz}$
- 考虑高可用后的最终 GPU 数 $G_{final}$
- 基于已推导 GPU 数可达到的并发、吞吐、生成速度、时延等预期效果

本文强调：

- 核心用途是**采购前 GPU 资源预估**
- 用户不需要预先给出“单卡完整装载”还是“多卡分片装载”
- 这些是部署测算过程中的**自动推断结果**，而不是用户输入

---

## 2. 适用范围与方法边界

### 2.1 适用范围

支持以下模型与结构的前期估算：

- Dense / MoE
- MHA / GQA / MQA / MLA / Hybrid Attention
- 在线推理场景下的 prefill / decode 分阶段容量估算
- 基于 TTFT / E2E / 并发目标反推资源规模
- 基于模型、GPU、框架候选进行部署方案比较

### 2.2 方法边界

本文属于**容量规划方法**，不等价于线上精确 SLA 保证或框架 benchmark。以下内容默认采用工程近似：

1. 横向扩副本时吞吐近似线性增长。
2. 单实例内部 TP / EP / PP 的通信开销只通过实例级扩展效率近似吸收，不做精细链路建模。
3. Prefill 吞吐在缺少到达率信息时采用保守上界近似。
4. 平均 / P95 对话时延反推属于 sanity check，不构成 SLA 证明。
5. 多维 P95 长度拼装属于保守 sizing，而非严格联合分布建模。
6. 多卡实例的显存可行性，在采购前阶段按**实例级 pooled/sharded 可行性上界**处理，不等价于 replicated 多卡实例的严格显存证明。
7. 文中核心推导统一使用 $byte$、$byte/s$、$FLOP$、$FLOP/s$、$token$、$second$；仅在结果展示层转换为人类易读单位。

若需要更高精度建模，应补充：

- 多卡实例内通信拓扑与并行方式
- 请求到达率与 burst 窗口
- request shape 联合分布
- 真实线上 profiling 或 benchmark 数据

---

## 3. 总体方法框架

本文采用四层结构：

```mermaid
flowchart LR
    A[输入条件]
    B[部署方案求解]
    C[资源规模输出]
    D[效果反推与校验]

    A --> B --> C --> D
```

其中：

### 第 1 层：输入条件

输入包括：

- 业务目标输入
- 模型输入
- GPU 选择输入
- 推理框架 / 部署档位输入
- HA 输入

### 第 2 层：部署方案求解

根据输入条件，自动推导：

- 单卡是否可完整装载
- 最小服务单元由几张卡组成
- 是否需要分片
- 单实例显存与吞吐能力
- 故障单元规模

### 第 3 层：资源规模输出

在部署方案确定后，计算：

- 显存约束实例数 / GPU 数
- Prefill 吞吐约束实例数 / GPU 数
- Decode 吞吐约束实例数 / GPU 数
- 业务所需最小 GPU 数 $G_{biz}$
- 高可用后最终 GPU 数 $G_{final}$

### 第 4 层：效果反推与校验

基于已推导的 GPU 数，反推：

- 可持续并发能力
- 可用吞吐上限
- 单请求生成速度
- 平均 / 保守 P95 时延近似
- 每日产能

说明：本层为工程近似与 sanity check，不作为采购结论的唯一依据。

---

## 4. 统一术语与记号

### 4.1 基本对象

- **GPU**：单张物理卡
- **实例（instance）**：一个可独立提供推理服务的最小服务单元
- **副本（replica）**：多个同构实例组成的横向扩展副本集合
- **集群（cluster）**：所有用于承载业务的实例总和

### 4.2 关键部署概念

- $instance\_gpus$：单实例占用的 GPU 数
- $instance\_count$：业务实例数
- $G_{total} = instance\_count \times instance\_gpus$
- $is\_sharded$：单实例内部是否需要跨卡切分模型 / 缓存
- $deployment\_mode$：$single\_gpu\_full$ 或 $multi\_gpu\_sharded$
- $failure\_unit\_gpus$：一次最小故障会损失的 GPU 数
- $memory\_feasibility\_type$：$strict$ 或 $sharded\_upper$

### 4.3 关键结论

本文不将以下量视为原始输入，而将其视为**部署求解结果**：

- $instance\_gpus$
- $is\_sharded$
- $deployment\_mode$
- $failure\_unit\_gpus$
- $memory\_feasibility\_type$

### 4.4 基础单位约定与辅助函数

除最终展示外，全文中间计算统一采用：

- 存储、显存、带宽：$byte$、$byte/s$
- token 长度与吞吐：$token$、$token/s$
- 计算量与算力：$FLOP$、$FLOP/s$
- 时间：$second$

辅助函数只用于承载**确定映射**，不吸收任何经验假设：

- $bytes\_per\_param(inference\_precision)$：每参数字节数
- $bytes\_per\_cache\_element(kv\_cache\_dtype)$：KV cache 元素字节数
- $peak\_compute\_flops\_per\_sec(inference\_precision)$：按推理精度选取 GPU 对应峰值算力，单位 $FLOP/s$

### 4.5 记号映射

- $S_{in,avg}$：平均输入长度，单位 $token$
- $S_{out,avg}$：平均输出长度，单位 $token$
- $S_{in,p95}$：P95 输入长度，单位 $token$
- $S_{out,p95}$：P95 输出长度，单位 $token$
- $S_{avg}$：平均总长度，单位 $token$
- $S_{p95}$：P95 总长度，单位 $token$
- $C_{avg}$：平均活跃并发
- $C_{peak}$：峰值活跃并发
- $r_{decode,avg}$：平均窗口 decode-active 比例
- $r_{decode,peak}$：峰值窗口 decode-active 比例

### 4.6 工程实现中的建议字段别名

为便于落地为配置文件、脚本或服务接口，建议在工程实现中使用更短的字段别名；文档正文仍保留可读性更高的正式记号。

| 正式记号 / 字段 | 建议工程别名 |
| --- | --- |
| `concurrency_avg` | `cc_avg` |
| `concurrency_peak` | `cc_peak` |
| `target_ttft_avg_sec` | `ttft_avg_s` |
| `target_ttft_p95_sec` | `ttft_p95_s` |
| `target_e2e_avg_sec` | `e2e_avg_s` |
| `target_e2e_p95_sec` | `e2e_p95_s` |
| `target_total_tokens_p95` | `tok_p95` |
| `decode_active_ratio_avg` | `dec_act_avg` |
| `decode_active_ratio_peak` | `dec_act_peak` |
| `memory_bandwidth_efficiency` | `bw_eff` |
| `compute_efficiency` | `cmp_eff` |
| `usable_vram_ratio` | `vram_use` |
| `weight_overhead_ratio` | `w_ovhd` |
| `runtime_buffer_ratio` | `rt_buf` |
| `inst_scale_efficiency` | `inst_eff` |
| `instance_gpus` | `gpus_per_inst` |
| `instance_count` | `inst_cnt` |
| `failure_unit_gpus` | `fail_unit_gpus` |

说明：

- 正式文档、评审材料与对外交付建议继续使用正式字段名。
- 工程别名只用于配置、代码与可视化表格，不应改变公式含义。

---

## 5. 输入条件

### 5.1 输入分类总表

| 分类 | 是否必填 | 说明 |
| --- | --- | --- |
| 业务目标输入 | 是 | 定义业务规模与体验目标 |
| 模型输入 | 是 | 定义模型结构、规模与精度 |
| GPU 选择输入 | 是 | 定义 GPU 规格与有效利用率假设 |
| 推理框架 / 部署档位输入 | 是 | 定义框架选择及少量必要画像参数 |
| HA 输入 | 是 | 定义高可用目标 |

### 5.2 业务目标输入（必填）

| 参数 | 含义 |
| --- | --- |
| `concurrency_avg` | 平均活跃并发 |
| `concurrency_peak` | 峰值活跃并发 |
| `target_ttft_avg_sec` | 平均首字延迟目标 |
| `target_ttft_p95_sec` | P95 首字延迟目标 |
| `target_e2e_avg_sec` | 平均单次对话总耗时目标 |
| `target_e2e_p95_sec` | P95 单次对话总耗时目标 |
| `request_shapes` | 请求长度分布，来自日志汇总 |
| `decode_active_ratio_avg` | 平均窗口 decode-active 比例 |
| `decode_active_ratio_peak` | 峰值窗口 decode-active 比例 |
| `target_total_tokens_p95` | P95 总 token 长度，选填，提供时优先使用 |

#### 5.2.1 `request_shapes`（最小版，必填）

```json
{
  "request_shapes": [
    {
      "name": "short_qa",
      "weight": 0.6,
      "input_tokens_avg": 300,
      "input_tokens_p95": 800,
      "output_tokens_avg": 120,
      "output_tokens_p95": 300
    },
    {
      "name": "long_context_summary",
      "weight": 0.4,
      "input_tokens_avg": 5000,
      "input_tokens_p95": 10000,
      "output_tokens_avg": 400,
      "output_tokens_p95": 900
    }
  ]
}
```

约束：

- 所有 $weight_i > 0$
- $\sum weight_i = 1$
- 每个 shape 至少包含 `name / weight / input_tokens_avg / input_tokens_p95 / output_tokens_avg / output_tokens_p95`

推荐默认值仅用于跑通首版 sizing；若已有日志，应直接由日志统计产出。

#### 5.2.2 `decode_active_ratio_avg` 与 `decode_active_ratio_peak`（必填）

定义域：

$$
0 < decode\_active\_ratio\_avg \le 1
$$

$$
0 < decode\_active\_ratio\_peak \le 1
$$

推荐原则：

- 优先由日志窗口统计得到
- 至少拆分平均与峰值两个口径
- 不建议继续使用单一 $r_{decode}$ 同时支配平均和峰值计算

#### 5.2.3 `target_total_tokens_p95`（选填，强烈建议）

用途：

- 若业务侧可直接给出真实的 P95 总长度，应优先使用
- 若未提供，则采用保守近似 $S_{p95} = S_{in,p95} + S_{out,p95}$

### 5.3 模型输入（必填）

| 参数 | 含义 |
| --- | --- |
| `model_name` | 模型名称，可选 |
| `arch_family` | `Dense` / `MoE` |
| `attention_type` | `MHA` / `GQA` / `MQA` / `MLA` / `Hybrid` |
| `total_params_b` | 总参数量，单位 B |
| `activated_params_per_token_b` | 每 token 激活参数量，MoE 使用 |
| `num_layers` | 层数 |
| `hidden_size` | 隐藏维度 |
| `num_heads` | Query heads 数 |
| `num_kv_heads` | KV heads 数 |
| `head_dim` | 每个 head 维度 |
| `latent_cache_dim` | MLA latent cache 维度 |
| `max_context_window` | 最大上下文窗口 |
| `inference_precision` | 权重推理精度 |
| `kv_cache_dtype` | KV Cache 精度 |
| `cache_aux_bytes_per_token_per_layer` | 每 token 每层额外 cache 开销，单位 $byte$ |
| `cache_bytes_per_token_per_layer` | 直接指定 cache 字节数，单位 $byte$，优先级最高 |

### 5.4 GPU 选择输入（必填）

| 参数 | 含义 |
| --- | --- |
| `gpu_name` | GPU 名称 |
| `vram_bytes` | 单卡显存，单位 $byte$ |
| `memory_bandwidth_bytes_per_sec` | HBM 带宽，单位 $byte/s$ |
| `fp16_flops_per_sec` | FP16 峰值算力，单位 $FLOP/s$ |
| `bf16_flops_per_sec` | BF16 峰值算力，单位 $FLOP/s$ |
| `fp8_flops_per_sec` | FP8 峰值算力，单位 $FLOP/s$ |
| `int8_flops_per_sec` | INT8 峰值算力，单位 $FLOP/s$ |
| `int4_flops_per_sec` | INT4 / W4A16 峰值算力，单位 $FLOP/s$ |
| `memory_bandwidth_efficiency` | 有效带宽折减系数 |
| `compute_efficiency` | 有效算力折减系数 |
| `usable_vram_ratio` | 单卡可用显存比例 |

推荐默认值（仅无实测时作为采购前基线）：

- `memory_bandwidth_efficiency = 0.70`
- `compute_efficiency = 0.35`
- `usable_vram_ratio = 0.90`

### 5.5 推理框架 / 部署档位输入（必填）

| 参数 | 含义 |
| --- | --- |
| `serving_framework` | `vLLM` / `SGLang` / `TensorRT-LLM` / 自研 |
| `framework_version` | 版本 |
| `weight_overhead_ratio` | 权重附加显存系数 |
| `runtime_buffer_ratio` | 运行时固定显存系数 |
| `inst_scale_efficiency` | 多卡实例吞吐扩展效率；单卡实例固定为 $1.0$ |

推荐默认值：

- `weight_overhead_ratio = 0.05`
- `runtime_buffer_ratio = 0.10`
- `inst_scale_efficiency = 1.00`（单卡实例）
- `inst_scale_efficiency = 0.85`（多卡实例且无实测值时的推荐起点）

说明：

- `weight_overhead_ratio` 只吸收**权重静态装载后**与权重驻留直接相关的额外显存，例如权重格式、量化元数据、内存布局与对齐附加开销。
- `runtime_buffer_ratio` 只吸收**运行时长期占用**的固定显存，例如 workspace、allocator 预留、运行时 buffer 等常驻开销。
- `usable_vram_ratio` 只吸收 OOM 安全水位、波动预留与运营安全边界，不应与前两者重复计入。
- `weight_overhead_ratio` 与 `runtime_buffer_ratio` 概念上应显式区分；前者属于权重层附加，后者属于运行时层附加。
- `inst_scale_efficiency` 仅在 $instance\_gpus > 1$ 时参与吞吐计算，用于吸收跨卡通信与并行切分导致的非线性扩展损失。

#### 5.5.1 采购前推荐档位

若缺少实测值，建议至少同时评估保守档与中性档，避免把单点默认值误当真值。

| 档位 | `memory_bandwidth_efficiency` | `compute_efficiency` | `weight_overhead_ratio` | `runtime_buffer_ratio` | `inst_scale_efficiency`（多卡） |
| --- | --- | --- | --- | --- | --- |
| 保守 | 0.60 | 0.25 | 0.08 | 0.15 | 0.75 |
| 中性 | 0.70 | 0.35 | 0.05 | 0.10 | 0.85 |
| 乐观 | 0.80 | 0.45 | 0.03 | 0.06 | 0.90 |

说明：

- 单卡实例固定取 `inst_scale_efficiency = 1.0`。
- 采购前正式输出至少应给出保守档与中性档两组结果。
- 若后续拿到框架实测或压测结果，应优先以实测值回填这些参数。

### 5.6 HA 输入（必填）

| 参数 | 含义 |
| --- | --- |
| `ha_mode` | `none` / `N+1` / `survive_failure_unit` / `survive_node` |
| `ha_target_units` | 需容忍的故障单元数 |
| `node_gpu_count` | 单节点 GPU 数；若按节点维度做 HA 建议提供 |

---

## 6. 输入校验与基础派生

### 6.1 合法性校验

至少应校验：

- $concurrency\_avg > 0$
- $concurrency\_peak >= concurrency\_avg$
- $target\_ttft\_avg\_sec > 0$
- $target\_ttft\_p95\_sec >= target\_ttft\_avg\_sec$
- $target\_e2e\_avg\_sec > target\_ttft\_avg\_sec$
- $target\_e2e\_p95\_sec > target\_ttft\_p95\_sec$
- $0 < decode\_active\_ratio\_avg \le 1$
- $0 < decode\_active\_ratio\_peak \le 1$
- $0 < usable\_vram\_ratio \le 1$
- $\sum request\_shapes.weight = 1$
- $memory\_bandwidth\_efficiency > 0$
- $compute\_efficiency > 0$

### 6.2 从 `request_shapes` 派生全局长度统计

全局平均输入长度：

$$
S_{in,avg} = \sum_i (w_i \times S_{in,avg,i})
$$

全局平均输出长度：

$$
S_{out,avg} = \sum_i (w_i \times S_{out,avg,i})
$$

全局 P95 输入与输出长度：

- 若有原始离散桶分布，可按累计分位近似求取。
- 若仅有少量离散 shape，工程上可取“高位主导 shape”的保守近似。

记最终派生结果为：

- $S_{in,p95}$
- $S_{out,p95}$

### 6.3 总长度 P95

优先顺序：

1. 若业务侧单独提供 `target_total_tokens_p95`，优先使用。
2. 若未提供，则保守近似：

$$
S_{p95} = S_{in,p95} + S_{out,p95}
$$

### 6.4 基础总长度

平均总长度：

$$
S_{avg} = S_{in,avg} + S_{out,avg}
$$

P95 总长度：

$$
S_{p95} =
\begin{cases}
 target\_total\_tokens\_p95, & \text{若已提供} \\
 S_{in,p95} + S_{out,p95}, & \text{否则采用保守近似}
\end{cases}
$$

---

## 7. 从体验目标到系统需求

### 7.1 平均阶段预算

平均 decode 预算：

$$
T_{decode,avg} = target\_e2e\_avg\_sec - target\_ttft\_avg\_sec
$$

默认平均 prefill 预算份额：

$$
ttft\_prefill\_share_{avg} = 0.8
$$

平均单请求 prefill 速率需求：

$$
R_{prefill,req}^{avg} = \frac{S_{in,avg}}{target\_ttft\_avg\_sec \times ttft\_prefill\_share_{avg}}
$$

平均单请求 decode 生成速率需求：

$$
R_{decode,req}^{avg} = \frac{S_{out,avg}}{T_{decode,avg}}
$$

### 7.2 P95 阶段预算

P95 decode 预算采用工程预算近似：

$$
T_{decode,p95}^{budget} = target\_e2e\_p95\_sec - target\_ttft\_p95\_sec
$$

默认 P95 prefill 预算份额：

$$
ttft\_prefill\_share_{p95} = 0.8
$$

P95 单请求 prefill 速率需求：

$$
R_{prefill,req}^{p95} = \frac{S_{in,p95}}{target\_ttft\_p95\_sec \times ttft\_prefill\_share_{p95}}
$$

P95 单请求 decode 生成速率需求：

$$
R_{decode,req}^{p95} = \frac{S_{out,p95}}{T_{decode,p95}^{budget}}
$$

说明：

- $ttft\_prefill\_share_{avg}$ 与 $ttft\_prefill\_share_{p95}$ 是将 TTFT 预算映射为 prefill 需求的工程缺省值。
- 若业务侧或框架侧已有更合适的预算拆分口径，应优先替换默认值。

### 7.3 系统总吞吐需求

平均 decode 总吞吐需求：

$$
TPS_{decode,target}^{avg} = C_{avg} \times r_{decode,avg} \times R_{decode,req}^{avg}
$$

峰值 decode 总吞吐需求：

$$
TPS_{decode,target}^{peak} = C_{peak} \times r_{decode,peak} \times R_{decode,req}^{p95}
$$

Prefill 总吞吐需求采用保守上界：

$$
TPS_{prefill,target}^{peak} = C_{peak} \times R_{prefill,req}^{p95}
$$

说明：

- Prefill 在峰值时按全部峰值活跃并发参与竞争处理，是保守近似。
- Decode 则按峰值窗口的 decode-active 比例折算。

---

## 8. 单实例显存模型

### 8.1 权重显存

原始权重字节数：

$$
M_{weights,raw,bytes} = total\_params\_b \times 10^9 \times bytes\_per\_param(inference\_precision)
$$

考虑权重格式、量化元数据与布局附加开销后的权重显存：

$$
M_{weights,bytes} = M_{weights,raw,bytes} \times (1 + weight\_overhead\_ratio)
$$

说明：

- $weight\_overhead\_ratio$ 应保持为小比例系数，默认用于采购前预估。
- 若框架实测可直接给出模型加载后的稳定驻留显存，应优先以实测值反推或替换该系数。

### 8.2 Cache 每 token 每层字节数

若显式给出：

$$
E_{cache/token/layer,bytes} = cache\_bytes\_per\_token\_per\_layer
$$

否则：

- MHA：

$$
E_{cache/token/layer,bytes} = 2 \times hidden\_size \times bytes\_per\_cache\_element(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

- GQA：

$$
E_{cache/token/layer,bytes} = 2 \times num\_kv\_heads \times head\_dim \times bytes\_per\_cache\_element(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

- MQA：

$$
E_{cache/token/layer,bytes} = 2 \times head\_dim \times bytes\_per\_cache\_element(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

- MLA：

$$
E_{cache/token/layer,bytes} = latent\_cache\_dim \times bytes\_per\_cache\_element(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

- Hybrid Attention：

对不同层类型分别按其对应结构计算 $E_{cache/token/layer,bytes}^{(l)}$，再按层求和；若无法提供逐层结构信息，则必须直接输入 `cache_bytes_per_token_per_layer`。

注：若实现上将 MQA 统一为 $num\_kv\_heads = 1$ 的 GQA 特例，也可复用 GQA 公式，但必须显式满足 $num\_kv\_heads = 1$。

### 8.3 单请求 Cache

对给定序列长度 $S$：

$$
M_{cache,req,bytes}(S) = num\_layers \times S \times E_{cache/token/layer,bytes}
$$

若为 Hybrid Attention 且使用逐层求和，则：

$$
M_{cache,req,bytes}(S) = S \times \sum_{l=1}^{num\_layers} E_{cache/token/layer,bytes}^{(l)}
$$

### 8.4 运行时固定显存

$$
M_{runtime,bytes} = runtime\_buffer\_ratio \times M_{weights,bytes}
$$

### 8.5 单实例总显存组成

单实例显存由三部分组成：

$$
M_{inst,total,bytes} = M_{weights,bytes} + M_{runtime,bytes} + M_{cache,active,bytes}
$$

其中：

- $M_{weights,bytes}$：权重静态驻留显存
- $M_{runtime,bytes}$：运行时长期固定显存
- $M_{cache,active,bytes}$：活跃请求占用的 cache 显存

采购前 sizing 中，后续通过单实例 cache 容量上限反推可承载请求数，而不在此处预设单实例请求数符号。

### 8.6 单实例显存可行性

#### 8.6.1 单卡完整装载场景

当 $instance\_gpus = 1$ 时，单实例可用显存为：

$$
M_{usable/inst,bytes} = vram\_bytes \times usable\_vram\_ratio
$$

若：

$$
M_{weights,bytes} + M_{runtime,bytes} \ge M_{usable/inst,bytes}
$$

则单卡完整装载不可行。

单实例可承载 cache 容量：

$$
M_{cache,cap/inst,bytes} = M_{usable/inst,bytes} - M_{weights,bytes} - M_{runtime,bytes}
$$

该场景下：

- $deployment\_mode = single\_gpu\_full$
- $memory\_feasibility\_type = strict$

#### 8.6.2 多卡分片场景

当 $instance\_gpus > 1$ 时，本文在前期 sizing 中采用**实例级 pooled/sharded 可行性上界**近似：

$$
M_{usable/inst,bytes} = instance\_gpus \times vram\_bytes \times usable\_vram\_ratio
$$

若：

$$
M_{weights,bytes} + M_{runtime,bytes} \ge M_{usable/inst,bytes}
$$

则当前 $instance\_gpus$ 方案不可行。

单实例可承载 cache 容量：

$$
M_{cache,cap/inst,bytes} = M_{usable/inst,bytes} - M_{weights,bytes} - M_{runtime,bytes}
$$

该场景下：

- $deployment\_mode = multi\_gpu\_sharded$
- $memory\_feasibility\_type = sharded\_upper$

说明：此处更适合作为多卡 / sharded 方案的前期可行性判断，而不应被解读为 replicated 多卡实例的严格显存证明。

---

## 9. 单实例吞吐模型

### 9.1 激活参数量定义

令：

- Dense：$active\_params\_b = total\_params\_b$
- MoE：$active\_params\_b = activated\_params\_per\_token\_b$

并记：

$$
active\_params = active\_params\_b \times 10^9
$$

单位为参数个数。

### 9.2 Decode 单实例吞吐

单 token 近似访存量：

$$
bytes_{decode/token} = active\_params \times bytes\_per\_param(inference\_precision)
$$

单 token 近似计算量：

$$
flops_{decode/token} = 2 \times active\_params
$$

其中 $peak\_compute\_flops\_per\_sec(inference\_precision)$ 按推理精度选择：

- FP16 $\rightarrow fp16\_flops\_per\_sec$
- BF16 $\rightarrow bf16\_flops\_per\_sec$
- FP8 $\rightarrow fp8\_flops\_per\_sec$
- INT8 $\rightarrow int8\_flops\_per\_sec$
- INT4 / W4A16 $\rightarrow int4\_flops\_per\_sec$

内存受限近似：

$$
TPS_{decode,memory}^{gpu} = \frac{memory\_bandwidth\_bytes\_per\_sec \times memory\_bandwidth\_efficiency}{bytes_{decode/token}}
$$

算力受限近似：

$$
TPS_{decode,compute}^{gpu} = \frac{peak\_compute\_flops\_per\_sec(inference\_precision) \times compute\_efficiency}{flops_{decode/token}}
$$

单 GPU decode 吞吐：

$$
TPS_{decode}^{gpu} = \min(TPS_{decode,memory}^{gpu}, TPS_{decode,compute}^{gpu})
$$

若 $instance\_gpus = 1$，取：

$$
inst\_scale\_efficiency = 1.0
$$

若 $instance\_gpus > 1$ 且无实测值，可先取推荐默认值 $0.85$。

单实例 decode 吞吐：

$$
TPS_{decode}^{inst} = instance\_gpus \times TPS_{decode}^{gpu} \times inst\_scale\_efficiency
$$

说明：

- 上式是前期 sizing 近似，attention、norm、rope、softmax、MLA 等额外 FLOPs 与 IO 被吸收到有效利用率与实例扩展效率中。
- 若有实测值，应优先替换。
- 对大模型在线推理，decode 常更偏 memory-bound，因此 $memory\_bandwidth\_efficiency$ 通常是更关键的主导参数。

### 9.3 Prefill 单实例吞吐

先定义基线单 token 访存量：

$$
bytes_{prefill/token} = active\_params \times bytes\_per\_param(inference\_precision)
$$

内存受限上界：

$$
TPS_{prefill,memory}^{gpu} = \frac{memory\_bandwidth\_bytes\_per\_sec \times memory\_bandwidth\_efficiency}{bytes_{prefill/token}}
$$

单 token 近似计算量：

$$
flops_{prefill/token} = 2 \times active\_params
$$

算力受限上界：

$$
TPS_{prefill,compute}^{gpu} = \frac{peak\_compute\_flops\_per\_sec(inference\_precision) \times compute\_efficiency}{flops_{prefill/token}}
$$

单 GPU prefill 吞吐：

$$
TPS_{prefill}^{gpu} = \min(TPS_{prefill,memory}^{gpu}, TPS_{prefill,compute}^{gpu})
$$

单实例 prefill 吞吐：

$$
TPS_{prefill}^{inst} = instance\_gpus \times TPS_{prefill}^{gpu} \times inst\_scale\_efficiency
$$

说明：

- 该模型对 prefill 吞吐给出的是采购前阶段的保守上界近似。
- 该模型将 prefill 视为不额外享受批内复用折减的保守上界近似。

---

## 10. 部署方案求解

### 10.1 总原则

部署模式不是用户输入，而是部署测算过程中的自动推断结果。

部署求解接收业务目标、模型参数、GPU 规格、框架档位与 HA 目标，并自动求解最小可行部署方案。

### 10.2 第一分支：单卡完整装载检查

从 $instance\_gpus = 1$ 开始，依次检查：

1. 显存是否可容纳 $M_{weights,bytes} + M_{runtime,bytes}$
2. 单实例显存是否足够承载目标请求数对应的 cache
3. 单实例 prefill / decode 吞吐是否满足需求

若单卡可行，则：

- $instance\_gpus = 1$
- $is\_sharded = false$
- $deployment\_mode = single\_gpu\_full$
- $memory\_feasibility\_type = strict$

### 10.3 第二分支：多卡实例枚举

若单卡不可行，则枚举：

$$
instance\_gpus \in \{2, 4, 8, ...\}
$$

对每个候选 $instance\_gpus$：

1. 检查显存可行性。
2. 估算 $TPS_{prefill}^{inst}$ 与 $TPS_{decode}^{inst}$。
3. 若当前候选依赖实例级 pooled/sharded 显存上界与跨卡协同吞吐建模，则视为分片方案。

若当前候选可行，则：

- $is\_sharded = true$
- $deployment\_mode = multi\_gpu\_sharded$
- $memory\_feasibility\_type = sharded\_upper$

### 10.4 故障单元推导

- 若单卡独立实例：通常 $failure\_unit\_gpus = 1$
- 若多卡强耦合实例：通常 $failure\_unit\_gpus = instance\_gpus$
- 若按整节点绑定故障：可取 $failure\_unit\_gpus = node\_gpu\_count$

### 10.5 候选方案选择原则

对所有可行候选，选择业务所需 GPU 数最小的方案；若多方案 $G_{biz}$ 相同，则优先：

1. 单卡完整装载方案
2. 故障单元更小的方案
3. 具有更高吞吐冗余的方案

---

## 11. 业务所需实例数与 GPU 数

### 11.1 Decode 约束实例数

$$
N_{inst,decode} = \left\lceil \frac{TPS_{decode,target}^{peak}}{TPS_{decode}^{inst}} \right\rceil
$$

### 11.2 Prefill 约束实例数

$$
N_{inst,prefill} = \left\lceil \frac{TPS_{prefill,target}^{peak}}{TPS_{prefill}^{inst}} \right\rceil
$$

### 11.3 显存约束实例数

先计算单实例在峰值保守场景下可承载的请求数：

$$
N_{req/inst,mem}^{max} = \left\lfloor \frac{M_{cache,cap/inst,bytes}}{M_{cache,req,bytes}(S_{p95})} \right\rfloor
$$

若：

$$
N_{req/inst,mem}^{max} \le 0
$$

则当前实例方案不可行。

显存约束实例数：

$$
N_{inst,memory} = \left\lceil \frac{C_{peak}}{N_{req/inst,mem}^{max}} \right\rceil
$$

### 11.4 业务实例数与业务 GPU 数

$$
N_{inst,biz} = \max(N_{inst,decode}, N_{inst,prefill}, N_{inst,memory})
$$

$$
G_{biz} = N_{inst,biz} \times instance\_gpus
$$

说明：

- 这是采购前阶段的业务最小 GPU 数。
- 它已经综合了显存、prefill、decode 三条主约束。

---

## 12. 高可用展开

### 12.1 HA 后实例数

- $ha\_mode = none$：

$$
N_{inst,final} = N_{inst,biz}
$$

- $ha\_mode = survive\_failure\_unit$：

$$
N_{inst,final} = N_{inst,biz} + \left\lceil \frac{ha\_target\_units \times failure\_unit\_gpus}{instance\_gpus} \right\rceil
$$

- $ha\_mode = survive\_node$：需按节点粒度折算。
- $ha\_mode = N+1$：可按至少增加 1 个实例处理。

### 12.2 最终 GPU 数

$$
G_{final} = N_{inst,final} \times instance\_gpus
$$

说明：

- $G_{biz}$ 用于业务承载。
- $G_{final}$ 用于采购与最终资源预留。

---

## 13. 基于已推导 GPU 数的预期效果反推

本章用于 sanity check，不作为采购前测算的主决策依据。

### 13.1 集群吞吐能力

业务集群 decode 吞吐：

$$
TPS_{decode,cluster} = N_{inst,biz} \times TPS_{decode}^{inst}
$$

业务集群 prefill 吞吐：

$$
TPS_{prefill,cluster} = N_{inst,biz} \times TPS_{prefill}^{inst}
$$

### 13.2 可持续并发能力

先定义 decode-active 可持续并发：

$$
C_{decode,sustainable} = \frac{TPS_{decode,cluster}}{R_{decode,req}^{p95}}
$$

再折算总活跃并发：

$$
C_{total,sustainable,decode} = \frac{C_{decode,sustainable}}{r_{decode,peak}}
$$

Prefill 可持续并发：

$$
C_{total,sustainable,prefill} = \frac{TPS_{prefill,cluster}}{R_{prefill,req}^{p95}}
$$

最终可持续并发能力：

$$
ConcurrentCapacity_{sustainable} = \min(C_{total,sustainable,decode}, C_{total,sustainable,prefill})
$$

### 13.3 单请求生成速度

$$
Speed_{decode/request} \approx \frac{TPS_{decode,cluster}}{C_{peak} \times r_{decode,peak}}
$$

### 13.4 平均时延近似

平均 TTFT 近似：

$$
TTFT_{avg,est} \approx \frac{S_{in,avg}}{TPS_{prefill,cluster} / C_{avg}}
$$

平均 E2E 近似：

$$
E2E_{avg,est} \approx \frac{S_{in,avg}}{TPS_{prefill,cluster} / C_{avg}} + \frac{S_{out,avg}}{TPS_{decode,cluster} / (C_{avg} \times r_{decode,avg})}
$$

说明：

- 平均 TTFT / E2E 中的 prefill 部分按全部平均活跃并发竞争 Prefill 吞吐的保守近似计算。
- 该式是工程近似与 sanity check，不是队列论严格推导。

### 13.5 保守 P95 时延近似

$$
TTFT_{p95,upper} \approx \frac{S_{in,p95}}{TPS_{prefill,cluster} / C_{peak}}
$$

$$
E2E_{p95,upper} \approx \frac{S_{in,p95}}{TPS_{prefill,cluster} / C_{peak}} + \frac{S_{out,p95}}{TPS_{decode,cluster} / (C_{peak} \times r_{decode,peak})}
$$

### 13.6 每日产能

$$
DailyDecodeTokens = TPS_{decode,cluster} \times 86400
$$

$$
DailyPrefillTokens = TPS_{prefill,cluster} \times 86400
$$

---

## 14. 输出项建议

### 14.1 输入摘要

- 并发、TTFT、E2E 目标
- `request_shapes` 摘要
- `decode_active_ratio_avg` / `decode_active_ratio_peak`
- 模型 / GPU / 框架 / HA 选择
- 是否使用默认值及其清单

### 14.2 部署求解结果

- $instance\_gpus$
- $is\_sharded$
- $deployment\_mode$
- $memory\_feasibility\_type$
- $failure\_unit\_gpus$
- $TPS_{prefill}^{inst}$
- $TPS_{decode}^{inst}$
- $N_{inst,memory}$ / $N_{inst,prefill}$ / $N_{inst,decode}$

### 14.3 资源规模结果

- $N_{inst,biz}$
- $G_{biz}$
- $N_{inst,final}$
- $G_{final}$
- 若使用档位输入，建议同时输出保守 / 中性 / 乐观三档结果

### 14.4 效果反推结果

- $TPS_{prefill,cluster}$
- $TPS_{decode,cluster}$
- $ConcurrentCapacity_{sustainable}$
- $Speed_{decode/request}$
- $TTFT_{avg,est}$ / $TTFT_{p95,upper}$
- $E2E_{avg,est}$ / $E2E_{p95,upper}$
- $DailyDecodeTokens$
- $DailyPrefillTokens$

### 14.5 结果展示单位建议

为避免中间推导混入单位换算，建议仅在展示层做如下换算：

- $byte \rightarrow GiB$：

$$
value_{GiB} = \frac{value_{bytes}}{2^{30}}
$$

- $byte/s \rightarrow GiB/s$：

$$
value_{GiB/s} = \frac{value_{bytes/s}}{2^{30}}
$$

- $FLOP/s \rightarrow TFLOP/s$：

$$
value_{TFLOPS} = \frac{value_{flops/s}}{10^{12}}
$$

- $FLOP \rightarrow TFLOP$：

$$
value_{TFLOP} = \frac{value_{flops}}{10^{12}}
$$

显存与带宽展示默认采用二进制单位 $GiB$ / $GiB/s$；核心公式内部仍统一使用 $byte$ 与 $byte/s$，不在中间过程混入任何展示单位换算。

---

## 15. 使用建议

### 15.1 关于必填但允许默认值的理解

本文将 `request_shapes`、`decode_active_ratio_avg`、`decode_active_ratio_peak` 设为必填，不意味着业务侧一开始就必须拥有精确真实值，而是要求：

- 这些项必须在测算输入中显式填写。
- 若没有真实观测，可采用推荐默认值或保守估计。
- 最终结果必须注明哪些结论建立在默认假设之上。

### 15.2 推荐的使用顺序

1. 先用已有日志数据生成 `request_shapes`、并发与时延目标。
2. 根据日志窗口统计得到 `decode_active_ratio_avg` 与 `decode_active_ratio_peak`。
3. 根据模型规格填写模型结构、参数规模与精度。
4. 根据 GPU 官网规格填写显存、带宽与峰值算力。
5. 采用保守 / 中性 / 乐观档位填写 `memory_bandwidth_efficiency`、`compute_efficiency`、`weight_overhead_ratio`、`runtime_buffer_ratio`、`inst_scale_efficiency`，至少保留保守与中性两组结果。
6. 自动求解最小可行部署方案，得到 $G_{biz}$ 与 $G_{final}$。
7. 用效果反推结果做 sanity check。
8. 若后续获得 profiling / benchmark，再回填修正关键效率参数。

### 15.3 最容易误用的点

- 把少量 shape 样本当成真实完整分布。
- 把 `decode_active_ratio` 当成 GPU 或框架固定属性。
- 把 `memory_bandwidth_efficiency` / `compute_efficiency` 误认为芯片天生常数。
- 把 `weight_overhead_ratio` 与 `runtime_buffer_ratio` 混为同一类开销并重复保守。
- 把多卡 `sharded_upper` 可行性误认为严格部署证明。
- 把效果反推结果误认为 SLA 保证值。
- 在中间计算里混用 `GB`、`TFLOP`、`B(十亿参数)` 等展示单位。

---

## 16. 一页式总结

本文最终采用以下主线：

**输入条件 $\rightarrow$ 部署方案求解 $\rightarrow$ 业务 GPU 数 $\rightarrow$ HA 后最终 GPU 数 $\rightarrow$ 基于 GPU 数反推可达到效果**

其中：

- 业务目标输入收敛为并发、TTFT、E2E、`request_shapes`、`decode_active_ratio_avg`、`decode_active_ratio_peak`，并可选补充 `target_total_tokens_p95`。
- 模型输入定义模型结构、规模与精度。
- GPU 选择输入同时包含 GPU 理论规格与有效利用率假设。
- 推理框架 / 部署输入收敛为少量必要档位参数。
- HA 输入单独定义可用性目标。
- $instance\_gpus$、$is\_sharded$、$deployment\_mode$、$failure\_unit\_gpus$ 是部署求解结果，而非原始输入。
- 核心推导中的存储统一用 $byte$，token 统一用 $token$，计算量统一用 $FLOP$，时间统一用 $second$。

因此，本文不是“先假设部署口径，再反推卡数”，而是：

**先给定业务与技术输入，再自动求最小可行部署方案与所需 GPU 数量。**
