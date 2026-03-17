# 大模型在线推理 GPU 资源测算方法

---

## 1. 文档目标

本文用于在线推理场景下的大模型 GPU 资源测算，给出一套从**输入条件**出发，经过**部署方案求解**，最终输出**GPU 数量与可达到效果**的完整方法。

本文目标不是做精确 benchmark 复现，而是提供一套：

- 逻辑自洽
- 公式口径统一
- 可工程实现
- 可用于前期 sizing / 方案比较 / 容量规划

的估算方法。

本文输出：

- 推荐部署方案
- 最小服务单元（实例）需要的 GPU 数 `instance_gpus`
- 是否需要 sharding `is_sharded`
- 业务所需 GPU 数 `G_biz`
- 考虑高可用后的最终 GPU 数 `G_final`
- 对应可达到的并发、吞吐、速度、时延等预期效果

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

1. 横向扩副本时吞吐近似线性增长
2. 单实例内部 TP / EP / PP 的通信开销只通过效率因子近似吸收，不做精细链路建模
3. Prefill 吞吐在缺少到达率信息时采用保守上界近似
4. 平均 / P95 对话时延反推属于 sanity check，不构成 SLA 证明
5. 多维 P95 长度拼装属于保守 sizing，而非严格联合分布建模

若需要高精度建模，应补充：

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

输入不只是数值参数，还包括业务目标与技术选择：

- 业务目标
- 模型选择
- GPU 选择
- 推理框架选择
- HA 选择

### 第 2 层：部署方案求解

根据输入条件，推导：

- 单卡是否可行
- 最小服务单元由几张卡组成
- 是否需要 sharding
- 单实例显存与吞吐能力
- 故障单元规模

### 第 3 层：资源规模输出

在部署方案确定后，计算：

- 显存约束实例数 / GPU 数
- Prefill 吞吐约束实例数 / GPU 数
- Decode 吞吐约束实例数 / GPU 数
- 业务所需最小 GPU 数 `G_biz`
- 高可用后最终 GPU 数 `G_final`

### 第 4 层：效果反推与校验

基于已推导的 GPU 数，反推：

- 可持续并发能力
- 满载吞吐上限
- 单请求生成速度
- 平均 / 保守 P95 时延近似
- 每日产能

---

## 4. 统一术语与记号

### 4.1 基本对象

为避免“GPU”“实例”“副本”“集群”混用，本文统一定义：

- **GPU**：单张物理卡
- **实例（instance）**：一个可独立提供推理服务的最小服务单元
- **副本（replica）**：多个同构实例组成的横向扩展副本集合
- **集群（cluster）**：所有用于承载业务的实例总和

### 4.2 关键部署概念

- `instance_gpus`：单实例占用的 GPU 数
- `instance_count`：业务实例数
- `G_total = instance_count × instance_gpus`
- `is_sharded`：单实例内部是否需要跨卡切分模型/缓存
- `failure_unit_gpus`：一次最小故障会损失的 GPU 数

### 4.3 关键结论

本文不将以下量视为原始输入，而将其视为**部署求解结果**：

- `instance_gpus`
- `is_sharded`
- `failure_unit_gpus`

这三个量通常由：

- 模型规模与结构
- GPU 规格
- 推理框架能力
- 用户体验目标
- HA 目标

共同决定。

### 4.4 记号映射

- `S_{in,avg}`：平均输入长度
- `S_{out,avg}`：平均输出长度
- `S_{in,p95}`：P95 输入长度
- `S_{out,p95}`：P95 输出长度
- `S_{p95}`：P95 总长度
- `C_avg`：平均活跃并发
- `C_peak`：峰值活跃并发
- `r_decode`：峰值时 decode-active 比例

---

## 5. 输入条件

本章所有内容都属于输入层。需要强调：输入既可以是**唯一指定值**，也可以是**候选集 / 约束条件**。

### 5.1 业务目标输入

| 参数 | 含义 |
| --- | --- |
| `concurrency_avg` | 平均活跃并发 |
| `concurrency_peak` | 峰值活跃并发 |
| `target_ttft_avg_sec` | 平均 TTFT 目标，可选 |
| `target_ttft_p95_sec` | P95 TTFT 目标 |
| `target_e2e_avg_sec` | 平均单次对话总耗时目标 |
| `target_e2e_p95_sec` | P95 单次对话总耗时目标 |
| `target_decode_avg_sec` | 平均 decode 时长目标，可选 |
| `target_decode_p95_sec` | P95 decode 时长目标，可选 |
| `target_daily_tokens` | 期望每日 token 产能，可选 |
| `cost_budget` | 成本预算，可选 |

说明：

- 业务目标是最典型的硬约束输入
- 若未显式给出 decode 时长目标，允许通过预算分配近似求得

### 5.2 模型输入

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
| `cache_aux_bytes_per_token_per_layer` | 每 token 每层额外 cache 开销 |
| `cache_bytes_per_token_per_layer` | 直接指定 KV Cache 字节数，优先级最高 |

说明：

- “模型输入”可以是已选定单一模型，也可以是候选模型集
- 若是候选模型集，则应分别计算并比较结果

### 5.3 请求画像输入

| 参数 | 含义 |
| --- | --- |
| `target_input_tokens_avg` | 平均输入长度 |
| `target_output_tokens_avg` | 平均输出长度 |
| `target_input_tokens_p95` | P95 输入长度 |
| `target_output_tokens_p95` | P95 输出长度 |
| `target_total_tokens_p95` | P95 总长度 |
| `request_shapes` | 请求类型分桶与占比 |
| `decode_active_ratio` | 峰值时处于 decode 阶段的活跃比例 |
| `prefix_cache_hit_rate` | Prefix Cache 命中率 |

说明：

- `decode_active_ratio` 不是模型常数，应来自业务统计或仿真
- `request_shapes` 优先级高于简单的平均值/P95 拼装

### 5.4 GPU 选择输入

| 参数 | 含义 |
| --- | --- |
| `gpu_candidates` | 候选 GPU 列表 |
| `gpu_name` | GPU 名称 |
| `vram_gb` | 单卡显存 |
| `memory_bandwidth_gb_per_sec` | HBM 带宽 |
| `fp16_tflops` | FP16 算力 |
| `bf16_tflops` | BF16 算力 |
| `fp8_tflops` | FP8 算力 |
| `int8_tflops` | INT8 算力 |
| `int4_tflops` | INT4 / W4A16 算力 |

说明：

- GPU 选择通常是候选输入，而非一开始唯一确定
- 若存在多种 GPU 候选，应分别求解，比较成本与规模

### 5.5 推理框架输入

| 参数 | 含义 |
| --- | --- |
| `framework_candidates` | 候选框架列表 |
| `framework` | `vLLM` / `SGLang` / 其他 |
| `decode_efficiency` | Decode 阶段系统效率 |
| `prefill_efficiency` | Prefill 阶段系统效率 |
| `compute_efficiency` | 峰值算力折减系数 |
| `prefill_memory_reuse_factor` | Prefill 权重复用因子 |
| `weight_overhead_ratio` | 权重额外开销比例 |
| `runtime_overhead_ratio` | 运行时比例开销 |
| `runtime_overhead_gb` | 运行时固定开销 |
| `usable_vram_ratio` | 安全可用显存比例 |

说明：

- 框架影响显存、调度和分阶段效率，必须作为输入层显式考虑
- 若未指定框架，可对候选框架分别计算

### 5.6 HA 选择输入

| 参数 | 含义 |
| --- | --- |
| `ha_mode` | `none` / `n_plus_1` / `active_standby` / `active_active` |
| `replica_count` | 多活副本数 |
| `ha_target` | 可承受的故障目标，如 1 实例/1 节点故障 |
| `failover_reserve_ratio` | 额外冗余比例 |
| `node_gpu_count` | 单节点 GPU 数，可选 |

说明：

- HA 选择是输入，因为它表达的是业务/平台给出的可用性要求
- 但 `failure_unit_gpus` 本身不是输入，而是由部署方案推导出来的结果

### 5.7 合法性检查

输入应满足：

- `concurrency_avg > 0`
- `concurrency_peak >= concurrency_avg`
- `0 < decode_active_ratio <= 1`
- `0 <= prefix_cache_hit_rate <= 1`
- `target_e2e_p95_sec > target_ttft_p95_sec`
- 若提供 `prefill_time_share_avg`，则 `0 < prefill_time_share_avg < 1`
- 若提供 `ttft_prefill_share_p95`，则 `0 < ttft_prefill_share_p95 < 1`
- 若提供 `decode_time_share_p95`，则 `0 < decode_time_share_p95 < 1`

---

## 6. 请求画像建模

设共有 `K` 类请求，第 `i` 类请求占比为 `p_i`，满足：

$$
\sum_{i=1}^{K} p_i = 1
$$

其中：

- 输入长度 `S_{in,i}`
- 输出长度 `S_{out,i}`
- 总长度 `S_i = S_{in,i} + S_{out,i}`

### 6.1 平均长度

$$
E[S_{in}] = \sum_{i=1}^{K} p_i \cdot S_{in,i}
$$

$$
E[S_{out}] = \sum_{i=1}^{K} p_i \cdot S_{out,i}
$$

$$
E[S] = \sum_{i=1}^{K} p_i \cdot S_i
$$

### 6.2 P95 长度

定义：

- `S_{in,p95}`：输入长度 P95
- `S_{out,p95}`：输出长度 P95
- `S_{p95}`：总长度 P95

说明：

- 三者不要求来自同一条请求
- 若将它们分别用于不同公式，应理解为保守上界建模
- 若具备联合分布，优先按 request shape 分桶计算，替代单变量 P95 拼装

### 6.3 Prefix Cache 修正

设命中率：

$$
r_{hit} = prefix\_cache\_hit\_rate
$$

则有效输入长度：

$$
S_{in,avg}^{effective} = S_{in,avg} \times (1-r_{hit})
$$

$$
S_{in,p95}^{effective} = S_{in,p95} \times (1-r_{hit})
$$

说明：

- Prefix Cache 仅影响 prefill 输入处理量
- 不改变输出长度
- 不改变已生成历史 token 的 KV Cache 占用口径

---

## 7. 从业务目标到系统需求

### 7.1 时间定义

首 token 延迟：

$$
TTFT = t_{first\_token} - t_{request}
$$

端到端时延：

$$
Latency_{e2e} = t_{last\_token} - t_{request}
$$

工程上通常有：

$$
TTFT \approx T_{queue} + T_{schedule} + T_{prefill\_compute} + T_{sync} + T_{network}
$$

因此，TTFT 不能直接等同于 prefill 计算时长。

### 7.2 Prefill 时间预算

P95 场景下，取：

$$
T_{prefill,p95} = target\_ttft\_p95\_sec \times r_{ttft\rightarrow prefill}
$$

其中：

$$
r_{ttft\rightarrow prefill} = ttft\_prefill\_share\_p95
$$

默认建议：

$$
0.7 \le r_{ttft\rightarrow prefill} \le 0.9
$$

平均场景下：

- 若提供 `target_ttft_avg_sec`，则取其作为平均 prefill 预算近似
- 若未提供，则用 `prefill_time_share_avg × target_e2e_avg_sec` 近似

### 7.3 Decode 时间预算

#### P95

严格来说，不能使用：

$$
P95(Latency_{e2e}) - P95(TTFT)
$$

来定义 `P95(T_decode)`。

因此采用以下优先级：

1. 若提供 `target_decode_p95_sec`，则

$$
T_{decode,p95} = target\_decode\_p95\_sec
$$

2. 否则按预算分配：

$$
T_{decode,p95} = target\_e2e\_p95\_sec \times decode\_time\_share\_p95
$$

3. 若未提供 `decode_time_share_p95`，可取：

$$
decode\_time\_share\_p95 = 1 - \frac{target\_ttft\_p95\_sec}{target\_e2e\_p95\_sec}
$$

但必须强调：这只是预算分配近似，不是严格统计恒等式。

#### 平均

1. 若提供 `target_decode_avg_sec`，则：

$$
T_{decode,avg} = target\_decode\_avg\_sec
$$

2. 否则若提供 `target_ttft_avg_sec`，则：

$$
T_{decode,avg} = target\_e2e\_avg\_sec - target\_ttft\_avg\_sec
$$

3. 否则：

$$
T_{prefill,avg} = prefill\_time\_share\_avg \times target\_e2e\_avg\_sec
$$

$$
T_{decode,avg} = target\_e2e\_avg\_sec - T_{prefill,avg}
$$

### 7.4 单请求速率目标

平均场景：

$$
R_{prefill,req}^{avg} = \frac{S_{in,avg}^{effective}}{T_{prefill,avg}}
$$

$$
R_{decode,req}^{avg} = \frac{S_{out,avg}}{T_{decode,avg}}
$$

P95 场景：

$$
R_{prefill,req}^{p95} = \frac{S_{in,p95}^{effective}}{T_{prefill,p95}}
$$

$$
R_{decode,req}^{p95} = \frac{S_{out,p95}}{T_{decode,p95}}
$$

### 7.5 系统总吞吐需求

设：

$$
C_{peak} = concurrency\_peak
$$

$$
r_{decode} = decode\_active\_ratio
$$

则峰值 decode-active 请求数：

$$
C_{decode} = C_{peak} \times r_{decode}
$$

#### Decode 总吞吐需求

$$
target\_decode\_tps\_total = C_{decode} \times R_{decode,req}^{p95}
$$

即：

$$
target\_decode\_tps\_total = \frac{C_{peak} \times r_{decode} \times S_{out,p95}}{T_{decode,p95}}
$$

#### Prefill 总吞吐需求

缺少到达率数据时，使用保守上界：

$$
target\_prefill\_tps\_total^{upper} = C_{peak} \times R_{prefill,req}^{p95}
$$

即：

$$
target\_prefill\_tps\_total^{upper} = \frac{C_{peak} \times S_{in,p95}^{effective}}{T_{prefill,p95}}
$$

注意：

- Decode 对应稳态活跃生成吞吐
- Prefill 对应缺少到达率时的保守突发上界
- 两者物理语义不同，不应机械对称理解

---

## 8. 部署方案求解

本章的核心不是直接算总卡数，而是先求出：

- 单卡是否可行
- 最小服务单元需要几张卡
- 是否需要 sharding
- 单实例的显存与吞吐能力

### 8.1 单卡可行性判断

设单卡可用显存：

$$
M_{usable,gpu} = vram\_gb \times usable\_vram\_ratio
$$

若单卡实例，则单实例可用显存：

$$
M_{usable,inst} = M_{usable,gpu}
$$

先检查：

1. 单卡是否能装下权重与运行时
2. 单卡吞吐是否能满足目标

若均满足，则：

- `instance_gpus = 1`
- `is_sharded = false`

若任一不满足，则进入多卡实例方案求解。

### 8.2 多卡实例与 sharding 方案枚举

对候选值 `instance_gpus ∈ {2,4,8,...}` 枚举，逐一判断可行性。

- 若单实例需要多张卡共同持有模型或缓存，则：

$$
is\_sharded = true
$$

- 若多张卡只是并列跑多个单卡副本，则仍属于多个单卡实例，而不是一个多卡实例

因此：

- `instance_gpus = 1` 是“一组卡”的特例
- “最小服务单元是一张卡还是一组卡”应由求解结果决定，不是预设输入

### 8.3 sharded 与 replicated 的区分

- **replicated**：每个实例保留完整权重与 runtime，cache 在实例内承载
- **sharded**：单实例内部需要跨卡切分权重 / cache / 计算状态

本文的严谨公式完整覆盖 `replicated` 模式；对于 `sharded` 模式，本文提供统一接口，但需通过单实例效率因子和容量口径进行工程近似。

### 8.4 故障单元推导

`failure_unit_gpus` 不是原始输入，而是部署方案确定后的派生结果。

典型规则：

- 单卡独立实例：`failure_unit_gpus = 1`
- 4 卡 TP 单实例：`failure_unit_gpus ≈ 4`
- 若整机 8 卡为最小调度/故障单元：`failure_unit_gpus ≈ 8`

更严格定义：

> 一次最小故障会导致多少 GPU 对应的业务能力同时失效，`failure_unit_gpus` 就取多少。

---

## 9. 单实例资源模型

### 9.1 权重显存

严格按字节：

$$
M_{weights,raw}^{bytes} = total\_params\_b \times 10^9 \times bytes(inference\_precision)
$$

换算成十进制 GB：

$$
M_{weights,raw}^{GB} = \frac{M_{weights,raw}^{bytes}}{10^9}
$$

工程上可简写为：

$$
M_{weights,raw}^{GB} \approx total\_params\_b \times bytes(inference\_precision)
$$

加入额外开销：

$$
M_{weights} = M_{weights,raw}^{GB} \times (1 + weight\_overhead\_ratio)
$$

### 9.2 运行时显存

$$
M_{runtime} = runtime\_overhead\_gb + runtime\_overhead\_ratio \times M_{weights,raw}^{GB}
$$

### 9.3 KV Cache 单 token 每层字节数

若显式给定：

$$
E_{cache/token/layer} = cache\_bytes\_per\_token\_per\_layer
$$

否则：

#### MHA / GQA / MQA

$$
E_{cache/token/layer} = 2 \times num\_kv\_heads \times head\_dim \times bytes(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

其中系数 2 对应 K 和 V。

#### MLA

$$
E_{cache/token/layer} = latent\_cache\_dim \times bytes(kv\_cache\_dtype) + cache\_aux\_bytes\_per\_token\_per\_layer
$$

#### Hybrid / Sparse

缺少精确定义时，建议显式输入 `cache_bytes_per_token_per_layer`，否则使用保守上界估计。

### 9.4 单请求 KV Cache 显存

对序列长度 `seq_len`：

$$
M_{cache,req}(seq\_len) = \frac{num\_layers \times seq\_len \times E_{cache/token/layer}}{10^9}
$$

平均请求：

$$
M_{cache,req}^{avg} = M_{cache,req}(E[S])
$$

保守 P95 请求：

$$
M_{cache,req}^{p95} = M_{cache,req}(S_{p95})
$$

### 9.5 replicated 模式下的单实例 cache 容量

先计算单实例可用显存：

$$
M_{usable,inst} = instance\_gpus \times vram\_gb \times usable\_vram\_ratio
$$

若为 `replicated` 模式，则单实例内部必须先容纳：

$$
M_{base,inst} = M_{weights} + M_{runtime}
$$

要求：

$$
M_{base,inst} < M_{usable,inst}
$$

否则该实例规模不可行。

单实例可用于承载 cache 的容量：

$$
M_{cache,cap,inst} = M_{usable,inst} - M_{base,inst}
$$

### 9.6 单实例总 cache 需求

平均场景：

$$
M_{cache,total}^{avg} = C_{avg} \times M_{cache,req}^{avg}
$$

保守峰值场景：

$$
M_{cache,total}^{p95} = C_{peak} \times M_{cache,req}^{p95}
$$

### 9.7 显存约束实例数

在 `replicated` 模式下，应按实例容量求解，而不是按全局显存池线性平摊。

$$
instance\_count_{memory} = \left\lceil \frac{M_{cache,total}^{p95}}{M_{cache,cap,inst}} \right\rceil
$$

对应 GPU 数：

$$
G_{memory} = instance\_count_{memory} \times instance\_gpus
$$

### 9.8 sharded 模式下的显存建模说明

若为 `sharded` 模式，本文不直接沿用 replicated 公式，而应以**单实例总可用显存**与**单实例实际切分后的 base/capacity**建模：

$$
M_{base,inst}^{sharded} = f_{shard}(M_{weights}, M_{runtime}, instance\_gpus)
$$

$$
M_{cache,cap,inst}^{sharded} = M_{usable,inst} - M_{base,inst}^{sharded}
$$

此处 `f_shard` 取决于 TP / EP / cache 切分方式。若无额外建模数据，建议在工具实现中将其作为外部 profile / 配置输入，而不要伪装成严格公式。

---

## 10. 单实例吞吐模型

### 10.1 计算峰值选择

根据精度选择峰值算力 `peak_compute_tflops`。例如：

- BF16 推理用 `bf16_tflops`
- FP8 推理用 `fp8_tflops`
- INT8 推理用 `int8_tflops`
- INT4 / W4A16 推理用 `int4_tflops`

有效算力：

$$
peak\_compute\_effective = peak\_compute\_tflops \times compute\_efficiency
$$

### 10.2 Decode 单实例吞吐

每输出一个 token，需要近似扫过激活参数：

$$
P_{act} = \begin{cases}
activated\_params\_per\_token\_b, & \text{MoE} \\
total\_params\_b, & \text{Dense}
\end{cases}
$$

#### 带宽上界

$$
TPS_{decode,mem}^{inst} = \frac{instance\_gpus \times memory\_bandwidth\_gb\_per\_sec}{P_{act} \times bytes(inference\_precision)}
$$

#### 计算上界

取每 token 近似 FLOPs：

$$
F_{decode/token} \approx 2 \times P_{act} \times 10^9
$$

则：

$$
TPS_{decode,compute}^{inst} = \frac{peak\_compute\_effective \times 10^{12} \times instance\_gpus}{F_{decode/token}}
$$

#### Decode 单实例 spec 吞吐

$$
TPS_{decode}^{inst} = min(TPS_{decode,mem}^{inst},\ TPS_{decode,compute}^{inst}) \times decode\_efficiency
$$

### 10.3 Prefill 单实例吞吐

#### Prefill 每 token 内存 IO

定义：

$$
B_{prefill/token} = B_{weight/token} + B_{act\_io/token}
$$

其中：

$$
B_{weight/token} \approx total\_params\_b \times bytes(inference\_precision)
$$

为十进制 GB/token 的工程近似。

额外 activation / attention IO 用经验系数吸收：

$$
B_{act\_io/token} = \alpha_{prefill\_io} \times B_{weight/token}
$$

因此：

$$
B_{prefill/token} = (1 + \alpha_{prefill\_io}) \times total\_params\_b \times bytes(inference\_precision)
$$

其中 `alpha_prefill_io` 建议取 `0.1 ~ 0.5`，由框架与 batch 特征决定。

#### 带宽上界

$$
TPS_{prefill,mem}^{inst} = \frac{instance\_gpus \times memory\_bandwidth\_gb\_per\_sec \times prefill\_memory\_reuse\_factor}{B_{prefill/token}}
$$

#### 计算上界

Prefill 对整段输入做全量前向，仍近似取：

$$
F_{prefill/token} \approx 2 \times total\_params\_b \times 10^9
$$

则：

$$
TPS_{prefill,compute}^{inst} = \frac{peak\_compute\_effective \times 10^{12} \times instance\_gpus}{F_{prefill/token}}
$$

#### Prefill 单实例 spec 吞吐

$$
TPS_{prefill}^{inst} = min(TPS_{prefill,mem}^{inst},\ TPS_{prefill,compute}^{inst}) \times prefill\_efficiency
$$

### 10.4 关于 sharded 模式的吞吐说明

若 `instance_gpus > 1` 且 `is_sharded = true`，则上式仍只可视为单实例工程近似，需要额外引入：

- `eta_decode_inst`
- `eta_prefill_inst`

修正为：

$$
TPS_{decode}^{inst,sharded} = TPS_{decode}^{inst} \times eta_{decode,inst}
$$

$$
TPS_{prefill}^{inst,sharded} = TPS_{prefill}^{inst} \times eta_{prefill,inst}
$$

其中 `eta` 由多卡通信与框架实现决定，通常 `< 1`。

---

## 11. 从单实例能力到业务 GPU 数

### 11.1 吞吐约束实例数

#### Decode

$$
instance\_count_{decode} = \left\lceil \frac{target\_decode\_tps\_total}{TPS_{decode}^{inst}} \right\rceil
$$

$$
G_{decode} = instance\_count_{decode} \times instance\_gpus
$$

#### Prefill

$$
instance\_count_{prefill} = \left\lceil \frac{target\_prefill\_tps\_total^{upper}}{TPS_{prefill}^{inst}} \right\rceil
$$

$$
G_{prefill} = instance\_count_{prefill} \times instance\_gpus
$$

### 11.2 业务最小 GPU 数

先取业务所需实例数：

$$
instance\_count_{biz} = max(instance\_count_{memory},\ instance\_count_{prefill},\ instance\_count_{decode})
$$

对应业务 GPU 数：

$$
G_{biz} = instance\_count_{biz} \times instance\_gpus
$$

这条公式比直接写：

$$
G_{biz} = max(G_{memory}, G_{prefill}, G_{decode})
$$

更一般，因为它先统一在实例层求解，再换算 GPU 数，更适用于“实例 ≠ 单卡”的情况。

---

## 12. HA 展开与最终 GPU 数

### 12.1 故障单元

部署方案确定后，得到：

$$
failure\_unit\_gpus
$$

或等价地得到故障单元实例数：

$$
failure\_unit\_instances = \frac{failure\_unit\_gpus}{instance\_gpus}
$$

要求该值为正整数。

### 12.2 常见 HA 模式

#### 无高可用

$$
G_{final} = G_{biz}
$$

#### N+1 / 可承受一个故障单元失效

$$
G_{final} = G_{biz} + failure\_unit\_gpus
$$

#### 冗余比例模式

$$
G_{final} = \left\lceil G_{biz} \times (1 + failover\_reserve\_ratio) \right\rceil
$$

#### Active-Active

若要求 `replica_count` 个业务副本同时可独立承载，则：

$$
G_{final} = replica\_count \times G_{biz}
$$

若还要求单副本内具备 N+1，则应在每个副本上再加故障单元。

### 12.3 节点级故障

若 HA 目标是承受 1 台整机故障，且单节点 GPU 数为 `node_gpu_count`，则：

$$
failure\_unit\_gpus \ge node\_gpu\_count
$$

---

## 13. 基于已推导 GPU 数的预期效果反推

本章回答：

> 在已求得 `G_biz` 或 `G_final` 后，这套资源大致能达到什么效果？

为避免误导，区分两种口径：

- **业务基线能力**：按本文 sizing 假设与目标 SLA 运行时的可持续能力
- **工程上限能力**：在相同 spec 假设下，资源压满时的吞吐上界

### 13.1 集群业务基线吞吐

业务实例数：

$$
instance\_count_{biz} = \frac{G_{biz}}{instance\_gpus}
$$

则：

$$
TPS_{decode,cluster} = instance\_count_{biz} \times TPS_{decode}^{inst}
$$

$$
TPS_{prefill,cluster} = instance\_count_{biz} \times TPS_{prefill}^{inst}
$$

### 13.2 可持续并发能力

#### Decode 侧

每个 decode-active 请求的 P95 速率需求为 `R_{decode,req}^{p95}`，因此可持续 decode-active 并发：

$$
ConcurrentCapacity_{decode-active} = \frac{TPS_{decode,cluster}}{R_{decode,req}^{p95}}
$$

折算成总活跃并发：

$$
ConcurrentCapacity_{sustainable,p95} = \frac{ConcurrentCapacity_{decode-active}}{r_{decode}}
$$

#### Prefill 侧上界

$$
ConcurrentCapacity_{prefill,upper} = \frac{TPS_{prefill,cluster}}{R_{prefill,req}^{p95}}
$$

说明：

- decode 侧更接近稳态可持续并发
- prefill 侧更接近短时突发容量上界

### 13.3 工程上限吞吐

在本文的 spec 假设下：

$$
TPS_{decode,cluster}^{max,spec} \approx TPS_{decode,cluster}
$$

$$
TPS_{prefill,cluster}^{max,spec} \approx TPS_{prefill,cluster}
$$

说明：

- 这里的“上限”不是指硬件绝对物理峰值
- 而是在本文效率参数与部署假设下的工程可用上界

### 13.4 单请求生成速度

#### 平均场景

若平均并发为 `C_avg`，则平均 decode-active 请求数约为：

$$
C_{decode,avg} = C_{avg} \times r_{decode}
$$

则平均单请求生成速度：

$$
Speed_{decode,req}^{avg} \approx \frac{TPS_{decode,cluster}}{C_{decode,avg}}
$$

#### 峰值场景

$$
Speed_{decode,req}^{peak} \approx \frac{TPS_{decode,cluster}}{C_{peak} \times r_{decode}}
$$

单位均为 tokens/s/request。

### 13.5 时延近似反推

#### 平均 TTFT

$$
TTFT_{avg,approx} \approx \frac{S_{in,avg}^{effective}}{TPS_{prefill,cluster} / C_{avg}}
$$

#### 平均 Decode 时长

$$
T_{decode,avg,approx} \approx \frac{S_{out,avg}}{TPS_{decode,cluster} / (C_{avg} \times r_{decode})}
$$

#### 平均 E2E

$$
Latency_{e2e,avg,approx} \approx TTFT_{avg,approx} + T_{decode,avg,approx}
$$

#### 保守 P95 TTFT

$$
TTFT_{p95,upper} \approx \frac{S_{in,p95}^{effective}}{TPS_{prefill,cluster} / C_{peak}}
$$

#### 保守 P95 Decode 时长

$$
T_{decode,p95,upper} \approx \frac{S_{out,p95}}{TPS_{decode,cluster} / (C_{peak} \times r_{decode})}
$$

#### 保守 P95 E2E

$$
Latency_{e2e,p95,upper} \approx TTFT_{p95,upper} + T_{decode,p95,upper}
$$

说明：

- 以上近似默认资源在请求间公平分享
- 未显式建模排队、调度抖动和缓存命中波动
- 仅用于 sanity check，不可反向视为 SLA 承诺

### 13.6 每日产能

若每日业务运行时间为 `86400 sec`，则：

$$
DailyTokens_{decode} = TPS_{decode,cluster} \times 86400
$$

$$
DailyTokens_{prefill} = TPS_{prefill,cluster} \times 86400
$$

若业务主要受输出 token 限制，则通常取 `DailyTokens_decode` 作为更关键指标。

---

## 14. 结果输出建议

建议输出以下字段：

### 14.1 输入与方案

- `ModelName`
- `GPUName`
- `Framework`
- `HAMode`
- `InstanceGPUs`
- `IsSharded`
- `FailureUnitGPUs`

### 14.2 资源规模

- `InstanceCountMemory`
- `InstanceCountPrefill`
- `InstanceCountDecode`
- `InstanceCountBiz`
- `GMemory`
- `GPrefill`
- `GDecode`
- `GBiz`
- `GFinal`

### 14.3 集群能力

- `TPSPrefillCluster`
- `TPSDecodeCluster`
- `ConcurrentCapacitySustainableP95`
- `ConcurrentCapacityPrefillUpper`
- `DecodeSpeedAvg`
- `DecodeSpeedPeak`

### 14.4 时延近似

- `TTFTAvgApproxSec`
- `TTFTP95UpperSec`
- `DecodeAvgApproxSec`
- `DecodeP95UpperSec`
- `E2EAvgApproxSec`
- `E2EP95UpperSec`

### 14.5 产能

- `DailyTokensDecode`
- `DailyTokensPrefill`

---

## 15. 使用建议与常见误区

### 15.1 不要把部署方案当输入写死

以下量通常应由求解结果给出，而不是在输入层一开始就定死：

- `instance_gpus`
- `is_sharded`
- `failure_unit_gpus`

它们应该由：

- 模型
- GPU
- 框架
- 业务目标
- HA 目标

共同决定。

### 15.2 不要把全局显存池简单平摊

错误写法：

$$
G_{memory} = \left\lceil \frac{M_{weights} + M_{runtime} + M_{cache,total}}{M_{usable,gpu}} \right\rceil
$$

在 replicated 模式下会低估卡数，因为 `weights + runtime` 是按实例重复出现的。

### 15.3 不要把 `P95(e2e) - P95(TTFT)` 当严格 decode P95

分位数不可线性相减，只能作为预算分配近似，不能写成严格恒等式。

### 15.4 不要把“效果反推”误当成 SLA 保证

第 13 章输出的并发、速度和时延是：

- 在本文假设下的工程近似
- 用于结果解释与 sanity check

而不是线上承诺值。

### 15.5 `decode_active_ratio` 应做敏感性分析

该参数高度依赖：

- 请求结构
- 输出长度分布
- 调度策略
- 峰值负载形态

建议至少做 `0.4 / 0.6 / 0.8` 三档敏感性分析。

---

## 16. 一页式总结

本文推荐的完整主线为：

1. **输入层**：明确业务目标、模型、GPU、框架、HA 选择
2. **求解层**：判断单卡可行性，枚举多卡实例与 sharding 方案，推导 `instance_gpus / is_sharded / failure_unit_gpus`
3. **资源层**：基于单实例显存与吞吐，求得 `G_memory / G_prefill / G_decode / G_biz / G_final`
4. **解释层**：基于已求得 GPU 数，反推并发、吞吐、速度、时延与每日产能

一句话概括：

> **输入要求与候选条件 → 求解部署方案 → 输出 GPU 数 → 反推可达到效果**

这比“先假设部署口径，再算卡数”的写法更符合真实工程 sizing 流程。

