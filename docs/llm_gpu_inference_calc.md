# 大模型在线推理 GPU 资源测算方法

---

## 1. 文档目标

本文用于在线推理场景下的大模型 GPU 资源测算，给出一套从**输入条件**出发，经过**部署方案求解**，最终输出**GPU 数量与可达到效果**的完整方法。

本文目标不是复现精确 benchmark，而是提供一套：

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
- 基于已推导 GPU 数可达到的并发、吞吐、生成速度、时延等预期效果

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
2. 单实例内部 TP / EP / PP 的通信开销只通过实例级扩展效率近似吸收，不做精细链路建模
3. Prefill 吞吐在缺少到达率信息时采用保守上界近似
4. 平均 / P95 对话时延反推属于 sanity check，不构成 SLA 证明
5. 多维 P95 长度拼装属于保守 sizing，而非严格联合分布建模
6. 本文核心推导中的存储统一用 `byte`，token 统一用 `token`，计算量统一用 `FLOP`，时间统一用 `second`；仅在结果展示层换算成人类易读单位

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

输入包括：

- 业务目标输入
- 模型输入
- GPU 选择输入
- 推理框架输入
- HA 输入

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
- 可用吞吐上限
- 单请求生成速度
- 平均 / 保守 P95 时延近似
- 每日产能

---

## 4. 统一术语与记号

### 4.1 基本对象

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

### 4.4 基础单位约定

除最终展示外，全文中间计算统一采用：

- 存储、显存、带宽：`byte`、`byte/s`
- token 长度与吞吐：`token`、`token/s`
- 计算量与算力：`FLOP`、`FLOP/s`
- 时间：`second`

辅助函数：

- `bytes_per_param(inference_precision)`：每参数字节数
- `bytes_per_cache_element(kv_cache_dtype)`：KV cache 元素字节数
- `peak_compute_flops_per_sec(inference_precision)`：按推理精度选取 GPU 对应峰值算力，单位 `FLOP/s`

### 4.5 记号映射

- `S_{in,avg}`：平均输入长度，单位 `token`
- `S_{out,avg}`：平均输出长度，单位 `token`
- `S_{in,p95}`：P95 输入长度，单位 `token`
- `S_{out,p95}`：P95 输出长度，单位 `token`
- `S_{p95}`：P95 总长度，单位 `token`
- `C_avg`：平均活跃并发
- `C_peak`：峰值活跃并发
- `r_decode`：峰值时 decode-active 比例
- `r_prefix`：prefix cache 命中率
- `r_prefill`：平均时延近似中的 prefill-active 比例

---

## 5. 输入条件

### 5.1 输入分类总表

| 分类 | 是否必填 | 说明 |
| --- | --- | --- |
| 业务目标输入 | 是 | 定义业务规模与体验目标 |
| 模型输入 | 是 | 定义模型结构、规模与精度 |
| GPU 选择输入 | 是 | 定义 GPU 规格与有效利用率假设 |
| 推理框架输入 | 是 | 定义框架选择及其推荐画像参数 |
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
| `request_shapes` | 请求长度分布，采用最小版 |
| `decode_active_ratio` | decode-active 比例 |
| `prefix_cache_hit_rate` | prefix cache 命中率 |
| `prefill_active_ratio` | 平均时延近似中的 prefill-active 比例 |

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

- 所有 `weight_i > 0`
- `Σ weight_i = 1`
- 每个 shape 至少包含 `name / weight / input_tokens_avg / input_tokens_p95 / output_tokens_avg / output_tokens_p95`

推荐默认值：

```json
{
  "request_shapes": [
    {
      "name": "default",
      "weight": 1.0,
      "input_tokens_avg": 1000,
      "input_tokens_p95": 4000,
      "output_tokens_avg": 300,
      "output_tokens_p95": 800
    }
  ]
}
```

#### 5.2.2 `decode_active_ratio`（必填）

定义域：

`0 < decode_active_ratio <= 1`

推荐默认值：

- 默认：`0.7`
- 短输出问答类：`0.5 ~ 0.7`
- 长输出生成类：`0.7 ~ 0.9`

#### 5.2.3 `prefix_cache_hit_rate`（必填）

定义域：

`0 <= prefix_cache_hit_rate <= 1`

推荐默认值：

- 默认：`0.0`
- 若存在稳定 system prompt / 模板复用 / 固定前缀，可从 `0.2` 起步

#### 5.2.4 `prefill_active_ratio`（必填）

定义域：

`0 < prefill_active_ratio <= 1`

推荐默认值：

- 默认：`0.3`
- 短输入问答类：`0.2 ~ 0.3`
- 长输入 / 长上下文类：`0.3 ~ 0.5`

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
| `cache_aux_bytes_per_token_per_layer` | 每 token 每层额外 cache 开销，单位 `byte` |
| `cache_bytes_per_token_per_layer` | 直接指定 KV Cache 字节数，单位 `byte`，优先级最高 |

### 5.4 GPU 选择输入（必填）

| 参数 | 含义 |
| --- | --- |
| `gpu_name` | GPU 名称 |
| `vram_bytes` | 单卡显存，单位 `byte` |
| `memory_bandwidth_bytes_per_sec` | HBM 带宽，单位 `byte/s` |
| `fp16_flops_per_sec` | FP16 峰值算力，单位 `FLOP/s` |
| `bf16_flops_per_sec` | BF16 峰值算力，单位 `FLOP/s` |
| `fp8_flops_per_sec` | FP8 峰值算力，单位 `FLOP/s` |
| `int8_flops_per_sec` | INT8 峰值算力，单位 `FLOP/s` |
| `int4_flops_per_sec` | INT4 / W4A16 峰值算力，单位 `FLOP/s` |
| `memory_bandwidth_efficiency` | 有效带宽折减系数 |
| `compute_efficiency` | 有效算力折减系数 |
| `usable_vram_ratio` | 单卡可用显存比例 |

推荐默认值：

- `memory_bandwidth_efficiency = 0.70`
- `compute_efficiency = 0.35`
- `usable_vram_ratio = 0.90`

### 5.5 推理框架输入（必填）

| 参数 | 含义 |
| --- | --- |
| `serving_framework` | `vLLM` / `SGLang` / `TensorRT-LLM` / 自研 |
| `framework_version` | 版本 |
| `support_prefix_cache` | 是否支持 prefix cache |
| `support_continuous_batching` | 是否支持 continuous batching |
| `weight_overhead_ratio` | 权重额外显存系数 |
| `runtime_buffer_ratio` | 运行时固定显存系数 |
| `prefill_memory_reuse_factor` | Prefill 批内复用系数 |

推荐默认值：

- `weight_overhead_ratio = 0.05`
- `runtime_buffer_ratio = 0.10`
- `prefill_memory_reuse_factor = 1.00`

说明：

- `weight_overhead_ratio` 只吸收权重格式、量化元数据、内存布局等额外显存开销
- `runtime_buffer_ratio` 只吸收运行时必需 buffer、workspace、allocator 常驻开销
- `usable_vram_ratio` 只吸收 OOM 安全水位、波动预留与运营安全边界
- `prefill_memory_reuse_factor` 仅用于 Prefill 带宽模型，反映批内多个 token 对权重访问的分摊效应

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

- `concurrency_avg > 0`
- `concurrency_peak >= concurrency_avg`
- `target_ttft_avg_sec > 0`
- `target_ttft_p95_sec >= target_ttft_avg_sec`
- `target_e2e_avg_sec > target_ttft_avg_sec`
- `target_e2e_p95_sec > target_ttft_p95_sec`
- `0 < decode_active_ratio <= 1`
- `0 <= prefix_cache_hit_rate <= 1`
- `0 < prefill_active_ratio <= 1`
- 若 `support_prefix_cache = false`，则 `prefix_cache_hit_rate = 0`
- `0 < usable_vram_ratio <= 1`
- `Σ request_shapes.weight = 1`
- `memory_bandwidth_efficiency > 0`
- `compute_efficiency > 0`

### 6.2 从 `request_shapes` 派生全局长度统计

全局平均输入长度：

`S_{in,avg} = Σ_i (w_i × S_{in,avg,i})`

全局平均输出长度：

`S_{out,avg} = Σ_i (w_i × S_{out,avg,i})`

全局 P95 输入与输出长度：

- 若有原始离散桶分布，可按累计分位近似求取
- 若仅有少量离散 shape，工程上可取“高位主导 shape”的保守近似

记最终派生结果为：

- `S_{in,p95}`
- `S_{out,p95}`

### 6.3 总长度 P95

优先顺序：

1. 若业务侧单独提供 `target_total_tokens_p95`，优先使用
2. 若未提供，则保守近似：

`S_{p95} = S_{in,p95} + S_{out,p95}`

### 6.4 有效输入长度

考虑 prefix cache 后：

- `S_{in,avg}^{eff} = S_{in,avg} × (1 - r_prefix)`
- `S_{in,p95}^{eff} = S_{in,p95} × (1 - r_prefix)`

其中 `r_prefix = prefix_cache_hit_rate`。

---

## 7. 从体验目标到系统需求

### 7.1 平均阶段预算

平均 decode 预算：

`T_{decode,avg} = target_e2e_avg_sec - target_ttft_avg_sec`

默认平均 prefill 预算份额：

`ttft_prefill_share_avg = 0.8`

平均单请求 prefill 速率需求：

`R_{prefill,req}^{avg} = S_{in,avg}^{eff} / (target_ttft_avg_sec × ttft_prefill_share_avg)`

平均单请求 decode 生成速率需求：

`R_{decode,req}^{avg} = S_{out,avg} / T_{decode,avg}`

### 7.2 P95 阶段预算

- 若业务侧直接给出 `target_decode_p95_sec`，则优先使用
- 否则采用工程预算近似：

`T_{decode,p95}^{budget} = target_e2e_p95_sec - target_ttft_p95_sec`

默认 P95 prefill 预算份额：

`ttft_prefill_share_p95 = 0.8`

P95 单请求 prefill 速率需求：

`R_{prefill,req}^{p95} = S_{in,p95}^{eff} / (target_ttft_p95_sec × ttft_prefill_share_p95)`

P95 单请求 decode 生成速率需求：

`R_{decode,req}^{p95} = S_{out,p95} / T_{decode,p95}^{budget}`

### 7.3 系统总吞吐需求

平均 decode 总吞吐需求：

`TPS_{decode,target}^{avg} = C_{avg} × r_decode × R_{decode,req}^{avg}`

峰值 decode 总吞吐需求：

`TPS_{decode,target}^{peak} = C_{peak} × r_decode × R_{decode,req}^{p95}`

Prefill 总吞吐需求采用保守上界：

`TPS_{prefill,target}^{peak} = C_{peak} × R_{prefill,req}^{p95}`

---

## 8. 单实例显存模型

### 8.1 权重显存

原始权重字节数：

`M_{weights,raw,bytes} = total_params_b × 1e9 × bytes_per_param(inference_precision)`

加上额外权重开销：

`M_{weights,bytes} = M_{weights,raw,bytes} × (1 + weight_overhead_ratio)`

### 8.2 KV Cache 每 token 每层字节数

若显式给出：

`E_{cache/token/layer,bytes} = cache_bytes_per_token_per_layer`

否则：

- MHA：`E_{cache/token/layer,bytes} = 2 × hidden_size × bytes_per_cache_element(kv_cache_dtype) + cache_aux_bytes_per_token_per_layer`
- GQA：`E_{cache/token/layer,bytes} = 2 × num_kv_heads × head_dim × bytes_per_cache_element(kv_cache_dtype) + cache_aux_bytes_per_token_per_layer`
- MQA：`E_{cache/token/layer,bytes} = 2 × head_dim × bytes_per_cache_element(kv_cache_dtype) + cache_aux_bytes_per_token_per_layer`
- MLA：`E_{cache/token/layer,bytes} = latent_cache_dim × bytes_per_cache_element(kv_cache_dtype) + cache_aux_bytes_per_token_per_layer`

注：若实现上将 MQA 统一为 `num_kv_heads = 1` 的 GQA 特例，也可复用 GQA 公式，但必须显式满足 `num_kv_heads = 1`。

### 8.3 单请求 KV Cache

对给定序列长度 `S`：

`M_{cache,req,bytes}(S) = num_layers × S × E_{cache/token/layer,bytes}`

### 8.4 运行时固定显存

`M_{runtime,bytes} = runtime_buffer_ratio × M_{weights,bytes}`

### 8.5 单实例总显存需求

平均负载：

`M_{inst,total,avg,bytes} = M_{weights,bytes} + M_{runtime,bytes} + N_{req/inst}^{avg} × M_{cache,req,bytes}(S_{avg})`

峰值保守上界：

`M_{inst,total,p95,bytes} = M_{weights,bytes} + M_{runtime,bytes} + N_{req/inst}^{peak} × M_{cache,req,bytes}(S_{p95})`

其中：

- `S_{avg} = S_{in,avg} + S_{out,avg}`
- `N_{req/inst}` 为单实例承载请求数

### 8.6 单实例显存可行性（实例级 pooled/sharded 上界）

当 `instance_gpus = 1` 时，本节与 replicated 严格口径一致。

当 `instance_gpus > 1` 时，本文在前期 sizing 中采用**实例级 pooled/sharded 可行性上界**近似，即把单实例内多张卡的可用显存视为一个可协同利用的上界池；这更适合作为多卡 / sharded 方案的前期可行性判断，而不应被解读为 replicated 多卡实例的严格显存证明。

单实例可用显存：

`M_{usable/inst,bytes} = instance_gpus × vram_bytes × usable_vram_ratio`

若：

`M_{weights,bytes} + M_{runtime,bytes} >= M_{usable/inst,bytes}`

则当前 `instance_gpus` 方案不可行。

单实例可承载 cache 容量：

`M_{cache,cap/inst,bytes} = M_{usable/inst,bytes} - M_{weights,bytes} - M_{runtime,bytes}`

---

## 9. 单实例吞吐模型

### 9.1 激活参数量定义

令：

- Dense：`active_params_b = total_params_b`
- MoE：`active_params_b = activated_params_per_token_b`

并记：

`active_params = active_params_b × 1e9`

单位为参数个数。

### 9.2 Decode 单实例吞吐

单 token 近似访存量：

`bytes_{decode/token} = active_params × bytes_per_param(inference_precision)`

单 token 近似计算量：

`flops_{decode/token} = 2 × active_params`

其中 `peak_compute_flops_per_sec(inference_precision)` 按推理精度选择：

- FP16 → `fp16_flops_per_sec`
- BF16 → `bf16_flops_per_sec`
- FP8 → `fp8_flops_per_sec`
- INT8 → `int8_flops_per_sec`
- INT4 / W4A16 → `int4_flops_per_sec`

内存受限近似：

`TPS_{decode,memory}^{gpu} = (memory_bandwidth_bytes_per_sec × memory_bandwidth_efficiency) / bytes_{decode/token}`

算力受限近似：

`TPS_{decode,compute}^{gpu} = (peak_compute_flops_per_sec(inference_precision) × compute_efficiency) / flops_{decode/token}`

单 GPU decode 吞吐：

`TPS_{decode}^{gpu} = min(TPS_{decode,memory}^{gpu}, TPS_{decode,compute}^{gpu})`

若单实例为多卡，实例级扩展效率记为 `inst_scale_efficiency`，默认 `1.0`。

单实例 decode 吞吐：

`TPS_{decode}^{inst} = instance_gpus × TPS_{decode}^{gpu} × inst_scale_efficiency`

说明：

- 上式是前期 sizing 近似，attention、norm、rope、softmax、MLA 等额外 FLOPs 与 IO 被吸收到有效利用率与实例扩展效率中
- 若有实测值，应优先替换

### 9.3 Prefill 单实例吞吐

先定义不含批内复用的基线单 token 访存量：

`bytes_{prefill/token,base} = active_params × bytes_per_param(inference_precision)`

考虑批内复用后的有效单 token 访存量：

`bytes_{prefill/token,effective} = bytes_{prefill/token,base} / prefill_memory_reuse_factor`

内存受限上界：

`TPS_{prefill,memory}^{gpu} = (memory_bandwidth_bytes_per_sec × memory_bandwidth_efficiency) / bytes_{prefill/token,effective}`

单 token 近似计算量：

`flops_{prefill/token} = 2 × active_params`

算力受限上界：

`TPS_{prefill,compute}^{gpu} = (peak_compute_flops_per_sec(inference_precision) × compute_efficiency) / flops_{prefill/token}`

单 GPU prefill 吞吐：

`TPS_{prefill}^{gpu} = min(TPS_{prefill,memory}^{gpu}, TPS_{prefill,compute}^{gpu})`

单实例 prefill 吞吐：

`TPS_{prefill}^{inst} = instance_gpus × TPS_{prefill}^{gpu} × inst_scale_efficiency`

说明：

- `prefill_memory_reuse_factor` 不表示突破物理带宽，而是表示 batch 内多个 token 对权重访问的分摊下降
- 这里把它显式作用在 `bytes_{prefill/token,effective}`，避免与基线访存量重复含义

---

## 10. 部署方案求解

### 10.1 单卡可行性判断

从 `instance_gpus = 1` 开始，依次检查：

1. 显存是否可容纳 `M_{weights,bytes} + M_{runtime,bytes}`
2. 单实例显存是否足够承载目标请求数对应的 cache
3. 单实例 prefill / decode 吞吐是否满足需求

若单卡可行，则：

- `instance_gpus = 1`
- `is_sharded = false`

### 10.2 多卡实例枚举

若单卡不可行，则枚举：

- `instance_gpus ∈ {2, 4, 8, ...}`

对每个候选 `instance_gpus`：

1. 检查显存可行性
2. 估算 `TPS_{prefill}^{inst}` 与 `TPS_{decode}^{inst}`
3. 评估是否需要 sharding

若必须通过 TP / EP / Cache Sharding 才可满足，则：

- `is_sharded = true`

### 10.3 故障单元推导

- 若单卡独立实例：通常 `failure_unit_gpus = 1`
- 若多卡强耦合实例：通常 `failure_unit_gpus = instance_gpus`
- 若按整节点绑定故障：可取 `failure_unit_gpus = node_gpu_count`

---

## 11. 业务所需实例数与 GPU 数

### 11.1 Decode 约束实例数

`N_{inst,decode} = ceil(TPS_{decode,target}^{peak} / TPS_{decode}^{inst})`

### 11.2 Prefill 约束实例数

`N_{inst,prefill} = ceil(TPS_{prefill,target}^{peak} / TPS_{prefill}^{inst})`

### 11.3 显存约束实例数

先计算单实例在峰值保守场景下可承载的请求数：

`N_{req/inst,mem}^{max} = floor(M_{cache,cap/inst,bytes} / M_{cache,req,bytes}(S_{p95}))`

若 `N_{req/inst,mem}^{max} <= 0`，则当前实例方案不可行。

显存约束实例数：

`N_{inst,memory} = ceil(C_{peak} / N_{req/inst,mem}^{max})`

### 11.4 业务实例数与业务 GPU 数

`N_{inst,biz} = max(N_{inst,decode}, N_{inst,prefill}, N_{inst,memory})`

`G_{biz} = N_{inst,biz} × instance_gpus`

---

## 12. 高可用展开

### 12.1 HA 后实例数

- `ha_mode = none`：`N_{inst,final} = N_{inst,biz}`
- `ha_mode = survive_failure_unit`：

`N_{inst,final} = N_{inst,biz} + ceil((ha_target_units × failure_unit_gpus) / instance_gpus)`

- `ha_mode = survive_node`：需按节点粒度折算
- `ha_mode = N+1`：可按至少增加 1 个实例处理

### 12.2 最终 GPU 数

`G_{final} = N_{inst,final} × instance_gpus`

---

## 13. 基于已推导 GPU 数的预期效果反推

### 13.1 集群吞吐能力

业务集群 decode 吞吐：

`TPS_{decode,cluster} = N_{inst,biz} × TPS_{decode}^{inst}`

业务集群 prefill 吞吐：

`TPS_{prefill,cluster} = N_{inst,biz} × TPS_{prefill}^{inst}`

### 13.2 可持续并发能力

先定义 decode-active 可持续并发：

`C_{decode,sustainable} = TPS_{decode,cluster} / R_{decode,req}^{p95}`

再折算总活跃并发：

`C_{total,sustainable,decode} = C_{decode,sustainable} / r_decode`

Prefill 可持续并发：

`C_{total,sustainable,prefill} = TPS_{prefill,cluster} / R_{prefill,req}^{p95}`

最终可持续并发能力：

`ConcurrentCapacity_{sustainable} = min(C_{total,sustainable,decode}, C_{total,sustainable,prefill})`

### 13.3 单请求生成速度

`Speed_{decode/request} ≈ TPS_{decode,cluster} / (C_{peak} × r_decode)`

### 13.4 平均时延近似

平均 TTFT 近似：

`TTFT_{avg,est} ≈ S_{in,avg}^{eff} / (TPS_{prefill,cluster} / (C_{avg} × r_{prefill}))`

平均 E2E 近似：

`E2E_{avg,est} ≈ S_{in,avg}^{eff} / (TPS_{prefill,cluster} / (C_{avg} × r_{prefill})) + S_{out,avg} / (TPS_{decode,cluster} / (C_{avg} × r_decode))`

说明：

- 该式是工程近似与 sanity check，不是队列论严格推导
- `r_prefill` 只是平均 prefill 竞争程度近似

### 13.5 保守 P95 时延近似

`TTFT_{p95,upper} ≈ S_{in,p95}^{eff} / (TPS_{prefill,cluster} / C_{peak})`

`E2E_{p95,upper} ≈ S_{in,p95}^{eff} / (TPS_{prefill,cluster} / C_{peak}) + S_{out,p95} / (TPS_{decode,cluster} / (C_{peak} × r_decode))`

### 13.6 每日产能

`DailyDecodeTokens = TPS_{decode,cluster} × 86400`

`DailyPrefillTokens = TPS_{prefill,cluster} × 86400`

---

## 14. 输出项建议

### 14.1 输入摘要

- 并发、TTFT、E2E 目标
- request_shapes 摘要
- `decode_active_ratio`
- `prefix_cache_hit_rate`
- `prefill_active_ratio`
- 模型 / GPU / 框架 / HA 选择
- 是否使用默认值及其清单

### 14.2 部署求解结果

- `instance_gpus`
- `is_sharded`
- `failure_unit_gpus`
- `TPS_{prefill}^{inst}`
- `TPS_{decode}^{inst}`
- `N_{inst,memory}` / `N_{inst,prefill}` / `N_{inst,decode}`

### 14.3 资源规模结果

- `N_{inst,biz}`
- `G_{biz}`
- `N_{inst,final}`
- `G_{final}`

### 14.4 效果反推结果

- `TPS_{prefill,cluster}`
- `TPS_{decode,cluster}`
- `ConcurrentCapacity_{sustainable}`
- `Speed_{decode/request}`
- `TTFT_{avg,est}` / `TTFT_{p95,upper}`
- `E2E_{avg,est}` / `E2E_{p95,upper}`
- `DailyDecodeTokens`
- `DailyPrefillTokens`

### 14.5 结果展示单位建议

为避免中间推导混入单位换算，建议仅在展示层做如下换算：

- `byte -> GB`：`value_GB = value_bytes / 1e9`
- `byte/s -> GB/s`：`value_GBps = value_bytes_per_sec / 1e9`
- `FLOP/s -> TFLOP/s`：`value_TFLOPS = value_flops_per_sec / 1e12`
- `FLOP -> TFLOP`：`value_TFLOP = value_flops / 1e12`

若内部实现采用二进制单位，也应统一在展示层完成，不应混入核心公式。

---

## 15. 使用建议

### 15.1 关于必填但允许默认值的理解

本版文档将 `request_shapes`、`decode_active_ratio`、`prefix_cache_hit_rate` 设为必填，不意味着业务侧一开始就必须拥有精确真实值，而是要求：

- 这三项必须在测算输入中显式填写
- 若没有真实观测，可采用推荐默认值
- 最终结果必须注明哪些结论建立在默认假设之上

### 15.2 推荐的使用顺序

1. 先用默认值跑通首版 sizing
2. 再根据实际业务与线上日志替换 `request_shapes`
3. 根据 profiling 或运营数据更新 `decode_active_ratio`
4. 根据 prefix cache 实测命中更新 `prefix_cache_hit_rate`
5. 根据框架实际加载与运行数据更新 `weight_overhead_ratio`、`runtime_buffer_ratio`
6. 根据 benchmark 或 profiling 更新 GPU 利用率参数

### 15.3 最容易误用的点

- 把默认 `request_shapes` 当成真实业务分布
- 把 `decode_active_ratio` 当成 GPU 或框架固定属性
- 在不支持 prefix cache 的框架上填入正的 `prefix_cache_hit_rate`
- 把 `memory_bandwidth_efficiency` / `compute_efficiency` 误认为芯片天生常数
- 把效果反推结果误认为 SLA 保证值
- 在中间计算里混用 `GB`、`TFLOP`、`B(十亿参数)` 等展示单位

---

## 16. 一页式总结

本文最终采用以下主线：

**输入条件 → 部署方案求解 → 业务 GPU 数 → HA 后最终 GPU 数 → 基于 GPU 数反推可达到效果**

其中：

- 业务目标输入收敛为并发、TTFT、E2E、`request_shapes`、`decode_active_ratio`、`prefix_cache_hit_rate`、`prefill_active_ratio`
- 模型输入定义模型结构、规模与精度
- GPU 选择输入同时包含 GPU 理论规格与有效利用率假设
- 推理框架输入同时包含框架选择与推荐画像参数
- HA 输入单独定义可用性目标
- `instance_gpus`、`is_sharded`、`failure_unit_gpus` 是部署求解结果，而非原始输入
- 核心推导中的存储统一用 `byte`，token 统一用 `token`，计算量统一用 `FLOP`，时间统一用 `second`

因此，本文不是“先假设部署口径，再反推卡数”，而是：

**先给定业务与技术输入，再求最小可行部署方案与所需 GPU 数量。**
