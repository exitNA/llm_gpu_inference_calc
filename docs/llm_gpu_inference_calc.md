# 大模型推理 GPU 资源计算原理

## 1. 目标

本文档用于大模型在线推理场景下的 GPU 资源估算，输出：

* 显存约束卡数
* decode 吞吐约束卡数
* prefill 吞吐约束卡数
* 高可用后的最终 GPU 数量

支持：

* Dense / MoE
* MHA / GQA / MQA / MLA / Sparse
* 基于 GPU 厂商规格参数进行前期估算

整体链路如下：

```mermaid
flowchart LR
    A[体验目标]
    B[系统吞吐目标]
    C[显存/吞吐计算]
    D[GPU数量输出]

    A --> B --> C --> D
```

---

## 2. 总体计算链路

整套计算分为三层：

### 第一层：用户体验层

定义用户可感知目标：

* 首 token 延迟 `TTFT`
* 持续生成速度 `Streaming Speed`
* 峰值并发 `Peak Concurrency`
* 输入/输出长度分布

### 第二层：系统吞吐层

将体验目标转换为：

* `target_prefill_tps_total`
* `target_decode_tps_total`

这两项是系统输入约束。

### 第三层：资源层

结合模型结构、显存公式和 GPU 规格，输出：

* `G_memory`
* `G_prefill`
* `G_decode`
* `G_final`

---

## 3. 输入参数

### 3.1 模型参数

| 参数                           | 含义                             |
| ---------------------------- | ------------------------------ |
| `arch_family`                | Dense / MoE                    |
| `attention_type`             | MHA / GQA / MQA / MLA / Sparse |
| `total_params_b`             | 总参数量，单位 B                       |
| `activated_params_per_token_b` | 每 token 激活参数量，单位 B，MoE 使用      |
| `num_layers`                 | 层数                             |
| `hidden_size`                | 隐藏维度                           |
| `num_heads`                  | Query heads 数                  |
| `num_kv_heads`               | KV heads 数                     |
| `head_dim`                   | 每个 head 维度                     |
| `latent_cache_dim`           | MLA latent cache 维度            |
| `cache_aux_bytes_per_token_per_layer` | KV Cache 每个 token 的附加开销    |
| `cache_bytes_per_token_per_layer` | 可直接指定单 token cache 大小，覆盖推导逻辑   |

### 3.2 请求参数

| 参数                         | 含义                   |
| -------------------------- | -------------------- |
| `concurrency`              | 同时活跃请求数              |
| `batch_size_per_request`   | 单个请求的 Batch Size，默认 1 |
| `request_shapes`           | 请求长度分布               |
| `target_prefill_tps_total` | 总 prefill token/s 目标 |
| `target_decode_tps_total`  | 总 decode token/s 目标  |

其中 `request_shapes` 至少包含：

* 请求类型
* 占比
* 平均输入长度
* 平均输出长度

### 3.3 用户体验参数

| 参数                                 | 含义               |
| ---------------------------------- | ---------------- |
| `target_ttft_p95_sec`              | P95 首 token 延迟目标 |
| `target_streaming_speed_p95_floor` | P95 最低生成速度       |
| `target_peak_concurrency`          | 峰值并发             |
| `target_input_tokens_avg`          | 平均输入长度           |
| `target_input_tokens_p95`          | P95 输入长度         |
| `target_output_tokens_avg`         | 平均输出长度           |
| `target_output_tokens_p95`         | P95 输出长度         |

### 3.4 GPU 参数

| 参数                      | 含义      |
| ----------------------- | ------- |
| `gpu_name`              | GPU 名称  |
| `vram_gb`               | 显存容量，单位 GB |
| `memory_bandwidth_gb_per_sec` | 显存带宽，单位 GB/s |
| `fp32_tflops`           | FP32 算力 |
| `fp16_tflops`           | FP16 算力 |
| `bf16_tflops`           | BF16 算力 |
| `fp8_tflops`            | FP8 算力  |
| `int8_tflops`           | INT8 算力 |
| `int4_tflops`           | INT4 算力 / W4A16 算力 |

### 3.5 高可用参数

| 参数                       | 含义                                               |
| ------------------------ | ------------------------------------------------ |
| `ha_mode`                | none / n_plus_1 / active_standby / active_active |
| `replica_count`          | 多活副本数                                            |
| `failover_reserve_ratio` | 故障冗余比例                                           |

### 3.6 运行时参数 (Runtime Config)

这类参数**不属于**特定的 LLM 的模型结构或具体的一张显卡白皮书，这主要取决于你的推理引擎（如 vLLM/SGLang）、你选择的量化方案以及你的系统安全边际策略。虽然在当前 Python 的 `dataclasses` 里它们为了传参方便被存放在了模型或显卡的类下，但在规划时应作为独立的全局变量配置。

| 参数 | 挂载对象 | 含义 |
| :--- | :--- | :--- |
| `inference_precision`        | 模型配置 | 推理精度；权重字节数与主算力口径由此决定 |
| `kv_cache_dtype`             | 模型配置 | KV Cache 的数据存储格式 |
| `weight_overhead_ratio`      | 模型配置 | 模型权重显存膨胀的冗余比例，默认 0.15 |
| `runtime_overhead_ratio`     | 模型配置 | 运行时上下文等占模型的比例，默认 0.08 |
| `runtime_overhead_gb`        | 模型配置 | 基础运行时保底显存绝对值，单位 GB，默认 2.0 |
| `usable_vram_ratio`          | 显卡配置 | 防 OOM 的安全可用显存比例，默认 0.90 |
| `decode_efficiency`          | 显卡配置 | Decode 阶段基于硬件规格打底的理论折减系数，默认 0.40 |
| `prefill_efficiency`         | 显卡配置 | Prefill 阶段基于硬件规格打底的理论折减系数，默认 0.55 |
| `compute_efficiency`         | 显卡配置 | 极限发热/功耗墙等导致的峰值算力折减系数，默认 0.60 |
| `prefill_memory_reuse_factor`| 显卡配置 | Prefill 阶段批处理时显存带宽的虚拟读取放大倍率，默认 24.0 |

---

## 4. 体验指标到系统指标的映射

这一层的目标，是把用户能直接感知的体验指标，转换成系统容量规划可直接使用的 token/s 目标。

可按两步理解：

1. 先把体验目标拆成 prefill 阶段和 decode 阶段。
2. 再分别换算成系统总吞吐目标 `target_prefill_tps_total` 与 `target_decode_tps_total`。

其中：

* `prefill` 处理输入 prompt，对应首 token 延迟 `TTFT`
* `decode` 持续生成输出 token，对应流式生成速度 `Streaming Speed`

### 4.1 首 token 延迟

$$
TTFT = t_{first_token} - t_{request}
$$

TTFT 主要受 prefill 能力影响。

设：

* 峰值并发为 $C_{peak}$
* P95 输入长度为 $S_{in,p95}$
* prefill 时间预算为 $T_{prefill_budget}$

则：

$$
target\_prefill\_tps\_total
\approx
\frac{C_{peak} \times S_{in,p95}}{T_{prefill\_budget}}
$$

含义是：

* 在峰值时刻，系统需要同时完成约 `C_peak` 个请求的 prompt 处理
* 每个请求按保守口径使用 `S_in,p95` 作为输入 token 规模
* 这些 token 需要在 `T_prefill_budget` 时间内被处理完成

因此：

* 分子 `C_peak x S_in,p95` 表示高峰时刻需要处理的总输入 token 量
* 分母 `T_prefill_budget` 表示允许系统完成这些 prefill 工作的时间预算
* 两者相除后，得到系统需要具备的总 prefill 吞吐能力

工程上通常可将 `T_prefill_budget` 视为 TTFT 预算中主要由 prefill 消耗的那部分时间，因此它通常不应大于 `target_ttft_p95_sec`。

---

### 4.2 持续生成速度

$$
StreamingSpeed = \frac{N_{output\_tokens}}{t_{last\_token} - t_{first\_token}}
$$

设：

* 活跃生成请求数为 $C_{decode}$
* 单请求最低生成速度为 $R_{decode}$

则：

$$
target\_decode\_tps\_total = C_{decode} \times R_{decode}
$$

含义是：

* `C_decode` 不是总并发，而是同一时刻正在生成输出的活跃请求数
* `R_decode` 是单请求最低生成速度，例如 `20 token/s`

因此：

* 如果同时有 `C_decode` 个请求都要满足最低生成速度
* 系统总 decode 吞吐就至少要达到 `C_decode x R_decode`

在容量规划里，`C_decode` 往往小于 `C_peak`，因为并不是所有活跃请求都同时处于 decode 阶段；如果没有更细的业务分阶段统计，可先用一个保守比例从峰值并发估算。

---

### 4.3 端到端时延校验

$$
Latency_{e2e} = t_{last\_token} - t_{request}
$$

可近似校验为：

$$
Latency_{e2e}
\approx
TTFT + \frac{S_{out}}{StreamingSpeed}
$$

也就是说，体验目标到系统吞吐目标的转换关系可以概括为：

* `TTFT -> target_prefill_tps_total`
* `Streaming Speed -> target_decode_tps_total`

一个简单示例：

* 峰值并发 `C_peak = 200`
* P95 输入长度 `S_in,p95 = 4000`
* prefill 时间预算 `T_prefill_budget = 2s`
* 活跃生成请求数 `C_decode = 120`
* 单请求最低生成速度 `R_decode = 20 token/s`

则：

$$
target\_prefill\_tps\_total
\approx
\frac{200 \times 4000}{2}
= 400000 \ token/s
$$

$$
target\_decode\_tps\_total
= 120 \times 20
= 2400 \ token/s
$$

这两个量随后会分别进入后续的 `G_prefill` 和 `G_decode` 计算，最终与显存约束一起决定 GPU 数量。

> ⚠️ **工程实现提示**：`gpu_sizing.py` 和当前 UI 都支持显式输入 `target_prefill_tps_total` / `target_decode_tps_total`。Gradio 页面仍会基于预设画像给出默认值（例如 `concurrency * 500` 作为默认总 prefill TPS），但这两个值现在可以直接手动覆盖。与此同时，请求画像中的 Token 分布调整仍主要影响显存估算（即 $M_{sizing}$）；如果你的业务吞吐目标随画像变化，需要同步修改这两个 TPS 参数。

---

## 5. 显存计算

### 5.1 总显存公式

$$
M_{total} \approx M_{weights} + M_{cache} + M_{runtime}
$$

---

### 5.2 权重显存

$$
M_{weights,raw}^{GB} = total\_params\_b \times bytes(inference\_precision)
$$

考虑附加开销：

$$
M_{weights} = M_{weights,raw}^{GB} \times (1 + r_{weight})
$$

其中：

$$
r_{weight} = 10%\sim20%
$$

---

### 5.3 Cache 显存统一公式

$$
M_{cache} \approx num\_layers \times seq\_len \times batch \times E_{cache/token/layer}
$$

其中：

$$
seq\_len = input\_len + output\_len
$$

---

### 5.4 不同架构的 Cache 计算

#### MHA

$$
E_{cache/token/layer}^{MHA} \approx 2 \times num\_heads \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{MHA}
\approx
\frac{num\_layers \times seq\_len \times batch \times 2 \times num\_heads \times head\_dim \times cache\_dtype\_bytes}{10^9}
$$

若：

$$
hidden\_size = num\_heads \times head\_dim
$$

则可简化为：

$$
M_{cache}^{MHA}
\approx
\frac{2 \times num\_layers \times seq\_len \times hidden\_size \times batch \times cache\_dtype\_bytes}{10^9}
$$

#### GQA

$$
E_{cache/token/layer}^{GQA} \approx 2 \times num\_kv\_heads \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{GQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times num\_kv\_heads \times head\_dim \times batch \times cache\_dtype\_bytes}{10^9}
$$

#### MQA

$$
E_{cache/token/layer}^{MQA} \approx 2 \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{MQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times head\_dim \times batch \times cache\_dtype\_bytes}{10^9}
$$

#### MLA

$$
E_{cache/token/layer}^{MLA}
\approx
latent\_cache\_dim \times cache\_dtype\_bytes + E_{aux}
$$

$$
M_{cache}^{MLA}
\approx
\frac{num\_layers \times seq\_len \times batch \times (latent\_cache\_dim \times cache\_dtype\_bytes + E_{aux})}{10^9}
$$

#### Sparse / Hybrid Attention

$$
M_{cache}^{Sparse} = \text{按真实缓存结构计算}
$$

---

### 5.5 运行时开销

$$
M_{runtime} = M_{workspace} + M_{system}
$$

可采用：

#### 固定值法

$$
M_{runtime} = 2 \sim 8\ \text{GB}
$$

#### 比例法

$$
M_{runtime} = r_{runtime} \times M_{weights}
$$

其中：

$$
r_{runtime} = 5%\sim20%
$$

稳妥写法：

$$
M_{runtime} = \max(M_{runtime,fixed},\ r_{runtime}\times M_{weights})
$$

---

### 5.6 平均显存与 P95 显存

#### 平均场景

$$
M_{total}^{avg} = M_{weights} + concurrency \times M_{cache}(E[S]) + M_{runtime}
$$

#### P95 场景

$$
M_{total}^{p95} = M_{weights} + concurrency \times M_{cache}(S_{p95}) + M_{runtime}
$$

采购 sizing 建议采用：

$$
M_{sizing} = M_{total}^{p95}
$$

---

### 5.7 单卡可用显存与显存约束卡数

$$
M_{usable}^{GB} = vram\_gb \times r_{usable}
$$

其中：

$$
r_{usable} = 0.85\sim0.92
$$

若需统一：

$$
r_{usable} = 0.90
$$

显存约束卡数：

$$
G_{memory} = \left\lceil \frac{M_{sizing}}{M_{usable}} \right\rceil
$$

---

## 6. 请求长度分布

设共有 (K) 类请求，第 (i) 类请求占比为 (p_i)，满足：

$$
\sum_{i=1}^{K} p_i = 1
$$

总长度：

$$
S_i = S_{in,i} + S_{out,i}
$$

### 平均输入长度

$$
E[S_{in}] = \sum_{i=1}^{K} p_i \cdot S_{in,i}
$$

### 平均输出长度

$$
E[S_{out}] = \sum_{i=1}^{K} p_i \cdot S_{out,i}
$$

### 平均总长度

$$
E[S] = \sum_{i=1}^{K} p_i \cdot S_i
$$

### P95 总长度

将请求按 (S_i) 从小到大排序，累计占比达到 95% 时对应长度记为：

$$
S_{p95}
$$

---

## 7. 吞吐计算

当前版本统一采用 **GPU 厂商规格推导值**，不使用实测吞吐。

### 7.1 Decode 吞吐

$$
TPS_{decode}^{spec} = \min(TPS_{decode,compute},\ TPS_{decode,memory}) \times \eta_{decode}
$$

简化为带宽主导时：

$$
TPS_{decode}^{spec}
\approx
\frac{memory\_bandwidth\_gb\_per\_sec \times 10^9 \times \eta_{decode}}{B_{decode/token}}
$$

其中：

* $B_{decode/token}$：每生成 1 个 token 的主要字节访问量
* $\eta_{decode}$：折减系数

建议：

* 乐观：0.50–0.70
* 中性：0.35–0.50
* 保守：0.25–0.40

若需要算力上界：

$$
TPS_{decode,compute} = \frac{peak_compute \times \eta_{compute}}{F_{decode/token}}
$$

其中 `peak_compute` 按主计算精度选择：

* FP32：`fp32_tflops`
* FP16：`fp16_tflops`
* BF16：`bf16_tflops`
* FP8：`fp8_tflops`
* INT8：`int8_tflops`
* INT4：若未显式提供 `int4_tflops`，可降级取 `int8_tflops`（当前代码约定的算力回退机制）。

---

### 7.2 Prefill 吞吐

$$
TPS_{prefill}^{spec} = \min(TPS_{prefill,compute},\ TPS_{prefill,memory}) \times \eta_{prefill}
$$

#### 算力上界

$$
TPS_{prefill,compute} = \frac{peak\_compute \times \eta_{compute}}{F_{prefill/token}}
$$

#### 带宽上界

$$
TPS_{prefill,memory} = \frac{memory\_bandwidth\_gb\_per\_sec \times 10^9 \times \eta_{bw}}{B_{prefill/token}}
$$

其中 `peak_compute` 同样按主精度选择：

* FP32：`fp32_tflops`
* FP16：`fp16_tflops`
* BF16：`bf16_tflops`
* FP8：`fp8_tflops`
* INT8：`int8_tflops`

建议折减系数：

* 乐观：0.65–0.80
* 中性：0.45–0.65
* 保守：0.30–0.45

> 💡 **访存带宽放大效应 (`prefill_memory_reuse_factor`)：**
> 在 Prefill 阶段批处理处理大量 Prompt 个 Token 时（计算密集型场景），权重加载后能在多个 Token 之间充分复用（相比 Decode 只有一个 Token）。这会使得其显存带宽等效值急剧放大，计算受限而非访存受限成为常态。在计算引擎中，通常会在 $TPS_{decode,memory}$ 的基础上乘以一个极高的放大复用因子（如默认值 `24.0`）来推断 $TPS_{prefill,memory}$，突破显存带宽的物理限制。

---

### 7.3 Dense 与 MoE 的区别

#### Dense

$$
TPOT_{compute}^{Dense} \propto total\_params
$$

#### MoE

$$
TPOT_{compute}^{MoE} \propto activated\_params\_per\_token
$$

因此：

* 显存估算看总参数
* 吞吐估算看激活参数量

---

### 7.4 吞吐约束卡数

#### Decode 约束

$$
G_{decode} = \left\lceil \frac{target\_decode\_tps\_total}{TPS_{decode}^{spec}} \right\rceil
$$

#### Prefill 约束

$$
G_{prefill} = \left\lceil \frac{target\_prefill\_tps\_total}{TPS_{prefill}^{spec}} \right\rceil
$$

#### 业务最少卡数

$$
G_{biz} = \max(G_{memory},\ G_{decode},\ G_{prefill})
$$

---

## 8. 高可用计算

### 无高可用

$$
G_{final} = G_{biz}
$$

### N+1

$$
G_{final} = G_{biz} + 1
$$

### 主备

$$
G_{final} = 2 \times G_{biz}
$$

### 多活

$$
G_{final} = replica\_count \times G_{biz}
$$

### 故障冗余

$$
G_{final}' = \left\lceil G_{final} \times (1 + failover\_reserve\_ratio) \right\rceil
$$

---

## 9. 最终输出

资源计算最终输出为：

* `G_memory`
* `G_prefill`
* `G_decode`
* `G_biz`
* `G_final`

其中：

$$
G_{biz} = \max(G_{memory}, G_{prefill}, G_{decode})
$$

$$
G_{final} = G_{biz} + G_{HA}
$$
