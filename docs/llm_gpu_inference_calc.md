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
| `total_params`               | 总参数量                           |
| `activated_params_per_token` | 每 token 激活参数量，MoE 使用           |
| `num_layers`                 | 层数                             |
| `hidden_size`                | 隐藏维度                           |
| `num_heads`                  | Query heads 数                  |
| `num_kv_heads`               | KV heads 数                     |
| `head_dim`                   | 每个 head 维度                     |
| `latent_cache_dim`           | MLA latent cache 维度            |
| `weight_dtype_bytes`         | 权重精度字节数                        |
| `cache_dtype_bytes`          | cache 精度字节数                    |
| `activation_dtype_bytes`     | 激活精度字节数                        |

### 3.2 请求参数

| 参数                         | 含义                   |
| -------------------------- | -------------------- |
| `concurrency`              | 同时活跃请求数              |
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
| `vram_gb`               | 显存容量    |
| `memory_bandwidth_gbps` | 显存带宽    |
| `fp32_tflops`           | FP32 算力 |
| `fp16_tflops`           | FP16 算力 |
| `bf16_tflops`           | BF16 算力 |
| `fp8_tflops`            | FP8 算力  |
| `int8_tflops`           | INT8 算力 |

### 3.5 高可用参数

| 参数                       | 含义                                               |
| ------------------------ | ------------------------------------------------ |
| `ha_mode`                | none / n_plus_1 / active_standby / active_active |
| `replica_count`          | 多活副本数                                            |
| `failover_reserve_ratio` | 故障冗余比例                                           |
| `zone_redundancy`        | 是否双机房 / 双可用区                                     |

---

## 4. 体验指标到系统指标的映射

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

---

## 5. 显存计算

### 5.1 总显存公式

$$
M_{total} \approx M_{weights} + M_{cache} + M_{runtime}
$$

---

### 5.2 权重显存

$$
M_{weights,raw}^{GiB} = \frac{total\_params \times weight\_dtype\_bytes}{1024^3}
$$

考虑附加开销：

$$
M_{weights} = M_{weights,raw}^{GiB} \times (1 + r_{weight})
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
\frac{num\_layers \times seq\_len \times batch \times 2 \times num\_heads \times head\_dim \times cache\_dtype\_bytes}{1024^3}
$$

若：

$$
hidden\_size = num\_heads \times head\_dim
$$

则可简化为：

$$
M_{cache}^{MHA}
\approx
\frac{2 \times num\_layers \times seq\_len \times hidden\_size \times batch \times cache\_dtype\_bytes}{1024^3}
$$

#### GQA

$$
E_{cache/token/layer}^{GQA} \approx 2 \times num\_kv\_heads \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{GQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times num\_kv\_heads \times head\_dim \times batch \times cache\_dtype\_bytes}{1024^3}
$$

#### MQA

$$
E_{cache/token/layer}^{MQA} \approx 2 \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{MQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times head\_dim \times batch \times cache\_dtype\_bytes}{1024^3}
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
\frac{num\_layers \times seq\_len \times batch \times (latent\_cache\_dim \times cache\_dtype\_bytes + E_{aux})}{1024^3}
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
M_{runtime} = 2 \sim 8\ \text{GiB}
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
M_{usable} = vram_gb \times r_{usable}
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
\frac{memory\_bandwidth\_gbps \times 10^9 \times \eta_{decode}}{B_{decode/token}}
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
TPS_{prefill,memory} = \frac{memory\_bandwidth\_gbps \times 10^9 \times \eta_{bw}}{B_{prefill/token}}
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

### 双机房 / 双可用区

$$
G_{final}^{zone} = 2 \times G_{final}'
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

---

## 10. 一句话总结

整条链路统一为：

$$
\text{体验目标} \rightarrow \text{系统吞吐目标} \rightarrow \text{显存/吞吐计算} \rightarrow \text{GPU数量输出}
$$

其中：

* **TTFT 对应 prefill**
* **流畅度对应 decode**
* **prefill / decode token/s 是输入约束**
* **GPU 数量与配置是最终输出**
