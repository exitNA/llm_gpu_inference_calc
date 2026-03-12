# 大模型推理 GPU 资源计算原理

## 1. 目标

本文档用于说明大模型在线推理场景下的 GPU 资源计算原理，支持：

* 单模型推理资源估算
* 不同架构模型的统一估算
* 基于 GPU 厂商规格参数的前期容量规划
* 显存约束、吞吐约束和高可用约束的综合计算

本文档适用于前期方案设计、采购估算和部署 sizing。
不以压测结果为前提，但保留后续校准空间。

---

## 2. 统一计算框架

推理资源计算统一分为两层：

### 2.1 显存层

$$
M_{total} \approx M_{weights} + M_{cache} + M_{runtime}
$$

其中：

* (M_{weights})：模型权重显存
* (M_{cache})：推理缓存显存，通常为 KV Cache 或其变体
* (M_{runtime})：运行时开销，包括激活值、临时张量、框架缓存池、碎片化、通信缓冲等

### 2.2 吞吐层

$$
TPS \approx \min(TPS_{compute},\ TPS_{memory},\ TPS_{scheduler})
$$

其中：

* (TPS_{compute})：算力约束上限
* (TPS_{memory})：显存带宽约束上限
* (TPS_{scheduler})：调度、并行、路由等系统约束上限

在当前版本中，由于 GPU 侧不使用实测吞吐，吞吐估算统一采用：

* **decode：以显存带宽主导估算**
* **prefill：以算力上界与带宽上界取最小值估算**

---

## 3. 输入参数

## 3.1 模型参数

| 参数                           | 含义                             |
| ---------------------------- | ------------------------------ |
| `arch_family`                | Dense / MoE                    |
| `attention_type`             | MHA / GQA / MQA / MLA / Sparse |
| `total_params`               | 总参数量                           |
| `activated_params_per_token` | 每 token 激活参数量，MoE 必填           |
| `num_layers`                 | 层数                             |
| `hidden_size`                | 隐藏维度                           |
| `num_heads`                  | Query heads 数                  |
| `num_kv_heads`               | KV heads 数，GQA/MQA 用           |
| `head_dim`                   | 每个 attention head 维度           |
| `latent_cache_dim`           | MLA 的 latent cache 维度          |
| `weight_dtype_bytes`         | 权重精度字节数                        |
| `cache_dtype_bytes`          | cache 精度字节数                    |
| `activation_dtype_bytes`     | 激活精度字节数                        |

---

## 3.2 请求流量参数

| 参数                         | 含义                   |
| -------------------------- | -------------------- |
| `concurrency`              | 同时活跃请求数              |
| `target_decode_tps_total`  | 总 decode token/s 目标  |
| `target_prefill_tps_total` | 总 prefill token/s 目标 |
| `request_shapes`           | 请求长度分布               |

建议 `request_shapes` 至少包含：

* 请求类型名称
* 流量占比
* 平均输入长度
* 平均输出长度

---

## 3.3 GPU 参数

GPU 输入参数统一限定为：

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

---

## 3.4 高可用参数

| 参数                       | 含义                                               |
| ------------------------ | ------------------------------------------------ |
| `ha_mode`                | none / n_plus_1 / active_standby / active_active |
| `replica_count`          | 多活副本数                                            |
| `failover_reserve_ratio` | 故障冗余比例                                           |
| `zone_redundancy`        | 是否双机房 / 双可用区部署                                   |

---

## 4. 显存计算原理

## 4.1 权重显存

基础公式：

$$
M_{weights,raw}^{GiB} = \frac{total_params \times weight_dtype_bytes}{1024^3}
$$

若考虑量化元数据、附加模块、索引等额外开销：

$$
M_{weights} = M_{weights,raw}^{GiB} \times (1 + r_{weight})
$$

其中：

* (r_{weight}) 建议取 **10%–20%**

更通用写法：

$$
M_{weights} \approx \sum_i P_i \times b_i + M_{extra}
$$

其中：

* (P_i)：第 (i) 类参数量
* (b_i)：对应字节数
* (M_{extra})：附加模块开销

---

## 4.2 Cache 显存统一公式

为兼容不同架构，cache 统一写成：

$$
M_{cache} \approx num\_layers \times seq\_len \times batch \times E_{cache/token/layer}
$$

其中：

* `seq_len = input_len + output_len`
* (E_{cache/token/layer})：每层每 token 的真实缓存元素数 × 字节数

---

## 4.3 不同架构的 Cache 公式

### 4.3.1 标准 MHA

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

可简化为：

$$
M_{cache}^{MHA}
\approx
\frac{2 \times num\_layers \times seq\_len \times hidden\_size \times batch \times cache\_dtype\_bytes}{1024^3}
$$

---

### 4.3.2 GQA

$$
E_{cache/token/layer}^{GQA} \approx 2 \times num\_kv\_heads \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{GQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times num\_kv\_heads \times head\_dim \times batch \times cache\_dtype\_bytes}{1024^3}
$$

---

### 4.3.3 MQA

$$
E_{cache/token/layer}^{MQA} \approx 2 \times head\_dim \times cache\_dtype\_bytes
$$

$$
M_{cache}^{MQA}
\approx
\frac{2 \times num\_layers \times seq\_len \times head\_dim \times batch \times cache\_dtype\_bytes}{1024^3}
$$

---

### 4.3.4 MLA

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

其中：

* `latent_cache_dim` 为 latent cache 维度
* (E_{aux}) 为附加状态开销

---

### 4.3.5 Sparse / Hybrid Attention

稀疏注意力优先影响计算复杂度，不一定同比例降低缓存显存，因此：

$$
M_{cache}^{Sparse} = \text{按真实缓存结构计算}
$$

不建议简单按注意力稀疏比例折算显存。

---

## 4.4 运行时开销

$$
M_{runtime} = M_{workspace} + M_{system}
$$

其中：

* (M_{workspace})：激活值、临时张量、workspace
* (M_{system})：框架缓存池、显存碎片、通信缓冲、runtime 常驻开销

工程上建议采用：

### 固定值法

$$
M_{runtime} = 2 \sim 8 \text{ GiB}
$$

### 比例法

$$
M_{runtime} = r_{runtime} \times M_{weights}
$$

其中：

* (r_{runtime}) 建议取 **5%–20%**

更稳妥的写法：

$$
M_{runtime} = \max(M_{runtime,fixed},\ r_{runtime}\times M_{weights})
$$

---

## 4.5 平均显存与 P95 显存

### 平均场景

$$
M_{total}^{avg} = M_{weights} + concurrency \times M_{cache}(E[S]) + M_{runtime}
$$

### P95 场景

设 P95 总长度为 (S_{p95})，则：

$$
M_{total}^{p95} = M_{weights} + concurrency \times M_{cache}(S_{p95}) + M_{runtime}
$$

采购 sizing 建议使用：

$$
M_{sizing} = M_{total}^{p95}
$$

---

## 4.6 单卡可用显存与显存约束卡数

$$
M_{usable} = vram\_gb \times r_{usable}
$$

其中：

* (r_{usable}) 建议取 **0.85–0.92**
* 若需统一，可取 **0.90**

显存约束下所需 GPU 数量：

$$
G_{memory} = \left\lceil \frac{M_{sizing}}{M_{usable}} \right\rceil
$$

---

## 5. 请求长度分布计算

设共有 (K) 类请求，第 (i) 类请求占比为 (p_i)，满足：

$$
\sum_{i=1}^{K} p_i = 1
$$

总长度为：

$$
S_i = S_{in,i} + S_{out,i}
$$

则：

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

将各请求按 (S_i) 从小到大排序，按占比累计，首次达到或超过 95% 时对应的长度记为：

$$
S_{p95}
$$

---

## 6. 吞吐计算原理

当前版本不使用 GPU 实测吞吐，而统一使用 **GPU 厂商规格参数推导值**。

---

## 6.1 Decode 吞吐估算

decode 阶段通常更偏向显存带宽受限，因此建议写成：

$$
TPS_{decode}^{spec} = \min(TPS_{decode,compute},\ TPS_{decode,memory}) \times \eta_{decode}
$$

在简化口径下，优先采用带宽主导近似：

$$
TPS_{decode}^{spec}
\approx
\frac{memory\_bandwidth\_gbps \times 10^9 \times \eta_{decode}}{B_{decode/token}}
$$

其中：

* (B_{decode/token})：每生成 1 个 token 的主要字节访问量
* (\eta_{decode})：decode 折减系数

建议折减系数：

* 乐观：0.50–0.70
* 中性：0.35–0.50
* 保守：0.25–0.40

若需显式引入算力上界，则：

$$
TPS_{decode,compute} = \frac{peak\_compute \times \eta_{compute}}{F_{decode/token}}
$$

其中 `peak_compute` 按模型主计算精度选择：

* FP32：`fp32_tflops`
* FP16：`fp16_tflops`
* BF16：`bf16_tflops`
* FP8：`fp8_tflops`
* INT8：`int8_tflops`

---

## 6.2 Prefill 吞吐估算

prefill 更接近整段输入的一次前向批量计算，因此建议采用：

$$
TPS_{prefill}^{spec} = \min(TPS_{prefill,compute},\ TPS_{prefill,memory}) \times \eta_{prefill}
$$

### 算力上界

$$
TPS_{prefill,compute} = \frac{peak\_compute \times \eta_{compute}}{F_{prefill/token}}
$$

其中 `peak_compute` 按实际主计算精度选择：

* FP32：`fp32_tflops`
* FP16：`fp16_tflops`
* BF16：`bf16_tflops`
* FP8：`fp8_tflops`
* INT8：`int8_tflops`

### 带宽上界

$$
TPS_{prefill,memory} = \frac{memory\_bandwidth\_gbps \times 10^9 \times \eta_{bw}}{B_{prefill/token}}
$$

### Prefill 折减系数建议

* 乐观：0.65–0.80
* 中性：0.45–0.65
* 保守：0.30–0.45

一般来说，prefill 的折减可略高于 decode。

---

## 6.3 Dense 与 MoE 在吞吐上的区别

### Dense

$$
TPOT_{compute}^{Dense} \propto total\_params
$$

### MoE

$$
TPOT_{compute}^{MoE} \propto activated\_params\_per\_token
$$

因此：

* **显存估算**：看总参数 / 总 checkpoint
* **吞吐估算**：看激活参数量

两者不能混用。

---

## 6.4 吞吐约束卡数

### Decode 约束

$$
G_{decode} = \left\lceil \frac{target\_decode\_tps\_total}{TPS_{decode}^{spec}} \right\rceil
$$

### Prefill 约束

$$
G_{prefill} = \left\lceil \frac{target\_prefill\_tps\_total}{TPS_{prefill}^{spec}} \right\rceil
$$

### 业务最少卡数

$$
G_{biz} = \max(G_{memory},\ G_{decode},\ G_{prefill})
$$

如果未定义 prefill 指标，可忽略该项。

---

## 7. 高可用计算

默认按单模型高可用计算，不涉及多模型混部。

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

若副本数为 `replica_count`：

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

## 8. 统一计算流程

### 第 1 步：识别模型架构

确定：

* Dense / MoE
* MHA / GQA / MQA / MLA / Sparse
* 权重精度、cache 精度、激活精度

### 第 2 步：建立请求长度分布

得到：

* 平均输入长度
* 平均输出长度
* P95 总长度

### 第 3 步：计算权重显存

$$
M_{weights}
$$

### 第 4 步：按架构计算 cache 显存

$$
M_{cache}
$$

### 第 5 步：加入运行时开销

得到：

$$
M_{total}^{avg},\quad M_{total}^{p95}
$$

### 第 6 步：计算显存约束卡数

$$
G_{memory}
$$

### 第 7 步：根据 GPU 厂商规格推导 decode / prefill 吞吐

得到：

$$
TPS_{decode}^{spec},\quad TPS_{prefill}^{spec}
$$

### 第 8 步：计算吞吐约束卡数

$$
G_{decode},\quad G_{prefill}
$$

### 第 9 步：得到业务最少卡数

$$
G_{biz} = \max(G_{memory}, G_{decode}, G_{prefill})
$$

### 第 10 步：叠加高可用

得到：

$$
G_{final}
$$

---

## 9. 核心结论

本方法的核心不是为某一个模型写固定公式，而是建立统一框架：

### 显存统一公式

$$
M_{total} \approx M_{weights} + M_{cache} + M_{runtime}
$$

### Cache 统一公式

$$
M_{cache} \approx num\_layers \times seq\_len \times batch \times E_{cache/token/layer}
$$

### 吞吐统一公式

$$
TPS \approx \min(TPS_{compute},\ TPS_{memory}) \times \eta
$$

### 业务最少卡数

$$
G_{biz} = \max(G_{memory},\ G_{decode},\ G_{prefill})
$$

### 最终卡数

$$
G_{final} = G_{biz} + G_{HA}
$$

其中需要根据架构替换的核心量只有：

* `E_cache/token/layer`
* `activated_params_per_token`
* `peak_compute` 的选择
* `runtime` 的预留方式

---

## 10. 适用性说明

这套计算原理适合：

* 前期方案设计
* GPU 卡型筛选
* 采购数量初估
* 不同架构模型统一估算
* 基于 GPU 厂商规格参数进行前期估算
