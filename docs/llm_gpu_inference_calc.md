# 大模型在线推理 GPU 卡数测算与能力回推方法

---

## 1. 文档目标与适用边界

本文给出一套面向**采购前容量规划**的大模型在线推理 GPU 卡数测算方法。本文首先回答一个核心问题：

> **在给定业务目标、模型参数和 GPU 参数的条件下，最少需要多少张 GPU 卡？**

记最终所需的最少 GPU 卡数为：

$$
G_{req}
$$

在此基础上，本文进一步回答第二个配套问题：

> **当 GPU 卡数确定后，这批卡大致能够支撑到什么业务能力水平？**

后文将对总吞吐、可持续 QPS、日 token 总量、最大在途请求量和时延风险做近似回推。

本文只讨论“需要多少卡”与“这些卡的大致整体能力”，不讨论以下问题：

- 实例如何划分；
- 模型副本如何组织；
- TP / EP / PP 等并行策略如何选择；
- 具体部署映射、机器拓扑与跨机通信；
- 调度器、连续批处理和排队分布的严格建模。

因此，本文输出的是**采购前的合理估算值**，用于预算、选型、横向比较和容量规划，而不是生产环境中的严格 SLA 承诺值。

需要特别说明的是：

- **显存约束**和**吞吐约束**可以直接折算为总卡数下界；
- **时延约束**在不预设并行组织方式的前提下，不能稳定地直接折算为总卡数下界，因此本文将其改写为**单卡服务能力的必要条件**；
- 若时延必要条件本身都不满足，则仅靠增加总卡数而不改变单请求并行方式，通常无法保证对应时延目标；
- 已知卡数后的能力回推仍属于**总量近似能力**，适合做预算和容量说明，不应理解为对线上稳定能力或 SLA 的严格承诺。

---

## 2. 总体方法框架

全文围绕三类约束展开：

1. **显存约束**：模型与峰值活跃请求是否装得下；
2. **吞吐约束**：峰值流量下系统每秒是否处理得完输入 token 与输出 token；
3. **时延必要条件**：单卡服务能力是否至少能支撑目标 TTFT / E2E 预算。

其中：

- 显存约束给出总显存角度的卡数下界 $G_{mem}$；
- Prefill 吞吐约束给出输入工作量角度的卡数下界 $G_{pre}$；
- Decode 吞吐约束给出输出工作量角度的卡数下界 $G_{dec}$；
- 时延部分不再直接构造总卡数公式，而是提供**必要条件检查**。

因此，本文的主求解公式统一为：

$$
G_{req} = \max(G_{mem}, G_{pre}, G_{dec})
$$

并在得到 $G_{req}$ 后，额外检查：

- Prefill 单卡服务能力是否满足 $TTFT_{p95}^{target}$；
- Decode 单卡服务能力是否满足 $T_{dec,p95}$ 预算。

如果时延必要条件不满足，则说明：

- 当前 GPU 型号的单卡能力不足，或
- 后续必须依赖模型压缩、更强 GPU、或能提升单请求速度的并行组织方式。

此时，不能简单地把“增加总卡数”理解成一定能解决时延问题。

在完成最少卡数测算后，还可以基于同一套总量模型对整体能力做回推，包括：

- 总 Prefill / Decode 吞吐能力；
- 可持续 QPS；
- 日 token 总量；
- 最大在途请求量；
- 时延目标是否存在明显风险。

### 2.1 计算流程图

下图给出当前实现的整体计算流程。它把 `QPS` 建模、峰值在途建模、吞吐下界、显存下界和最终卡数汇总放在同一张图里，便于快速定位每个输入参数会影响哪一类约束。

```mermaid
flowchart TD
    A[输入业务目标 模型参数 GPU参数] --> B[请求长度画像]
    A --> C{QPS 建模方式}
    A --> D{峰值在途建模方式}

    B --> B1[P95 输入长度 S_in,p95]
    B --> B2[P95 输出长度 S_out,p95]
    B --> B3[P95 总长度 S_p95]

    C -->|直接输入峰值 QPS| C1[读取 lambda_peak]
    C -->|Poisson 日均反推| C2[Req_day -> lambda_avg]
    C2 --> C3[lambda_base = lambda_avg x gamma_burst]
    C3 --> C4[Q_q Poisson(lambda_base x delta_t) / delta_t]
    C1 --> E[解析后的峰值 QPS lambda_peak]
    C4 --> E

    D -->|Little 定律近似| D1[C_peak = lambda_peak x E2E_p95 x rho_safe]
    D -->|直接输入峰值在途| D2[C_peak = C_peak,input]

    E --> F[峰值 Prefill 工作量]
    E --> G[峰值 Decode 工作量]
    E --> D1
    B1 --> F
    B2 --> G
    D1 --> H[峰值在途预算 C_peak]
    D2 --> H

    F --> F1[TPS_pre,target = lambda_peak x S_in,p95]
    G --> G1[TPS_dec,target = lambda_peak x S_out,p95]

    F1 --> I[单卡 Prefill 吞吐 TPS_pre,card]
    G1 --> J[单卡 Decode 吞吐 TPS_dec,card]
    I --> I1[G_pre = ceil TPS_pre,target / TPS_pre,card]
    J --> J1[G_dec = ceil TPS_dec,target / TPS_dec,card]

    B3 --> K[单请求 Cache 显存]
    H --> L[总 Cache 显存]
    K --> L
    A --> M[权重显存 Mw 与运行时固定显存 Mr]
    A --> N[单卡有效显存 V_gpu,eff]
    L --> O[G_mem = ceil Mw + Mr + M_cache / V_gpu,eff]
    M --> O
    N --> O

    I --> P[Prefill 时延必要条件]
    J --> Q[Decode 时延必要条件]

    I1 --> R[G_req = max G_mem G_pre G_dec]
    J1 --> R
    O --> R

    R --> S[能力回推]
    S --> S1[总 Prefill/Decode 吞吐]
    S --> S2[保守可持续 QPS]
    S --> S3[日 token 总量]
    S --> S4[显存最大在途请求量]
    P --> T[时延风险判断]
    Q --> T
    R --> T
```

从这张图可以看到：

- `QPS` 建模方式同时影响 Prefill 和 Decode 两条吞吐链路；
- 峰值在途建模方式主要影响显存链路中的 `KV cache` 预算；
- 最终卡数始终取 `G_mem`、`G_pre`、`G_dec` 三者最大值；
- 时延目标不直接折算为总卡数，而是作为单卡能力层面的必要条件检查。

---

## 3. 输入参数与统计口径

### 3.1 业务目标

业务侧通常至少提供以下输入：

- 峰值请求到达率 $\lambda_{peak}$；
- P95 首字时延目标 $TTFT_{p95}^{target}$；
- P95 单次总时延目标 $E2E_{p95}^{target}$。

其中，$\lambda$ 的单位为 req/s，表示模型调用层面的请求到达率。若业务侧提供的是用户请求 QPS，而单个用户请求会触发多次模型调用，则应先折算为等效模型调用 QPS，再进入后续测算。

当前实现支持两种 QPS 输入方式：

1. **直接输入峰值 QPS**
   业务侧直接提供 sizing 使用的峰值请求率 $\lambda_{peak}$。

2. **Poisson 日均反推峰值 QPS**
   当业务侧只有日均调用量 $Req_{day}$ 时，先折算平均 QPS：

   $$
   \lambda_{avg} = \frac{Req_{day}}{86400}
   $$

   若还需要体现日内高峰，可额外引入高峰放大系数 $\gamma_{burst} \ge 1$：

   $$
   \lambda_{base} = \lambda_{avg} \gamma_{burst}
   $$

   再选定统计时间窗 $\Delta t$ 和分位数 $q$，把该时间窗内的到达请求数近似为：

   $$
   N_{\Delta t} \sim Poisson(\lambda_{base}\Delta t)
   $$

   则 sizing 采用的峰值 QPS 定义为：

   $$
   \lambda_{peak} = \frac{Q_q\left(Poisson(\lambda_{base}\Delta t)\right)}{\Delta t}
   $$

   该模式适合“只有日调用量、缺少秒级峰值监控”的场景。它本质上是采购前的保守近似，而不是对真实突发流量分布的严格拟合。

### 3.2 请求长度画像

至少需要以下统计量：

- P95 输入长度 $S_{in,p95}$；
- P95 输出长度 $S_{out,p95}$。

进一步定义：
$$
S_{p95} = S_{in,p95} + S_{out,p95}
$$

如果业务侧直接提供了真实的 P95 总长度，则优先采用真实值替代上式。

### 3.3 模型信息

模型侧至少需要：

- 总参数量 $P_{total}$；
- 每个 token 的激活参数量 $P_{act}$；
- 层数 $L$；
- attention 结构相关信息，例如 KV 头数、head 维度、MLA latent 维度；
- 权重精度与 KV cache 精度。

对于 Dense 模型，通常有：

$$
P_{act} = P_{total}
$$

对于 MoE 模型，$P_{act}$ 应理解为**单个 token 实际参与前向计算的总参数量**，应包含共享层、被激活 expert 的参数总和以及必要的路由开销近似。

### 3.4 GPU 信息与效率系数

GPU 侧至少需要：

- 单卡显存容量 $V_{gpu}$；
- 单卡峰值显存带宽 $B_{mem}$；
- 单卡在目标精度下的峰值算力 $F_{peak}$。

单位口径说明：

- GPU 规格输入中的显存容量沿用厂商标称值，字段仍记为 `GB`，按二进制容量解释，即 `1 GB(vram label) = 1024^3 bytes`；
- 带宽规格仍沿用厂商标称值，按十进制 `GB/s` 记录；
- 所有由程序从字节数换算出来的显存结果，在展示时统一使用二进制单位 `GiB / MiB / KiB`，即按 $1024$ 进位。

此外还需要少量效率系数：

- 可用显存比例 $\eta_{vram}$；
- 带宽利用率 $\eta_{bw}$；
- 算力利用率 $\eta_{cmp}$。

它们满足：

$$
0 < \eta_{vram}, \eta_{bw}, \eta_{cmp} \le 1
$$

### 3.5 显存附加系数

显存估算还需要两个附加系数：

- 权重附加显存系数 $\alpha_w$；
- 运行时固定显存系数 $\alpha_r$。

它们用于吸收框架实现、临时缓冲、碎片化和运行时常驻开销等影响。

### 3.6 当前实现的口径选择

当前实现已支持两组可切换口径：

- **QPS 建模方式**
  - 直接输入峰值请求率 $\lambda_{peak}$；
  - 或由日调用量通过 Poisson 分位数反推得到 $\lambda_{peak}$。

- **峰值在途建模方式**
  - 使用 Little 定律式近似：

    $$
    C_{peak}^{budget} = \lambda_{peak} E2E_{p95}^{target} \cdot \rho_{safe}
    $$

  - 或直接输入峰值在途请求量 $C_{peak}^{input}$，并令：

    $$
    C_{peak}^{budget} = C_{peak}^{input}
    $$

- **长度与时延口径**：**P95 输入 / 输出长度** $S_{in,p95}, S_{out,p95}$，以及 **P95 时延目标** $TTFT_{p95}^{target}, E2E_{p95}^{target}$。

因此，当前代码中的吞吐约束始终以**解析后的峰值 QPS + P95 长度**为准；显存约束则以**解析后的峰值在途预算 + P95 总长度**为准。

默认兼容模式仍为：

- 直接输入峰值 QPS；
- Little 定律近似峰值在途；
- P95 输入 / 输出长度与 P95 时延目标。

### 3.7 合法性校验

进入正式计算前，至少检查：

- 解析后的 $\lambda_{peak} > 0$；
- 若采用 Poisson QPS 模式，则 $Req_{day} > 0$、$\gamma_{burst} \ge 1$、$\Delta t > 0$、$0 < q < 1$；
- $TTFT_{p95}^{target} > 0$；
- $E2E_{p95}^{target} > TTFT_{p95}^{target}$；
- 若采用直接峰值在途模式，则 $C_{peak}^{input} > 0$；
- 若采用 Little 定律近似，则安全系数 $\rho_{safe} \ge 1$；
- 所有效率系数位于 $(0,1]$ 内。

$$
T_{dec,p95} = E2E_{p95}^{target} - TTFT_{p95}^{target}
$$

---

## 4. 由业务目标推导系统需求

### 4.1 两阶段近似

在线推理请求通常分成两个阶段：

- **Prefill**：处理整段输入，决定首字何时出现；
- **Decode**：逐 token 生成输出，决定后续生成时长与总时延。

本文采用以下近似：

> **将 TTFT 近似视为 Prefill 服务时间。**

于是：

- Prefill 主要对应输入 token 工作量与首字时延目标；
- Decode 主要对应输出 token 工作量与尾部生成时间预算。

### 4.2 峰值 token 工作量

当前实现中，系统在单位时间内需要处理的主约束 token 工作量为：

$$
TPS_{pre,target}^{peak} = \lambda_{peak} S_{in,p95}
$$

$$
TPS_{dec,target}^{peak} = \lambda_{peak} S_{out,p95}
$$

其中：

- $TPS_{pre,target}^{peak}$ 表示峰值流量下系统每秒至少要处理的输入 token 数；
- $TPS_{dec,target}^{peak}$ 表示峰值流量下系统每秒至少要生成的输出 token 数。

它们本质上是**系统总吞吐需求**，并不直接等价于单请求时延目标。

### 4.3 峰值活跃请求预算

当前实现支持两种峰值活跃请求预算输入方式。

#### 4.3.1 Little 定律近似

当只有峰值 QPS 与时延预算时，可采用 Little 定律式近似：

$$
C_{peak}^{budget} = \lambda_{peak} E2E_{p95}^{target} \cdot \rho_{safe}
$$

其中，$C_{peak}^{budget}$ 是显存估算中的核心保守量，因为 KV cache 与活跃序列数直接相关。

该预算是采购前粗粒度估算，不是严格的排队分布结论。若系统存在明显排队、长尾混跑或突发流量尖峰，应再乘上额外安全系数。它也更适合被理解为“采购前保守在途预算上界”，而非线上真实瞬时并发分布的精确预测。

#### 4.3.2 直接输入峰值在途请求量

若业务侧已经有更直接的监控或仿真结果，例如：

- 峰时实际在途请求量分位数；
- 网关 / 调度器统计得到的最大活跃会话数；
- 根据历史 trace 或离线回放得到的并发分布；

则可以直接输入：

$$
C_{peak}^{budget} = C_{peak}^{input}
$$

该模式更贴近显存问题本身，因为显存主要由“同时挂着多少条请求”决定，而不是由平均到达率本身决定。

---

## 5. 显存约束下的卡数测算

### 5.1 显存构成

在线推理中，总显存需求通常拆成三部分：

1. **模型权重显存**；
2. **运行时固定显存**；
3. **活跃请求相关显存**，主要是 KV cache 与中间状态。

因此，总显存需求近似为：

$$
M_{total} = M_w + M_r + M_{cache}
$$

在数值实现中，$M_w$、$M_r$、$M_{cache}$ 均先按字节计算，再在展示层统一格式化为 `GiB / MiB / KiB`。

### 5.2 权重显存

模型权重显存近似为：

$$
M_w = P_{total} \cdot b_w \cdot (1 + \alpha_w)
$$

其中：

- $b_w$ 为单个权重参数占用的字节数；
- $\alpha_w$ 为权重附加显存系数，用于吸收格式转换、元数据、额外缓冲区和碎片等影响。

### 5.3 运行时固定显存

运行时固定显存近似为：

$$
M_r = \alpha_r M_w
$$

其中，$\alpha_r$ 吸收框架运行时常驻缓冲区、kernel workspace、上下文管理和额外预留空间等因素。

### 5.4 活跃请求相关显存

活跃请求相关显存的主体通常是 **KV cache**。设单个活跃请求在长度 $S$ 下的 cache 显存为：

$$
M_{cache}^{req}(S)
$$

则峰值时刻的总 cache 显存上界可写为：

$$
M_{cache} = C_{peak}^{budget} \cdot M_{cache}^{req}(S_{p95})
$$

为统一不同 attention 结构，可先定义“**每层每 token 的 cache 字节数**”：

$$
b_{cache}^{layer,token}
$$

则单请求在长度 $S$ 下的 cache 显存统一写为：

$$
M_{cache}^{req}(S) = L \cdot S \cdot b_{cache}^{layer,token}
$$

其中：

- $L$ 为层数；
- $S$ 为该请求当前已占用 cache 的总长度；
- $b_{kv}$ 为 KV cache 精度对应的单元素字节数，例如 FP16/BF16 常取 2 bytes，FP8 常取 1 byte；
- $H_{kv}$ 为 KV 头数；
- $d_{head}$ 为每个头的维度；
- $d_{latent}$ 为 MLA latent cache 维度；
- $b_{aux}$ 为除 latent 向量外，每层每 token 额外常驻的辅助字节数。

对不同 attention 结构，$b_{cache}^{layer,token}$ 可按下式选取。

**MHA**

标准多头注意力下，每层每 token 需要同时保存 K 与 V，因此：

$$
b_{cache}^{layer,token} = 2 \cdot H_{kv} \cdot d_{head} \cdot b_{kv}
$$

若采用标准 MHA，通常 $H_{kv} = H$，且 $H \cdot d_{head} = d_{model}$，于是也可写成：

$$
b_{cache}^{layer,token} = 2 \cdot d_{model} \cdot b_{kv}
$$

从而：

$$
M_{cache}^{req}(S) = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b_{kv}
$$

**GQA**

Grouped-Query Attention 下，cache 规模由 KV 头数而不是 query 头数决定，因此：

$$
b_{cache}^{layer,token} = 2 \cdot H_{kv} \cdot d_{head} \cdot b_{kv}
$$

从而：

$$
M_{cache}^{req}(S) = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b_{kv}
$$

与 MHA 相比，GQA 的主要区别不是公式形式变化，而是 $H_{kv}$ 通常显著小于 query 头数，因此单请求 cache 显存更小。

**MQA**

Multi-Query Attention 可视为 $H_{kv} = 1$ 的特例，因此：

$$
b_{cache}^{layer,token} = 2 \cdot d_{head} \cdot b_{kv}
$$

从而：

$$
M_{cache}^{req}(S) = 2 \cdot L \cdot d_{head} \cdot S \cdot b_{kv}
$$

**MLA**

对 MLA，当前实现不再按显式 K/V 双张量建模，而是按每层每 token 的 latent cache 与辅助状态之和估算：

$$
b_{cache}^{layer,token} = d_{latent} \cdot b_{kv} + b_{aux}
$$

从而：

$$
M_{cache}^{req}(S) = L \cdot S \cdot (d_{latent} \cdot b_{kv} + b_{aux})
$$

例如，若某 MLA 模型取 $d_{latent} = 512$、$b_{kv} = 1$ byte、$b_{aux} = 128$ bytes，则每层每 token 的 cache 字节数为 $512 \times 1 + 128 = 640$ bytes。

若模型采用 Sparse/Hybrid attention，或你手头已有更准确的 profile 数据，则应直接提供更贴近实际的 $b_{cache}^{layer,token}$ 或 $M_{cache}^{req}(S)$，而不是机械套用上述近似式。

### 5.5 显存卡数下界

单张 GPU 的有效可用显存为：

$$
V_{gpu}^{eff} = V_{gpu} \eta_{vram}
$$

因此，由显存约束给出的 GPU 卡数下界为：

$$
G_{mem} = \left\lceil \frac{M_w + M_r + M_{cache}}{V_{gpu}^{eff}} \right\rceil
$$

该式表达的是**总显存量角度的理论下界**，不保证模型在任意单卡或任意卡数组合上都一定可以实际装载。由于本文不讨论部署，它没有显式建模：

- 权重如何在各卡之间分布；
- 多副本重复常驻显存；
- 逐卡装载不均衡；
- 局部显存热点。

因此，$G_{mem}$ 可能低估真实环境中的卡数需求。若后续进入部署设计阶段，应在本文结果基础上进一步做逐卡 profile 和部署可行性验证。

### 5.6 显存口径的适用边界

上述显存模型是总量近似，适用于采购前卡数测算。它没有显式建模以下因素：

- 逐卡显存分布不均；
- MoE expert 热点导致的局部显存压力；
- 通信缓冲、临时 workspace 和碎片化的极端长尾；
- 某些框架下 KV cache 以外的额外序列状态显著膨胀。

因此，若模型或框架具备明显不均匀性，应通过放大 $\alpha_r$、保守设置 $M_{cache}^{req}(S)$ 或直接用 profile 数据覆盖本文公式。

---

## 6. 吞吐约束下的卡数测算

### 6.1 吞吐建模总思路

吞吐约束回答的是：**峰值流量下，系统总共每秒需要处理多少 token，而单卡实际每秒能处理多少 token。**

只要能估计单卡在 Prefill 和 Decode 阶段的有效 token 吞吐能力，就可以直接换算出所需总卡数。

### 6.2 单卡 Prefill 吞吐能力

Prefill 的工作量包括：

- 参数主干计算；
- attention 计算；
- 权重与中间状态相关访存。

由于 Prefill 常对应长输入，attention 计算不应忽略。因此，单卡 Prefill 吞吐能力写为带宽受限与算力受限两种上界中的较小者：

$$
TPS_{pre}^{card}(S) = \min\left(TPS_{pre,bw}^{card}(S),\ TPS_{pre,cmp}^{card}(S)\right)
$$

#### 6.2.1 带宽受限上界

Prefill 带宽受限近似下，单卡有效 token 吞吐可写为：

$$
TPS_{pre,bw}^{card}(S) = \frac{B_{mem} \eta_{bw}}{b_{pre}(S)}
$$

其中，$b_{pre}(S)$ 表示单个输入 token 在 Prefill 阶段平均摊销到的字节访存量。若采用粗粒度量级估算，可进一步近似为：

$$
b_{pre}(S) \approx \frac{P_{act} b_w}{S}
$$

该式表达的是“整段输入在 Prefill 过程中平均摊销一次主要权重读取”的理想化近似，更适合用于量级判断，而非严格微观建模。它默认 Prefill 的主要访存瓶颈仍由权重主项主导；若 attention 中间张量读写、KV cache 写入、softmax 或框架实现带来的额外访存不可忽略，应通过放大 $b_{pre}(S)$ 或降低 $\eta_{bw}$ 进行修正。对短输入、低 batch、算子融合较差或实现存在额外访存放大的场景，该近似可能偏乐观。实际使用时建议通过 profile 标定 $\eta_{bw}$ 来吸收误差。

#### 6.2.2 算力受限上界

Prefill 的 FLOPs 近似写为参数主干项与 attention 项之和：

$$
F_{pre}(S) \approx 2 P_{act} S + \alpha_{attn} S^2
$$

其中，$\alpha_{attn}$ 为将层数、隐藏维度、头数和 attention 实现细节吸收后的 attention 计算系数。

于是单卡在算力受限下的 token 吞吐上界为：

$$
TPS_{pre,cmp}^{card}(S)
=
\frac{F_{peak} \eta_{cmp} \cdot S}{2 P_{act} S + \alpha_{attn} S^2}
=
\frac{F_{peak} \eta_{cmp}}{2 P_{act} + \alpha_{attn} S}
$$

当输入长度增大时，attention 项会抬高单 token 计算成本，因此该式会自然反映长输入对 Prefill 吞吐的压制。

#### 6.2.3 由 Prefill 吞吐约束给出的卡数下界

在主求解中，取保守输入长度 $S_{in,p95}$：

$$
TPS_{pre}^{card} = TPS_{pre}^{card}(S_{in,p95})
$$

于是由 Prefill 吞吐约束给出的卡数下界为：

$$
G_{pre} = \left\lceil \frac{TPS_{pre,target}^{peak}}{TPS_{pre}^{card}} \right\rceil
$$

### 6.3 单卡 Decode 吞吐能力

Decode 阶段通常逐 token 生成，更容易受到显存带宽约束。单卡 Decode token 吞吐可近似写为：

$$
TPS_{dec}^{card} = \min\left(TPS_{dec,bw}^{card},\ TPS_{dec,cmp}^{card}\right)
$$

#### 6.3.1 带宽受限上界

Decode 带宽受限下，可写为：

$$
TPS_{dec,bw}^{card} = \frac{B_{mem} \eta_{bw}}{b_{dec}}
$$

其中，单个输出 token 的平均访存量近似为：

$$
b_{dec} \approx P_{act} b_w
$$

该式表达的是“生成一个 token 时需要访问一次主要激活参数”的粗粒度近似。对于 MoE 模型，$P_{act}$ 应理解为平均激活参数量；若存在明显 expert 热点或路由倾斜，应进一步降低有效带宽利用率或直接用实测值替代。

#### 6.3.2 算力受限上界

Decode 的单 token FLOPs 量级近似为：

$$
F_{dec} \approx 2 P_{act}
$$

于是：

$$
TPS_{dec,cmp}^{card} = \frac{F_{peak} \eta_{cmp}}{2 P_{act}}
$$

#### 6.3.3 由 Decode 吞吐约束给出的卡数下界

$$
G_{dec} = \left\lceil \frac{TPS_{dec,target}^{peak}}{TPS_{dec}^{card}} \right\rceil
$$

### 6.4 吞吐口径的适用边界

本文的 Prefill / Decode 吞吐模型属于采购前总量估算，未显式建模：

- 动态 batching；
- 长短请求混跑；
- 排队与调度；
- 框架对 kernel 融合、paged attention 和 cache 管理的实现差异；
- MoE 路由不均衡造成的时变吞吐波动。

因此，若已有线上或离线 profile 数据，应优先用实测单卡吞吐替代理论公式，本文公式则用于提供量级估算和变量依赖关系。

---

## 7. 时延必要条件

### 7.1 为什么时延不再直接折算总卡数

显存和吞吐都是总量约束，因此可以直接折算为总卡数下界；但时延是**单请求体验量**，能否随着总卡数增加而改善，取决于卡如何组织到单个请求上。

在本文不预设并行组织方式的前提下，无法合理假设：

- 总卡数 $G$ 都能共同服务同一个请求；
- 新增 GPU 一定能线性缩短单请求服务时间；
- 总卡数的增加必然等价于单请求时延的下降。

因此，本文不再使用类似

$$
T^{svc} \approx \frac{S}{G \cdot TPS^{card}}
$$

这样的公式去直接构造时延卡数下界。

更稳妥的做法是：将时延写成**单卡服务能力的必要条件**。如果这个必要条件都不满足，则仅靠增加总卡数而不改变单请求并行方式，通常无法达到目标时延。

### 7.2 Prefill 时延必要条件

对保守输入长度 $S_{in,p95}$，单卡 Prefill 服务时间近似为：

$$
T_{pre}^{card}(S_{in,p95}) = \frac{S_{in,p95}}{TPS_{pre}^{card}(S_{in,p95})}
$$

为使首字时延目标至少在单卡能力层面不矛盾，应满足：

$$
T_{pre}^{card}(S_{in,p95}) \le TTFT_{p95}^{target}
$$

也即：

$$
\frac{S_{in,p95}}{TPS_{pre}^{card}} \le TTFT_{p95}^{target}
$$

若该条件不满足，则说明在当前 GPU 型号和当前单卡有效 Prefill 能力假设下，单请求首字时延目标已经偏紧。此时仅靠“多买卡”但不改变单请求并行方式，通常无法保证 TTFT 目标。

### 7.3 Decode 时延必要条件

对保守输出长度 $S_{out,p95}$，单卡 Decode 服务时间近似为：

$$
T_{dec}^{card}(S_{out,p95}) = \frac{S_{out,p95}}{TPS_{dec}^{card}}
$$

其中，Decode 时间预算为：

$$
T_{dec,p95} = E2E_{p95}^{target} - TTFT_{p95}^{target}
$$

为使尾部生成时间在单卡能力层面不矛盾，应满足：

$$
T_{dec}^{card}(S_{out,p95}) \le T_{dec,p95}
$$

也即：

$$
\frac{S_{out,p95}}{TPS_{dec}^{card}} \le E2E_{p95}^{target} - TTFT_{p95}^{target}
$$

若该条件不满足，则说明当前 GPU 型号或模型规模下，单请求生成速度不足，后续必须依赖更强 GPU、模型压缩，或能改善单请求速度的并行方式。

### 7.4 时延必要条件与吞吐下界的关系

时延必要条件与吞吐下界共享同一套单卡有效服务率模型：

- 吞吐下界约束的是峰值总工作量是否处理得完；
- 时延必要条件约束的是单请求服务时间是否足够短。

二者相关，但不冗余。前者回答“总量承载能力是否足够”，后者回答“单请求体验目标是否在能力上可达”。

在采购前测算中，这种组合方式比直接把时延也机械折算成总卡数更稳健，因为它避免了对并行组织方式的隐含假设。

---

## 8. 最终卡数合成

### 8.1 主结果

由显存与吞吐约束得到的主结果为：

$$
G_{req} = \max(G_{mem}, G_{pre}, G_{dec})
$$

其中：

- $G_{mem}$ 反映总显存量约束；
- $G_{pre}$ 反映峰值输入 token 工作量约束；
- $G_{dec}$ 反映峰值输出 token 工作量约束。

这三个量共同给出在本文总量模型下的**最少 GPU 卡数估算值**。

### 8.2 时延必要条件结论

在给出 $G_{req}$ 的同时，建议同步输出以下两个条件是否满足：

$$
\frac{S_{in,p95}}{TPS_{pre}^{card}} \le TTFT_{p95}^{target}
$$

$$
\frac{S_{out,p95}}{TPS_{dec}^{card}} \le E2E_{p95}^{target} - TTFT_{p95}^{target}
$$

它们不是卡数公式，而是对最终结论的重要解释：

- 若两者都满足，则说明在当前单卡能力假设下，时延目标与总量卡数测算结果基本一致；
- 若其中任一不满足，则说明总卡数虽然可支撑总量，但单请求时延目标仍可能存在根本性风险。

### 8.3 建议输出项

建议最终输出至少包括：

- 最少 GPU 卡数 $G_{req}$；
- 主导约束项（显存 / Prefill 吞吐 / Decode 吞吐）；
- 显存约束对应卡数 $G_{mem}$；
- Prefill 吞吐约束对应卡数 $G_{pre}$；
- Decode 吞吐约束对应卡数 $G_{dec}$；
- Prefill 时延必要条件是否满足；
- Decode 时延必要条件是否满足；
- 关键输入口径与效率系数；
- 额外安全系数或风险说明。

---

## 9. 已知 GPU 卡数后的整体能力回推

### 9.1 回推原则

当 GPU 卡数 $G$ 已经确定后，可以基于本文的总量模型回推这批卡的大致整体能力。这里的“能力”指的是：

- **总量能力**，例如系统总吞吐、可持续 QPS、日 token 总量；
- **容量能力**，例如最大在途请求量；
- **风险能力**，例如时延目标是否存在明显违背风险。

这些回推结果仍是采购前的近似值。它们不依赖具体部署映射，因此更适合作为预算评审、资源冗余分析和能力说明，而不是线上 SLA 承诺。

### 9.2 总 Prefill / Decode 吞吐能力回推

在总量模型下，可将 $G$ 张 GPU 的系统总吞吐近似写为单卡吞吐的线性叠加：

$$
TPS_{pre}^{cap}(G; S) \approx G \cdot TPS_{pre}^{card}(S)
$$

$$
TPS_{dec}^{cap}(G) \approx G \cdot TPS_{dec}^{card}
$$

其中：

- $TPS_{pre}^{cap}(G; S)$ 表示在输入长度 $S$ 下系统总 Prefill 吞吐能力；
- $TPS_{dec}^{cap}(G)$ 表示系统总 Decode 吞吐能力。

用于保守能力说明时，建议取：

$$
TPS_{pre,p95}^{cap}(G) = G \cdot TPS_{pre}^{card}(S_{in,p95})
$$

$$
TPS_{dec,p95}^{cap}(G) = G \cdot TPS_{dec}^{card}
$$

用于常态能力说明时，也可取：

$$
TPS_{pre,avg}^{cap}(G) = G \cdot TPS_{pre}^{card}(S_{in,avg})
$$

$$
TPS_{dec,avg}^{cap}(G) = G \cdot TPS_{dec}^{card}
$$

这里的线性叠加是**系统总服务能力近似**，适用于回答“总共能处理多少 token”，不意味着所有卡都能共同加速单个请求。

### 9.3 可持续 QPS 回推

给定长度口径后，可用系统总 Prefill / Decode 吞吐能力分别折算出相应的可持续请求到达率上限。

按保守口径：

$$
\lambda_{pre,p95}^{sus}(G) \approx \frac{TPS_{pre,p95}^{cap}(G)}{S_{in,p95}}
$$

$$
\lambda_{dec,p95}^{sus}(G) \approx \frac{TPS_{dec,p95}^{cap}(G)}{S_{out,p95}}
$$

于是系统在 P95 长度画像下的可持续 QPS 近似为：

$$
\lambda_{p95}^{sus}(G) \approx \min\left(\lambda_{pre,p95}^{sus}(G),\ \lambda_{dec,p95}^{sus}(G)\right)
$$

按平均口径，也可给出：

$$
\lambda_{pre,avg}^{sus}(G) \approx \frac{TPS_{pre,avg}^{cap}(G)}{S_{in,avg}}
$$

$$
\lambda_{dec,avg}^{sus}(G) \approx \frac{TPS_{dec,avg}^{cap}(G)}{S_{out,avg}}
$$

$$
\lambda_{avg}^{sus}(G) \approx \min\left(\lambda_{pre,avg}^{sus}(G),\ \lambda_{dec,avg}^{sus}(G)\right)
$$

其中：

- $\lambda_{p95}^{sus}(G)$ 更适合作为保守能力说明；
- $\lambda_{avg}^{sus}(G)$ 更适合作为常态能力说明。

### 9.4 日 token 总量与日请求量回推

已知系统总吞吐能力后，可进一步回推 24 小时内可持续处理的 token 总量。

保守口径下：

$$
Tok_{pre,day}^{p95}(G) \approx TPS_{pre,p95}^{cap}(G) \cdot 86400
$$

$$
Tok_{dec,day}^{p95}(G) \approx TPS_{dec,p95}^{cap}(G) \cdot 86400
$$

平均口径下：

$$
Tok_{pre,day}^{avg}(G) \approx TPS_{pre,avg}^{cap}(G) \cdot 86400
$$

$$
Tok_{dec,day}^{avg}(G) \approx TPS_{dec,avg}^{cap}(G) \cdot 86400
$$

对应的可持续日请求量近似为：

$$
Req_{day}^{p95}(G) \approx \lambda_{p95}^{sus}(G) \cdot 86400
$$

$$
Req_{day}^{avg}(G) \approx \lambda_{avg}^{sus}(G) \cdot 86400
$$

若业务具有明显昼夜波动或离散尖峰，上述日总量应理解为“在该口径下按全天持续运行折算出的近似能力上限”。

### 9.5 最大在途请求量回推

在不讨论部署的前提下，最大在途请求量最稳妥的回推方式来自显存剩余量。

给定总卡数 $G$ 后，总有效显存为：

$$
V_{cluster}^{eff}(G) = G \cdot V_{gpu}^{eff}
$$

扣除权重与运行时固定显存后，可用于活跃请求的显存预算近似为：

$$
V_{cache}^{avail}(G) = V_{cluster}^{eff}(G) - M_w - M_r
$$

若取参考序列长度 $S_{ref}$，则显存角度的最大在途请求量上界可写为：

$$
C_{max}^{mem}(G; S_{ref}) = \left\lfloor \frac{V_{cache}^{avail}(G)}{M_{cache}^{req}(S_{ref})} \right\rfloor
$$

其中 $S_{ref}$ 可按用途选取：

- 用 $S_{p95}$ 时，得到保守的最大在途请求量；
- 用 $S_{avg}$ 时，得到常态画像下的近似在途承载量。

因此可定义：

$$
C_{max,p95}^{mem}(G) = C_{max}^{mem}(G; S_{p95})
$$

$$
C_{max,avg}^{mem}(G) = C_{max}^{mem}(G; S_{avg})
$$

若需要与业务目标对比，还可计算显存余量系数：

$$
\rho_{conc,p95}(G) = \frac{C_{max,p95}^{mem}(G)}{C_{peak}^{budget}}
$$

其中：

- 若 $\rho_{conc,p95}(G) \ge 1$，则说明在峰值在途预算口径下显存容量基本够用；
- 若 $\rho_{conc,p95}(G) < 1$，则说明当前卡数在显存角度下对峰值在途请求量仍偏紧。

### 9.6 时延风险回推

在本文方法下，不对已知总卡数直接输出精确 TTFT 或 E2E 数值，因为那需要具体的并行组织方式、调度策略和排队建模。

但可以基于单卡时延必要条件给出风险判断：

- 若 Prefill 与 Decode 两个必要条件都满足，则说明在单卡能力层面，时延目标与当前总量能力没有明显根本冲突；
- 若 Prefill 必要条件不满足，则首字时延存在较高风险；
- 若 Decode 必要条件不满足，则尾部生成时长存在较高风险；
- 若系统实际运行点接近 $\lambda_{p95}^{sus}(G)$，即使单卡必要条件满足，排队和调度仍可能放大真实时延。

因此，时延回推建议输出为“风险级别”或“必要条件是否满足”，而不是拍定一个精确时延值。

### 9.7 建议新增输出项

当文档既要回答“需要多少卡”，又要回答“这些卡能干什么”时，建议在最终结果中额外输出：

- 已知总卡数 $G$ 下的总 Prefill 吞吐能力；
- 已知总卡数 $G$ 下的总 Decode 吞吐能力；
- 保守可持续 QPS $\lambda_{p95}^{sus}(G)$；
- 常态可持续 QPS $\lambda_{avg}^{sus}(G)$；
- 保守日输入 / 输出 token 总量；
- 常态日输入 / 输出 token 总量；
- 显存角度的最大在途请求量；
- 时延风险说明。

---

## 10. 使用建议与风险提示

### 10.1 优先使用实测单卡能力

若已有线上或离线 profile 数据，应优先使用以下实测值替代理论公式：

- 单卡 Prefill 吞吐 $TPS_{pre}^{card}(S)$；
- 单卡 Decode 吞吐 $TPS_{dec}^{card}$；
- 单请求 KV cache 显存 $M_{cache}^{req}(S)$。

理论公式主要用于：

- 提供量级估算；
- 帮助理解变量之间的依赖；
- 在缺少 profile 数据时给出初版预算值。

### 10.2 对 MoE 模型的额外提醒

对于 MoE 模型，文中的 $P_{act}$ 是平均激活参数量近似。若存在明显 expert 热点、路由倾斜或 token 分布高度不均，则：

- 吞吐可能低于本文公式预测；
- 显存局部热点可能高于本文总量估算；
- 时延必要条件可能比公式结果更苛刻。

因此，MoE 场景建议更保守地设置 $\eta_{bw}$、$\eta_{cmp}$ 和显存附加系数，或直接使用 profile 数据覆盖。

### 10.3 何时需要超出本文方法

若你的目标已经从“采购前需要多少卡”进一步转向：

- 具体实例部署；
- 单请求跨卡并行加速；
- TTFT / E2E 的严格 SLA 证明；
- 调度、排队和长尾请求建模；
- 多机拓扑与通信开销优化；

则应超出本文范围，转入真实部署设计与压测阶段。此时，本文结果应被视为**总量测算起点**，而不是终局答案。

---

## 11. 结论

本文将大模型在线推理的 GPU 卡数测算统一为一套纯总量方法：

- 用显存约束估算总显存量对应的卡数下界；
- 用 Prefill / Decode 吞吐约束估算峰值工作量对应的卡数下界；
- 取三者最大值作为最少 GPU 卡数：

$$
G_{req} = \max(G_{mem}, G_{pre}, G_{dec})
$$

同时，本文不再把时延机械折算成总卡数，而是将其改写为**单卡服务能力必要条件**。这种写法避免了在未知并行组织方式时，把“总卡数增加”直接等价为“单请求时延下降”的不稳妥假设。

在此基础上，本文进一步给出了一套已知卡数后的整体能力回推方法，用于回答：

- 这批卡的总吞吐大致有多大；
- 能支撑的可持续 QPS 大致有多高；
- 一天内大致能处理多少 token 和多少请求；
- 显存角度最多能容纳多少在途请求；
- 时延目标是否存在明显根本性风险。

因此，本文的最终使用方式应是：

1. 先用显存与吞吐公式得到总卡数估算值；
2. 再用时延必要条件判断该估算值在体验目标上是否存在根本性矛盾；
3. 最后结合能力回推结果，对资源预算、业务冗余和未来增长空间做整体判断。

这套方法既能回答“需要多少卡”，也能回答“这些卡大致能做多少事”，更适合采购前规划、预算评审和资源能力说明场景。
