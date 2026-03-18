# 大模型在线推理 GPU 卡数测算方法

---

## 1. 文档目标与边界

本文给出一套面向**采购前容量规划**的大模型在线推理 GPU 资源测算方法。本文的主目标不是先求“需要多少实例”，而是直接回答：

> **为了满足给定业务目标，最少需要多少张 GPU 卡？**

因此，本文的核心输出记为

$$
G_{biz}
$$

表示满足业务需求所需的**最少 GPU 卡数**。在此基础上，实例数、部署副本数、单实例卡数等量只作为**实现映射结果**出现，而不再作为主求解目标。

本文适用于以下场景：

- 采购前 GPU 数量预算；
- 不同模型 / GPU / 部署方式的横向比较；
- 容量规划和扩容预估；
- 对体验目标是否可达进行前期可行性判断。

本文不直接处理以下更细粒度问题，这些因素统一吸收到少量效率系数中：

- TP / EP / PP 的精细通信建模；
- 调度器、连续批处理和排队分布的严格建模；
- 特定推理框架对 kernel、算子融合、paged attention 的实现差异；
- 严格 SLA 证明。

因此，本文输出的是**采购前方法论上的合理估算值**，而不是生产环境最终承诺值。

---

## 2. 总体方法框架

整套方法按六个阶段展开。

```mermaid
flowchart LR
    A[阶段 1<br/>输入整理与预处理]
    B[阶段 2<br/>由 QPS 与时延目标推导系统需求]
    C[阶段 3<br/>候选单实例能力建模]
    D[阶段 4<br/>换算为总 GPU 卡数需求]
    E[阶段 5<br/>选择最小 GPU 卡数方案]
    F[阶段 6<br/>部署映射与一致性校验]

    A --> B --> C --> D --> E --> F
```

这六个阶段分别回答六个问题：

- 阶段 1：业务、模型、GPU 信息需要整理成哪些基础统计量？
- 阶段 2：业务给出的 QPS、TTFT、E2E 目标，对系统提出了什么 Prefill / Decode 工作量和时延要求？
- 阶段 3：当单实例使用 $g$ 张 GPU 时，该实例在显存、Prefill、Decode 维度上分别有多少能力？
- 阶段 4：在该 $g$ 下，为满足显存、Prefill、Decode 约束，总共需要多少张 GPU 卡？
- 阶段 5：在所有可行候选 $g$ 中，哪一种方案对应的总 GPU 卡数最小？
- 阶段 6：在得到总 GPU 卡数后，应如何映射为实例数、部署方案，以及如何做结果一致性校验？

本文的主线始终围绕：

> **先求总 GPU 卡数，再讨论如何把这些 GPU 组织成实例。**

---

## 3. 基本对象定义

### 3.1 GPU

GPU 是本文最基本的硬件资源单位。单张 GPU 具有自己的显存容量、显存带宽和峰值算力，分别记为：

- $V_{gpu}$：单卡显存容量；
- $B_{mem}$：单卡峰值显存带宽；
- $F_{peak}$：单卡在目标精度下的峰值算力。

本文最终关心的是总 GPU 卡数，即满足业务需求所需的总卡数：

$$
G
$$

### 3.2 实例

实例是模型服务的最小运行单元。一个实例由 $g$ 张 GPU 组成，这里的 $g$ 表示**单实例 GPU 数**。

实例在本文中的角色是：

- 用于判断某种单实例组织方式是否显存可行；
- 用于计算该组织方式下的单实例 Prefill / Decode 能力；
- 用于把总 GPU 卡数映射为部署方案。

因此，实例是**中间建模对象**，不是主输出目标。

### 3.3 副本与集群

若某种实例配置已经确定，例如“每个实例使用 $g$ 张 GPU”，那么部署 $N_{inst}$ 个这样的实例，就意味着系统中有 $N_{inst}$ 个副本。

集群总 GPU 数为

$$
G=N_{inst}\cdot g
$$

后文会先求总 GPU 卡数 $G$，再在给定 $g$ 的条件下反推出实例数

$$
N_{inst}=\left\lceil\frac{G}{g}\right\rceil
$$

---

## 4. 输入整理与预处理

### 4.1 输入项

测算至少需要五类输入。

#### 4.1.1 业务目标

业务侧通常提供以下目标：

- 平均请求到达率 $\lambda_{avg}$；
- 峰值请求到达率 $\lambda_{peak}$；
- 平均首字时延目标 $TTFT_{avg}^{target}$；
- P95 首字时延目标 $TTFT_{p95}^{target}$；
- 平均单次总时延目标 $E2E_{avg}^{target}$；
- P95 单次总时延目标 $E2E_{p95}^{target}$。

这里的 $\lambda$ 指**模型调用层面的请求到达率**，单位为 req/s。对外服务型 Agent 应用若业务侧只给出用户请求 QPS，则应先按“单个用户请求平均触发多少次模型调用”折算为等效模型调用 QPS，再进入后续测算。

#### 4.1.2 请求长度画像

至少需要：

- 平均输入长度 $S_{in,avg}$；
- 平均输出长度 $S_{out,avg}$；
- P95 输入长度 $S_{in,p95}$；
- P95 输出长度 $S_{out,p95}$。

进一步定义平均总长度与保守总长度：

$$
S_{avg}=S_{in,avg}+S_{out,avg}
$$

$$
S_{p95}=S_{in,p95}+S_{out,p95}
$$

若业务侧单独提供真实的 P95 总长度，则优先使用真实值替代上式。

#### 4.1.3 模型信息

模型侧至少需要：

- 总参数量 $P_{total}$；
- 每个 token 的激活参数量 $P_{act}$；
- 层数 $L$；
- 隐藏维度 $H$；
- attention 结构相关信息，例如 KV 头数、head 维度、MLA latent 维度；
- 权重精度和 KV cache 精度。

对于 Dense 模型，通常有

$$
P_{act}=P_{total}
$$

对于 MoE 模型，$P_{act}$ 必须理解为**每个 token 实际参与前向的总参数量**，应包含共享层、路由相关开销近似以及被激活 expert 的总和。

#### 4.1.4 GPU 与效率系数

GPU 侧至少需要：

- 单卡显存容量 $V_{gpu}$；
- 单卡峰值显存带宽 $B_{mem}$；
- 单卡在当前精度下的峰值算力 $F_{peak}$。

此外还需要少量效率系数：

- 可用显存比例 $\eta_{vram}$；
- 带宽利用率 $\eta_{bw}$；
- 算力利用率 $\eta_{cmp}$；
- 多卡实例扩展效率 $\eta_{inst}$。

这些量都应满足

$$
0<\eta_{vram},\eta_{bw},\eta_{cmp},\eta_{inst}\le 1
$$

#### 4.1.5 显存附加系数与候选单实例卡数

显存建模还需要两个附加系数：

- 权重附加显存系数 $\alpha_w$；
- 运行时固定显存系数 $\alpha_r$。

此外还需给出候选单实例 GPU 数集合：

$$
\mathcal{G}
$$

其中

$$
g\in\mathcal{G}
$$

表示“一个实例使用 $g$ 张 GPU”是一种候选部署组织方式。本文不会把实例数作为主目标，但仍需要枚举不同的 $g$ 来判断单实例能力，并最终选出总 GPU 卡数最小的方案。

### 4.2 合法性校验

进入正式计算前，至少检查：

- $\lambda_{peak}\ge \lambda_{avg}>0$；
- $TTFT_{avg}^{target}>0$，且 $TTFT_{p95}^{target}\ge TTFT_{avg}^{target}$；
- $E2E_{avg}^{target}>TTFT_{avg}^{target}$；
- $E2E_{p95}^{target}>TTFT_{p95}^{target}$；
- 所有效率系数位于 $(0,1]$ 内。

并定义 Decode 的时间预算：

$$
T_{dec,avg}=E2E_{avg}^{target}-TTFT_{avg}^{target}
$$

$$
T_{dec,p95}=E2E_{p95}^{target}-TTFT_{p95}^{target}
$$

---

## 5. 由 QPS 与时延目标推导系统需求

本阶段回答的问题是：**业务侧给出 QPS、TTFT、E2E 目标后，系统在 Prefill 和 Decode 两个阶段分别至少要承担多大的 token 工作量？时延目标又如何影响总 GPU 卡数测算？**

### 5.1 建模思路

在线推理请求通常分成两个阶段：

- **Prefill**：处理整段输入，决定首字何时出来；
- **Decode**：逐 token 生成输出，决定尾部时延和生成速度。

本文采用一个显式假设：

> **将 TTFT 近似视为模型 Prefill 时间。**

这意味着排队、调度、网络和服务栈开销不再单独拆分，而是统一吸收到效率系数和结果保守性中。

### 5.2 QPS 驱动下的系统总 token 工作量

在 QPS 驱动口径下，系统总工作量直接由“请求到达率 × 单请求 token 量”得到。

平均 Prefill 工作量为

$$
TPS_{pre,target}^{avg}=\lambda_{avg}S_{in,avg}
$$

峰值 Prefill 工作量为

$$
TPS_{pre,target}^{peak}=\lambda_{peak}S_{in,p95}
$$

平均 Decode 工作量为

$$
TPS_{dec,target}^{avg}=\lambda_{avg}S_{out,avg}
$$

峰值 Decode 工作量为

$$
TPS_{dec,target}^{peak}=\lambda_{peak}S_{out,p95}
$$

这组公式只表达**单位时间内必须处理多少 token**，不直接表达时延约束。

### 5.3 由 QPS 与时延目标反推在途请求预算

时延目标不会改变 token 总量，但会改变系统必须容纳多少“在途请求”，从而影响 KV cache 压力。

平均口径下，可按 Little 定律近似得到在途请求数：

$$
C_{avg}^{budget}\approx \lambda_{avg}E2E_{avg}^{target}
$$

峰值口径下，为采购前保守估算 cache 压力，采用峰值请求到达率与高分位端到端时延组合后的预算上界：

$$
C_{peak}^{budget}\approx \lambda_{peak}E2E_{p95}^{target}
$$

因此：

- $C_{avg}^{budget}$ 近似表示平均在途请求规模；
- $C_{peak}^{budget}$ 更准确地说是峰值在途请求预算上界。

### 5.4 Prefill 与 Decode 的统一阶段约束

在本文采用的两阶段稳态近似下，Prefill 与 Decode 阶段都可以写成“系统阶段工作量不超过集群阶段处理能力”的形式。

对 Prefill 而言，峰值阶段工作量为

$$
TPS_{pre,target}^{peak}=\lambda_{peak}S_{in,p95}
$$

若集群总 Prefill 能力至少不低于这一工作量，则既能承接峰值输入 token 流量，也与 $TTFT_{p95}^{target}$ 的阶段要求保持一致。因此，本文将 Prefill 的工作量约束与阶段时延要求合并为统一的 Prefill 约束。

对 Decode 而言，峰值阶段工作量为

$$
TPS_{dec,target}^{peak}=\lambda_{peak}S_{out,p95}
$$

在本文当前的稳态近似下，Decode 阶段的主要时延压力由上述 Decode 工作量约束代表，因此主流程中不再单独列出一个独立的 Decode 时延算卡项。但对外服务场景下，真实的流式输出体验仍可能受到排队、调度与输出长度波动影响，因此第 9 章仍保留一致性校验说明。

因此，在主流程中，系统最终只保留三类约束：

- **Prefill 约束**；
- **Decode 约束**；
- **显存约束**。

---

## 6. 候选单实例能力建模

本阶段回答的问题是：**当单实例使用 $g$ 张 GPU 时，该实例在显存、Prefill、Decode 三个维度上分别有多少能力？**

### 6.1 建模说明

虽然本文最终目标是总 GPU 卡数，但单 GPU 如何被组织成实例，仍会影响：

- 模型是否装得下；
- 单实例扩展效率；
- 单实例可承载多少请求；
- 单实例 Prefill / Decode 能力。

因此，需要先对每个候选 $g$ 建立单实例能力模型，再把这些单实例能力换算为总 GPU 卡数需求。

### 6.2 单实例显存能力

设权重精度对应每个参数占用 $b_w$ 个字节，则原始权重字节数为

$$
M_w^{raw}=P_{total}b_w
$$

考虑附加开销后，权重显存为

$$
M_w=M_w^{raw}(1+\alpha_w)
$$

运行时固定显存近似为

$$
M_r=\alpha_r M_w
$$

设 cache 精度对应每个 cache 元素占用 $b_c$ 个字节，单层单 token cache 字节数记为 $e_{cache}$。若为标准 MHA，可近似写为

$$
e_{cache}\approx 2Hb_c+e_{aux}
$$

这里可将标准 MHA 下的 K/V 总维度近似视为隐藏维度 $H$。

于是，单请求在总长度为 $S$ 时的 cache 显存近似为

$$
M_{cache,req}(S)=LS e_{cache}
$$

当单实例由 $g$ 张 GPU 组成时，其可用显存近似为

$$
M_{use}(g)=gV_{gpu}\eta_{vram}
$$

这里采用的是实例级可用显存近似，默认权重与相关常驻状态可在该实例的 $g$ 张 GPU 上近似分摊；它不等价于具体 TP、EP 或 PP 策略下的逐卡严格显存证明。

若

$$
M_w+M_r\ge M_{use}(g)
$$

则当前 $g$ 显存不可行。

若显存可行，则单实例可用于承载活跃请求 cache 的容量为

$$
M_{cache,cap}(g)=M_{use}(g)-M_w-M_r
$$

于是，单实例在显存约束下可承载的保守请求数上界为

$$
N_{req}^{mem}(g)=\left\lfloor\frac{M_{cache,cap}(g)}{M_{cache,req}(S_{p95})}\right\rfloor
$$

### 6.3 单实例 Decode 能力

将单个输出 token 的近似权重访存量写为

$$
b_{dec}=P_{act}b_w
$$

将单个输出 token 的近似计算量写为

$$
f_{dec}=2P_{act}
$$

这里抓的是线性层、MLP、MoE 主体的主计算项；attention、norm、rope 等次级开销统一吸收到效率系数中。

于是，单 GPU 的 Decode 带宽上界为

$$
TPS_{dec,mem}^{gpu}=\frac{B_{mem}\eta_{bw}}{b_{dec}}
$$

单 GPU 的 Decode 算力上界为

$$
TPS_{dec,cmp}^{gpu}=\frac{F_{peak}\eta_{cmp}}{f_{dec}}
$$

单 GPU Decode 吞吐为

$$
TPS_{dec}^{gpu}=\min\left(TPS_{dec,mem}^{gpu},\,TPS_{dec,cmp}^{gpu}\right)
$$

因此，单实例 Decode 吞吐近似为

$$
TPS_{dec}^{inst}(g)=g\,TPS_{dec}^{gpu}\,\eta_{inst}
$$

### 6.4 单实例 Prefill 能力

设某次 Prefill 的输入长度为 $S$。

对主体计算部分，总计算量近似为

$$
F_{body}^{pre}(S)\approx 2P_{act}S
$$

对 self-attention 部分，若按每层 $QK^\top$ 与 $AV$ 两个主矩阵乘做同阶近似，则额外计算量写为

$$
F_{attn}^{pre}(S)\approx 4LHS^2
$$

这里的常数 4 是把每层 attention 的两个主矩阵乘按粗粒度常数合并后的近似系数。

于是，单次 Prefill 总计算量近似为

$$
F_{pre}(S)\approx 2P_{act}S+4LHS^2
$$

每个输入 token 的平均计算量为

$$
f_{pre}(S)=\frac{F_{pre}(S)}{S}\approx 2P_{act}+4LHS
$$

对访存部分，可将“读取一遍激活权重，服务整段输入”作为近似，因此每个输入 token 分摊到的平均权重访存量为

$$
b_{pre}(S)=\frac{P_{act}b_w}{S}
$$

这里的带宽模型抓的是权重主项；attention 中间读写、softmax、cache 写入等二级访存统一吸收到 $\eta_{bw}$ 中。

于是，单 GPU 的 Prefill 带宽上界为

$$
TPS_{pre,mem}^{gpu}(S)=\frac{B_{mem}\eta_{bw}S}{P_{act}b_w}
$$

单 GPU 的 Prefill 算力上界为

$$
TPS_{pre,cmp}^{gpu}(S)=\frac{F_{peak}\eta_{cmp}}{2P_{act}+4LHS}
$$

单 GPU Prefill 吞吐为

$$
TPS_{pre}^{gpu}(S)=\min\left(TPS_{pre,mem}^{gpu}(S),\,TPS_{pre,cmp}^{gpu}(S)\right)
$$

因此，单实例 Prefill 吞吐近似为

$$
TPS_{pre}^{inst}(g,S)=g\,TPS_{pre}^{gpu}(S)\,\eta_{inst}
$$

用于峰值资源求解时，通常取

$$
S=S_{in,p95}
$$

---

## 7. 由单实例能力换算总 GPU 卡数需求

本阶段回答的问题是：**在给定单实例 GPU 数 $g$ 的情况下，为满足显存、Prefill、Decode 三类约束，总共需要多少张 GPU 卡？**

### 7.1 基本思路

对每个候选 $g\in\mathcal{G}$，先计算：

- 单实例显存承载能力 $N_{req}^{mem}(g)$；
- 单实例 Prefill 吞吐 $TPS_{pre}^{inst}(g,S_{in,p95})$；
- 单实例 Decode 吞吐 $TPS_{dec}^{inst}(g)$。

然后分别换算为三类总 GPU 卡数需求：

- 显存约束给出的卡数需求 $G_{mem}(g)$；
- Prefill 约束给出的卡数需求 $G_{pre}(g)$；
- Decode 约束给出的卡数需求 $G_{dec}(g)$。

最终，该 $g$ 下的业务卡数需求为三者最大值。

### 7.2 Prefill 约束对应的总 GPU 卡数

若单实例在 $S_{in,p95}$ 下的 Prefill 吞吐为 $TPS_{pre}^{inst}(g,S_{in,p95})$，则为了满足峰值 Prefill 工作量

$$
TPS_{pre,target}^{peak}=\lambda_{peak}S_{in,p95}
$$

至少需要的实例数为

$$
N_{inst}^{pre}(g)=\left\lceil\frac{TPS_{pre,target}^{peak}}{TPS_{pre}^{inst}(g,S_{in,p95})}\right\rceil
$$

对应的总 GPU 卡数为

$$
G_{pre}(g)=g\cdot N_{inst}^{pre}(g)
$$

### 7.3 Decode 约束对应的总 GPU 卡数

若单实例 Decode 吞吐为 $TPS_{dec}^{inst}(g)$，则为了满足峰值 Decode 工作量

$$
TPS_{dec,target}^{peak}=\lambda_{peak}S_{out,p95}
$$

至少需要的实例数为

$$
N_{inst}^{dec}(g)=\left\lceil\frac{TPS_{dec,target}^{peak}}{TPS_{dec}^{inst}(g)}\right\rceil
$$

对应的总 GPU 卡数为

$$
G_{dec}(g)=g\cdot N_{inst}^{dec}(g)
$$

### 7.4 显存约束对应的总 GPU 卡数

在纯 GPU 总量视角下，显存约束可先给出一个理论下界。峰值在途请求预算上界对应的总 cache 需求近似为

$$
M_{cache,total}^{upper}=C_{peak}^{budget}\cdot M_{cache,req}(S_{p95})
$$

于是，总 GPU 卡数的显存下界可写为

$$
G_{mem,lb}=
\left\lceil
\frac{M_w+M_r+M_{cache,total}^{upper}}
{V_{gpu}\eta_{vram}}
\right\rceil
$$

该式回答的是“总显存量是否足够”，可作为快速下界。

但本文最终仍需在候选 $g$ 上检查单实例显存可行性，并将总显存需求映射到可部署的实例组织方式。若单实例显存上最多承载 $N_{req}^{mem}(g)$ 个保守长度请求，而峰值在途请求预算上界为 $C_{peak}^{budget}$，则至少需要的实例数为

$$
N_{inst}^{mem}(g)=\left\lceil\frac{C_{peak}^{budget}}{N_{req}^{mem}(g)}\right\rceil
$$

对应的总 GPU 卡数为

$$
G_{mem}(g)=g\cdot N_{inst}^{mem}(g)
$$

若 $N_{req}^{mem}(g)\le 0$，则当前 $g$ 显存不可行，应直接剔除。

### 7.5 给定 $g$ 时的业务最少 GPU 卡数

对当前候选 $g$，业务所需最少 GPU 卡数为

$$
G_{biz}(g)=\max\left(G_{mem}(g),\,G_{pre}(g),\,G_{dec}(g)\right)
$$

其中取到最大值的那一项，即为该 $g$ 下的主导约束项。

这里需要强调两点：

1. 在本文当前的两阶段稳态近似下，Prefill 的阶段时延要求已经并入统一 Prefill 约束中；对 Decode 而言，主流程中采用工作量约束来代表其主要阶段时延压力，但真实流式体验仍可能受到排队、调度与输出长度波动影响，因此第 9 章保留一致性校验说明；
2. 在 QPS 驱动口径下，更严格的 E2E 目标不会改变单位时间 token 工作量本身，因此不直接改变 $G_{pre}(g)$ 与 $G_{dec}(g)$ 的定义；它主要通过提高在途请求预算上界来抬高 $G_{mem}(g)$，并在第 9 章的一致性校验中体现对可持续 QPS 的收紧。

---

## 8. 选择最小 GPU 卡数方案

### 8.1 业务所需最少 GPU 卡数

在所有可行候选 $g\in\mathcal{G}$ 中，业务所需最少 GPU 卡数定义为

$$
G_{biz}=\min_{g\in\mathcal{G}_{feasible}} G_{biz}(g)
$$

其中 $\mathcal{G}_{feasible}$ 表示显存可行的候选集合。

这一步体现了本文与“实例数驱动”写法的根本区别：

> **本文直接优化总 GPU 卡数，而不是先优化实例数。**

### 8.2 选型规则

若多个候选 $g$ 得到相同的最少 GPU 卡数，则可按以下顺序择优：

1. 优先选择分片更少、部署更简单的方案；
2. 若仍相同，优先故障单元更小的方案；
3. 若仍相同，优先吞吐冗余更大的方案。

上述偏好主要适用于不同候选 $g$ 下总 GPU 卡数相同、且不存在额外拓扑或库存约束的场景。

### 8.3 高可用展开后的最终 GPU 卡数

设高可用展开后的最终 GPU 卡数记为

$$
G_{final}
$$

若无额外高可用要求，则

$$
G_{final}=G_{biz}
$$

若需要 N+1、整节点容灾或其他冗余要求，则在 $G_{biz}$ 基础上按故障单元折算增加相应 GPU 卡数。

---

## 9. 部署映射与一致性校验

本阶段不再是主求解，而是在已经得到总 GPU 卡数后，回答两个问题：

1. 这些 GPU 应如何组织成实例与副本？
2. 在这一资源规模下，反推出的能力与输入目标是否一致？

### 9.1 由总 GPU 卡数映射为实例数

一旦选定最终采用的单实例 GPU 数 $g^*$，则业务实例数为

$$
N_{inst}^{biz}=\left\lceil\frac{G_{biz}}{g^*}\right\rceil
$$

高可用展开后的最终实例数为

$$
N_{inst}^{final}=\left\lceil\frac{G_{final}}{g^*}\right\rceil
$$

因此，实例数仅用于把总 GPU 卡数映射为具体部署形态；它不是主求解目标，也不参与第 7、8 章的主优化目标定义。

### 9.2 集群总吞吐能力

在最终选定 $g^*$ 并完成部署映射后，若实际承载业务流量的实例数记为 $N_{inst}^{eff}$，则集群总 Decode 吞吐为

$$
TPS_{dec}^{cluster}=N_{inst}^{eff}TPS_{dec}^{inst}(g^*)
$$

集群总 Prefill 吞吐为

$$
TPS_{pre}^{cluster}=N_{inst}^{eff}TPS_{pre}^{inst}(g^*,S_{in,p95})
$$

### 9.3 可持续 QPS 近似能力

在当前规模下，Decode 侧可持续 QPS 近似为

$$
\lambda_{dec}^{sus}=\frac{TPS_{dec}^{cluster}}{S_{out,p95}}
$$

Prefill 侧可持续 QPS 近似为

$$
\lambda_{pre}^{sus}=\frac{TPS_{pre}^{cluster}}{S_{in,p95}}
$$

因此，总可持续 QPS 近似为

$$
\lambda^{sus}=\min\left(\lambda_{dec}^{sus},\,\lambda_{pre}^{sus}\right)
$$

### 9.4 在途请求规模校验

平均在途请求数近似为

$$
C_{avg}^{est}\approx \lambda_{avg}E2E_{avg}^{target}
$$

峰值在途请求预算上界近似为

$$
C_{peak}^{upper}\approx \lambda_{peak}E2E_{p95}^{target}
$$

当前方案在显存约束下的集群请求承载能力近似为

$$
C_{mem}^{cap}\approx N_{inst}^{eff}\cdot N_{req}^{mem}(g^*)
$$

应检查其不低于峰值在途请求预算上界

$$
C_{mem}^{cap}\ge C_{peak}^{upper}
$$

若前者显著小于后者，则说明当前方案在 KV cache 维度上仍不满足目标。

### 9.5 时延一致性说明

在本文当前的两阶段稳态近似下，Prefill 与 Decode 的阶段时延要求已经直接并入第 7 章的 GPU 卡数约束中。因此，第 9 章不再额外给出一个独立于主流程之外的“时延算卡公式”。

需要强调的是：当实际请求到达率逼近或超过第 9.3 节反推的可持续 QPS 能力时，排队与调度效应会迅速放大，真实时延可能显著劣化。因此，第 9 章仍然是**一致性校验**，而不是严格 SLA 证明。

---

## 10. 输出结果建议

最终输出建议至少包括五类内容。

### 10.1 核心假设摘要

建议首先明确列出关键假设，例如：

- TTFT 近似等于模型 Prefill 时间；
- Prefill 吞吐采用输入长度相关近似模型；
- 多卡显存按实例级分片上界处理；
- 效率系数用于吸收框架实现、通信与服务栈差异。

### 10.2 输入摘要

概括业务目标、长度画像、模型关键信息、GPU 规格、效率系数与候选 $g$ 集合，尤其明确采用的是模型调用 QPS 还是由用户请求 QPS 折算得到的等效模型调用 QPS。

### 10.3 GPU 卡数测算结果

明确给出：

- 各候选 $g$ 下的 $G_{mem}(g)$、$G_{pre}(g)$、$G_{dec}(g)$；
- 各候选 $g$ 下的主导约束项；
- 最终选定候选 $g^*$ 及其对应的 $G_{biz}(g^*)$；
- 业务最少 GPU 卡数 $G_{biz}$；
- 高可用后的最终 GPU 卡数 $G_{final}$。

### 10.4 部署映射结果

在确定最终采用的 $g^*$ 后，再给出：

- 单实例 GPU 数 $g^*$；
- 业务实例数 $N_{inst}^{biz}$；
- 高可用后的最终实例数 $N_{inst}^{final}$；
- 当前方案是单卡完整装载还是多卡分片装载。

### 10.5 一致性校验结果

建议补充：

- 当前规模下的可持续 QPS 近似能力；
- 在途请求规模校验结果；
- 主导约束项解释；
- 对偏保守来源的说明，例如采用了 $S_{in,p95}$、$S_{out,p95}$、$S_{p95}$ 等保守长度口径。
