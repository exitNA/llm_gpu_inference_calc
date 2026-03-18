# 大模型在线推理 GPU 资源测算方法

---

## 1. 文档目标

本文用于在线推理场景下的大模型 GPU 资源测算，给出一套从**输入条件**出发，经过**部署方案求解**，最终输出**GPU 数量与预期可达到效果**的完整方法。

本文目标不是复现精确 benchmark，而是提供一套：

- 逻辑自洽
- 公式口径统一
- 便于理解与评审
- 可用于前期 sizing / 方案比较 / 容量规划 / 采购前预算评估

的估算方法。

本文输出：

- 推荐部署方案
- 单实例所需 GPU 数 $g$
- 是否需要分片
- 部署模式
- 业务所需 GPU 数 $G_{biz}$
- 考虑高可用后的最终 GPU 数 $G_{final}$
- 基于已推导 GPU 数可达到的并发、吞吐、生成速度、时延等预期效果

本文强调：

- 核心用途是**采购前 GPU 资源预估**
- 用户不需要预先指定“单卡完整装载”还是“多卡分片装载”
- 这些都是测算过程自动推导出来的结果，而不是原始输入

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
2. 单实例内部 TP / EP / PP 的通信开销只通过“多卡扩展效率”近似吸收，不做精细链路建模。
3. Prefill 吞吐在缺少实测与详细到达率建模时，采用**保守代理模型**，其定位是采购前上界估算，而不是对真实 prefill 机理的严格还原。
4. 平均 / P95 对话时延反推属于 sanity check，不构成 SLA 证明。
5. 多维 P95 长度拼装属于保守 sizing，而非严格联合分布建模。
6. 多卡实例的显存可行性，在采购前阶段按**实例级 pooled/sharded 可行性上界**处理，不等价于 replicated 多卡实例的严格显存证明。
7. 文中主变量默认遵循全局单位约定：显存相关量用 $byte$，带宽相关量用 $byte/s$，算力相关量用 $FLOP/s$，token 吞吐相关量用 $token/s$；仅在结果展示层转换为人类易读单位。

若需要更高精度建模，应补充：

- 多卡实例内通信拓扑与并行方式
- 请求到达率与 burst 窗口
- request shape 联合分布
- 真实线上 profiling 或 benchmark 数据
- Prefill / Decode 分阶段实测吞吐

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

- $g$：单实例占用的 GPU 数
- $N_{inst}$：实例数
- $G_{total}=N_{inst}\times g$
- $is_{shard}$：单实例内部是否需要跨卡切分模型 / 缓存
- $mode_{deploy}$：部署模式，取值为“单卡完整装载”或“多卡分片装载”
- $G_{fail}$：一次最小故障会损失的 GPU 数
- $type_{mem}$：显存可行性口径，取值为“严格可行”或“分片上界可行”

### 4.3 哪些量是求解结果

本文不将以下量视为原始输入，而将其视为**部署求解结果**：

- 单实例 GPU 数 $g$
- 是否分片 $is_{shard}$
- 部署模式 $mode_{deploy}$
- 故障单元规模 $G_{fail}$
- 显存可行性口径 $type_{mem}$

### 4.4 单位约定与辅助映射

除最终展示外，全文中间计算统一采用：

- 显存相关量的单位为 $byte$
- 带宽相关量的单位为 $byte/s$
- token 长度与吞吐相关量的单位为 $token$、$token/s$
- 计算量与算力相关量的单位为 $FLOP$、$FLOP/s$
- 时间单位为 $s$

辅助函数只用于表达**确定映射**，不吸收任何经验假设：

- $b_w$：每个权重参数占用的字节数，由推理精度决定
- $b_c$：每个 cache 元素占用的字节数，由 cache 精度决定
- $F_{peak}$：GPU 在当前推理精度下的峰值算力

### 4.5 主要记号

#### 业务侧记号

- $C_{avg}$：平均活跃并发
- $C_{peak}$：峰值活跃并发
- $TTFT_{avg}^{target}$：平均首字延迟目标
- $TTFT_{p95}^{target}$：P95 首字延迟目标
- $E2E_{avg}^{target}$：平均单次总时延目标
- $E2E_{p95}^{target}$：P95 单次总时延目标

#### 长度侧记号

- $S_{in,avg}$：平均输入长度
- $S_{out,avg}$：平均输出长度
- $S_{in,p95}$：P95 输入长度
- $S_{out,p95}$：P95 输出长度
- $S_{avg}$：平均总长度
- $S_{p95}$：P95 总长度

#### 阶段活跃比例

- $r_{dec,avg}$：平均窗口内处于 decode 阶段的请求比例
- $r_{dec,peak}$：峰值窗口内处于 decode 阶段的请求比例
- $r_{pre,peak}$：峰值窗口内处于 prefill 阶段的请求比例

#### 模型与硬件侧记号

- $P_{total}$：模型总参数量
- $P_{act}$：每个 token 实际激活参数量
- $L$：层数
- $H$：隐藏维度
- $H_{kv}$：KV 头数
- $d_{head}$：单个 attention head 的维度
- $d_{latent}$：MLA latent cache 维度
- $V_{gpu}$：单卡显存容量
- $B_{mem}$：单卡峰值显存带宽
- $\eta_{bw}$：有效带宽利用率
- $\eta_{cmp}$：有效算力利用率
- $\eta_{vram}$：可用显存比例
- $\eta_{inst}$：多卡实例扩展效率

#### 显存系数

- $\alpha_w$：权重附加显存系数
- $\alpha_r$：运行时固定显存系数

#### Prefill 代理参数

- $k_{pre}$：Prefill 相对 Decode 的代理倍率

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

建议至少提供以下内容：

- 平均活跃并发 $C_{avg}$
- 峰值活跃并发 $C_{peak}$
- 平均首字延迟目标 $TTFT_{avg}^{target}$
- P95 首字延迟目标 $TTFT_{p95}^{target}$
- 平均单次总时延目标 $E2E_{avg}^{target}$
- P95 单次总时延目标 $E2E_{p95}^{target}$
- 请求长度分布 `request_shapes`
- 平均 decode-active 比例 $r_{dec,avg}$
- 峰值 decode-active 比例 $r_{dec,peak}$
- 峰值 prefill-active 比例 $r_{pre,peak}$
- P95 总长度 $S_{p95}$（若业务侧可直接提供，建议显式给出）

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

- 所有 $w_i>0$
- $\sum_i w_i=1$
- 每个 shape 至少包含名称、权重、平均输入长度、P95 输入长度、平均输出长度、P95 输出长度

#### 5.2.2 Decode-active 比例（必填）

定义域：

$$
0<r_{dec,avg}\le 1
$$

$$
0<r_{dec,peak}\le 1
$$

推荐原则：

- 优先由日志窗口统计得到
- 至少拆分平均与峰值两个口径
- 不建议继续使用单一比例同时支配平均与峰值计算

#### 5.2.3 Prefill-active 比例（选填，建议显式提供）

定义域：

$$
0<r_{pre,peak}\le 1
$$

用途：

- 用于把峰值活跃并发折算为同一时间真正竞争 prefill 资源的请求比例
- 若无法提供，可取保守缺省值

$$
r_{pre,peak}=1.0
$$

说明：该缺省值会把 prefill 总需求推向上界，通常偏保守。

#### 5.2.4 P95 总长度（选填，强烈建议）

用途：

- 若业务侧可直接给出真实的 P95 总长度，应优先使用
- 若未提供，则采用保守近似 $S_{p95}=S_{in,p95}+S_{out,p95}$

### 5.3 模型输入（必填）

建议至少提供：

- 模型名称（可选）
- 架构类型：Dense 或 MoE
- Attention 类型：MHA / GQA / MQA / MLA / Hybrid
- 总参数量 $P_{total}$
- 每 token 激活参数量 $P_{act}$（MoE 必填）
- 层数 $L$
- 隐藏维度 $H$
- Query 头数
- KV 头数 $H_{kv}$
- 单 head 维度 $d_{head}$
- MLA latent cache 维度 $d_{latent}$
- 最大上下文窗口
- 权重推理精度
- KV cache 精度
- 每 token 每层额外 cache 开销
- 若可直接给出，则直接输入“每 token 每层 cache 字节数”

对 MoE，$P_{act}$ 的口径必须统一为：

- 所有每 token 必经的 shared / dense 参数
- router 及相关公共路径的近似开销
- top-k 被激活 expert 参数总和

不得只填“被选中的 expert 参数”而遗漏 shared 路径，否则会系统性低估吞吐约束。

### 5.4 GPU 选择输入（必填）

建议至少提供：

- GPU 名称
- 单卡显存容量 $V_{gpu}$
- 峰值显存带宽 $B_{mem}$
- FP16 / BF16 / FP8 / INT8 / INT4 峰值算力
- 有效带宽利用率 $\eta_{bw}$
- 有效算力利用率 $\eta_{cmp}$
- 可用显存比例 $\eta_{vram}$

推荐默认值（仅无实测时作为采购前基线）：

- $\eta_{bw}=0.70$
- $\eta_{cmp}=0.35$
- $\eta_{vram}=0.90$

### 5.5 推理框架 / 部署档位输入（必填）

建议至少提供：

- 推理框架及版本
- 权重附加显存系数 $\alpha_w$
- 运行时固定显存系数 $\alpha_r$
- 多卡实例扩展效率 $\eta_{inst}$
- Prefill 代理倍率 $k_{pre}$

推荐默认值：

- $\alpha_w=0.05$
- $\alpha_r=0.10$
- 单卡实例时 $\eta_{inst}=1.00$
- 多卡实例且无实测值时，可取 $\eta_{inst}=0.85$
- 无实测值时，可取 $k_{pre}=1.5$

说明：

- $\alpha_w$ 只吸收与权重驻留直接相关的附加显存，例如量化元数据、布局与对齐开销。
- $\alpha_r$ 只吸收运行时长期占用的固定显存，例如 workspace、allocator 预留、运行时 buffer 等。
- $\eta_{vram}$ 只吸收 OOM 安全水位、波动预留与运营安全边界，不应与前两者重复计入。
- $\eta_{inst}$ 仅在 $g>1$ 时参与吞吐计算，用于吸收跨卡通信与并行切分导致的非线性扩展损失。
- $k_{pre}$ 不是物理常数，而是用于采购前估算的代理参数；若已有实测 prefill 吞吐，应直接替换代理模型。

#### 5.5.1 采购前推荐档位

若缺少实测值，建议至少同时评估保守档与中性档，避免把单点默认值误当真值。

| 档位 | $\eta_{bw}$ | $\eta_{cmp}$ | $\alpha_w$ | $\alpha_r$ | $\eta_{inst}$（多卡） | $k_{pre}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 保守 | 0.60 | 0.25 | 0.08 | 0.15 | 0.75 | 1.0 |
| 中性 | 0.70 | 0.35 | 0.05 | 0.10 | 0.85 | 1.5 |
| 乐观 | 0.80 | 0.45 | 0.03 | 0.06 | 0.90 | 2.0 |

说明：

- 单卡实例固定取 $\eta_{inst}=1.0$
- 采购前正式输出至少应给出保守档与中性档两组结果
- 若后续拿到框架实测或压测结果，应优先以实测值回填这些参数

### 5.6 HA 输入（必填）

建议至少提供：

- HA 模式：无 HA / N+1 / 容忍故障单元 / 容忍整节点
- 需容忍的故障单元数
- 单节点 GPU 数（若按节点维度做 HA，建议提供）

---

## 6. 输入校验与基础派生

### 6.1 合法性校验

至少应校验：

- $C_{avg}>0$
- $C_{peak}\ge C_{avg}$
- $TTFT_{avg}^{target}>0$
- $TTFT_{p95}^{target}\ge TTFT_{avg}^{target}$
- $E2E_{avg}^{target}>TTFT_{avg}^{target}$
- $E2E_{p95}^{target}>TTFT_{p95}^{target}$
- $0<r_{dec,avg}\le 1$
- $0<r_{dec,peak}\le 1$
- $0<r_{pre,peak}\le 1$
- $0<\eta_{vram}\le 1$
- $0<\eta_{bw}\le 1$
- $0<\eta_{cmp}\le 1$
- $0<\eta_{inst}\le 1$
- $k_{pre}>0$
- $\sum_i w_i=1$

### 6.2 从 `request_shapes` 派生全局长度统计

全局平均输入长度：

$$
S_{in,avg}=\sum_i w_i\,S_{in,avg,i}
$$

全局平均输出长度：

$$
S_{out,avg}=\sum_i w_i\,S_{out,avg,i}
$$

全局 P95 输入与输出长度：

- 若有原始离散桶分布，可按累计分位近似求取
- 若仅有少量离散 shape，工程上可取“高位主导 shape”的保守近似

记最终派生结果为：

- $S_{in,p95}$
- $S_{out,p95}$

### 6.3 总长度 P95

优先顺序：

1. 若业务侧单独提供真实的 P95 总长度，优先使用。
2. 若未提供，则保守近似：

$$
S_{p95}=S_{in,p95}+S_{out,p95}
$$

说明：该近似通常大于真实总长度 P95，因此偏保守。

### 6.4 基础总长度

平均总长度：

$$
S_{avg}=S_{in,avg}+S_{out,avg}
$$

P95 总长度：

$$
S_{p95}=\begin{cases}
S_{p95}^{given}, & \text{若业务侧已提供}\\
S_{in,p95}+S_{out,p95}, & \text{否则采用保守近似}
\end{cases}
$$

---

## 7. 从体验目标到系统需求

### 7.1 平均阶段预算

平均 decode 预算：

$$
T_{dec,avg}=E2E_{avg}^{target}-TTFT_{avg}^{target}
$$

默认平均 prefill 预算份额：

$$
\phi_{pre,avg}=0.8
$$

平均单请求 prefill 速率需求：

$$
R_{pre,req}^{avg}=\frac{S_{in,avg}}{TTFT_{avg}^{target}\times \phi_{pre,avg}}
$$

平均单请求 decode 生成速率需求：

$$
R_{dec,req}^{avg}=\frac{S_{out,avg}}{T_{dec,avg}}
$$

### 7.2 P95 阶段预算

P95 decode 预算采用工程预算近似：

$$
T_{dec,p95}=E2E_{p95}^{target}-TTFT_{p95}^{target}
$$

默认 P95 prefill 预算份额：

$$
\phi_{pre,p95}=0.8
$$

P95 单请求 prefill 速率需求：

$$
R_{pre,req}^{p95}=\frac{S_{in,p95}}{TTFT_{p95}^{target}\times \phi_{pre,p95}}
$$

P95 单请求 decode 生成速率需求：

$$
R_{dec,req}^{p95}=\frac{S_{out,p95}}{T_{dec,p95}}
$$

说明：

- $\phi_{pre,avg}$ 与 $\phi_{pre,p95}$ 是将 TTFT 预算映射为 prefill 需求的工程缺省值。
- 若业务侧或框架侧已有更合适的预算拆分口径，应优先替换默认值。

### 7.3 系统总吞吐需求

平均 decode 总吞吐需求：

$$
TPS_{dec,target}^{avg}=C_{avg}\times r_{dec,avg}\times R_{dec,req}^{avg}
$$

峰值 decode 总吞吐需求：

$$
TPS_{dec,target}^{peak}=C_{peak}\times r_{dec,peak}\times R_{dec,req}^{p95}
$$

峰值 prefill 总吞吐需求：

$$
TPS_{pre,target}^{peak}=C_{peak}\times r_{pre,peak}\times R_{pre,req}^{p95}
$$

说明：

- Prefill 与 Decode 在系统总需求推导中采用对称口径。
- 若无法提供 $r_{pre,peak}$，可取保守缺省值 $1.0$。
- 更严格的做法应使用请求到达率、burst 窗口与阶段驻留时间来建模；在缺少这些数据时，本文采用 active-ratio 近似。

---

## 8. 单实例显存模型

### 8.1 权重显存

原始权重字节数：

$$
M_w^{raw}=P_{total}\times b_w
$$

考虑权重格式、量化元数据与布局附加开销后的权重显存：

$$
M_w=M_w^{raw}(1+\alpha_w)
$$

说明：

- $\alpha_w$ 应保持为小比例系数，默认用于采购前预估。
- 若框架实测可直接给出模型加载后的稳定驻留显存，应优先以实测值替换该系数。

### 8.2 每个 token、每层的 cache 字节数

记每个 token 在单层上占用的 cache 字节数为 $e_{cache}$。

若可直接给出该值，则直接使用：

$$
e_{cache}=e_{cache}^{given}
$$

否则可按 attention 结构近似计算：

- MHA：

$$
e_{cache}=2H\,b_c+e_{aux}
$$

- GQA：

$$
e_{cache}=2H_{kv}d_{head}\,b_c+e_{aux}
$$

- MQA：

$$
e_{cache}=2d_{head}\,b_c+e_{aux}
$$

- MLA：

$$
e_{cache}=d_{latent}\,b_c+e_{aux}
$$

- Hybrid Attention：

若各层结构不同，则对每层分别计算 $e_{cache}^{(l)}$，再按层求和；若无法提供逐层结构信息，则应直接输入 $e_{cache}$。

### 8.3 单请求 cache 显存

对给定序列长度 $S$：

$$
M_{cache,req}(S)=L\times S\times e_{cache}
$$

若为 Hybrid Attention 且使用逐层求和，则：

$$
M_{cache,req}(S)=S\sum_{l=1}^{L} e_{cache}^{(l)}
$$

### 8.4 运行时固定显存

$$
M_r=\alpha_r M_w
$$

说明：这是采购前常用的经验代理，不表示运行时固定显存与权重之间存在严格物理线性关系。

### 8.5 单实例总显存组成

单实例显存由三部分组成：

$$
M_{inst}=M_w+M_r+M_{cache,active}
$$

其中：

- $M_w$：权重静态驻留显存
- $M_r$：运行时长期固定显存
- $M_{cache,active}$：活跃请求占用的 cache 显存

采购前 sizing 中，后续通过单实例 cache 容量上限反推可承载请求数，而不在此处预设单实例请求数。

### 8.6 单实例显存可行性

#### 8.6.1 单卡完整装载场景

当 $g=1$ 时，单实例可用显存为：

$$
M_{use}=V_{gpu}\times \eta_{vram}
$$

若：

$$
M_w+M_r\ge M_{use}
$$

则单卡完整装载不可行。

单实例可承载 cache 容量：

$$
M_{cache,cap}=M_{use}-M_w-M_r
$$

该场景下：

- $mode_{deploy}=$ 单卡完整装载
- $type_{mem}=$ 严格可行

#### 8.6.2 多卡分片场景

当 $g>1$ 时，本文在前期 sizing 中采用**实例级 pooled / sharded 可行性上界**近似：

$$
M_{use}=g\times V_{gpu}\times \eta_{vram}
$$

若：

$$
M_w+M_r\ge M_{use}
$$

则当前 $g$ 不可行。

单实例可承载 cache 容量：

$$
M_{cache,cap}=M_{use}-M_w-M_r
$$

该场景下：

- $mode_{deploy}=$ 多卡分片装载
- $type_{mem}=$ 分片上界可行

说明：此处更适合作为多卡 / sharded 方案的前期可行性判断，而不应被解读为 replicated 多卡实例的严格显存证明。

---

## 9. 单实例吞吐模型

### 9.1 激活参数量定义

令：

- Dense：$P_{act}=P_{total}$
- MoE：$P_{act}$ 为每 token 实际激活参数量

### 9.2 Decode 单实例吞吐

单个 token 的近似访存量：

$$
b_{dec}=P_{act}\times b_w
$$

单个 token 的近似计算量：

$$
f_{dec}=2P_{act}
$$

内存受限近似：

$$
TPS_{dec,mem}^{gpu}=\frac{B_{mem}\eta_{bw}}{b_{dec}}
$$

算力受限近似：

$$
TPS_{dec,cmp}^{gpu}=\frac{F_{peak}\eta_{cmp}}{f_{dec}}
$$

单 GPU decode 吞吐：

$$
TPS_{dec}^{gpu}=\min\left(TPS_{dec,mem}^{gpu},\,TPS_{dec,cmp}^{gpu}\right)
$$

若 $g=1$，取 $\eta_{inst}=1.0$。

单实例 decode 吞吐：

$$
TPS_{dec}^{inst}=g\times TPS_{dec}^{gpu}\times \eta_{inst}
$$

说明：

- 上式是前期 sizing 近似，attention、norm、rope、softmax、MLA 等额外 FLOPs 与 IO 被吸收到有效利用率与实例扩展效率中。
- 多卡场景默认各卡承担近似均衡的 shard 工作量，且跨卡同步开销已被 $\eta_{inst}$ 吸收；不适用于明显异构并行或流水并行瓶颈主导的场景。
- 若有实测值，应优先替换。
- 对大模型在线推理，decode 常更偏 memory-bound，因此 $\eta_{bw}$ 往往比 $\eta_{cmp}$ 更敏感。

### 9.3 Prefill 单实例吞吐

#### 9.3.1 定位

Prefill 与 Decode 的计算 / 访存机理并不相同。尤其是：

- Prefill 会显著受序列长度影响。
- Attention 计算模式与 Decode 不同。
- Prefill 往往具有更高的 GEMM 利用率与更强的计算密度。

因此，本文**不把 Prefill 写成与 Decode 同构的严格推导公式**，而是采用采购前可落地的**代理模型**。

#### 9.3.2 基线代理

先以 Decode 单 GPU 吞吐作为基线：

$$
TPS_{pre}^{gpu}=k_{pre}\times TPS_{dec}^{gpu}
$$

再得到单实例 Prefill 代理吞吐：

$$
TPS_{pre}^{inst}=g\times TPS_{pre}^{gpu}\times \eta_{inst}
$$

说明：

- $k_{pre}$ 用于表达“Prefill 相比 Decode 的工程代理倍率”。
- 它不是物理常数，也不意味着 Prefill 真正按该倍率线性成立。
- 若已有框架实测 Prefill 吞吐，应直接用实测值替换本节代理模型。
- 若后续需要更高精度，可改为按序列长度区间建立 $TPS_{pre}(S)$ 分段模型，或直接按 benchmark 表插值。

---

## 10. 部署方案求解

### 10.1 总原则

部署模式不是用户输入，而是测算过程中的自动推断结果。

部署求解接收业务目标、模型参数、GPU 规格、框架档位与 HA 目标，并自动求解最小可行部署方案。

### 10.2 候选实例能力计算

从 $g=1$ 开始，依次枚举候选实例规模；若单卡不可行，再继续枚举：

$$
g\in\{2,4,8,\ldots\}
$$

对每个候选 $g$，只做三类计算：

1. **显存可行性计算**：判断是否可容纳 $M_w+M_r$，并求出 $M_{cache,cap}$。
2. **单实例能力计算**：估算 $TPS_{pre}^{inst}$ 与 $TPS_{dec}^{inst}$。
3. **故障单元归属**：确定该候选下的 $G_{fail}$。

注意：

- 这一阶段不判断“单实例是否满足业务总需求”。
- 业务总需求由后续**实例数求解**完成。
- 因此这里不存在“先知道每实例分担多少业务，再判断实例是否可行”的循环依赖。

### 10.3 候选方案类型判定

若候选为单卡且显存可行，则：

- $g=1$
- $is_{shard}=false$
- $mode_{deploy}=$ 单卡完整装载
- $type_{mem}=$ 严格可行

若候选为多卡且显存可行，则：

- $is_{shard}=true$
- $mode_{deploy}=$ 多卡分片装载
- $type_{mem}=$ 分片上界可行

### 10.4 故障单元推导

- 若单卡独立实例：通常 $G_{fail}=1$
- 若多卡强耦合实例：通常 $G_{fail}=g$
- 若按整节点绑定故障：可取 $G_{fail}=G_{node}$

### 10.5 候选方案选择原则

对所有可行候选，先分别求出其：

- $N_{inst}^{mem}$
- $N_{inst}^{pre}$
- $N_{inst}^{dec}$
- $N_{inst}^{biz}$
- $G_{biz}$

再选择业务所需 GPU 数最小的方案；若多方案 $G_{biz}$ 相同，则优先：

1. 单卡完整装载方案
2. 故障单元更小的方案
3. 具有更高吞吐冗余的方案

---

## 11. 业务所需实例数与 GPU 数

### 11.1 Decode 约束实例数

$$
N_{inst}^{dec}=\left\lceil\frac{TPS_{dec,target}^{peak}}{TPS_{dec}^{inst}}\right\rceil
$$

### 11.2 Prefill 约束实例数

$$
N_{inst}^{pre}=\left\lceil\frac{TPS_{pre,target}^{peak}}{TPS_{pre}^{inst}}\right\rceil
$$

### 11.3 显存约束实例数

先计算单实例在峰值保守场景下可承载的请求数上界：

$$
N_{req}^{mem}=\left\lfloor\frac{M_{cache,cap}}{M_{cache,req}(S_{p95})}\right\rfloor
$$

若：

$$
N_{req}^{mem}\le 0
$$

则当前实例方案不可行。

显存约束实例数：

$$
N_{inst}^{mem}=\left\lceil\frac{C_{peak}}{N_{req}^{mem}}\right\rceil
$$

说明：

- 这里的 $N_{req}^{mem}$ 是“每个活跃请求都按 P95 总长度占满 cache”的保守可承载请求数上界。
- 它不是线上真实可承载并发的严格最大值。

### 11.4 业务实例数与业务 GPU 数

$$
N_{inst}^{biz}=\max\left(N_{inst}^{dec},N_{inst}^{pre},N_{inst}^{mem}\right)
$$

$$
G_{biz}=N_{inst}^{biz}\times g
$$

说明：

- 这是采购前阶段的业务最小 GPU 数。
- 它已经综合了显存、prefill、decode 三条主约束。

---

## 12. 高可用展开

### 12.1 HA 后实例数

- 无 HA：

$$
N_{inst}^{final}=N_{inst}^{biz}
$$

- 容忍故障单元：

$$
N_{inst}^{final}=N_{inst}^{biz}+\left\lceil\frac{N_{fail}\times G_{fail}}{g}\right\rceil
$$

其中 $N_{fail}$ 为需容忍的故障单元数。

- 容忍整节点：需按节点粒度折算。
- N+1：可按至少增加 1 个实例处理。

### 12.2 最终 GPU 数

$$
G_{final}=N_{inst}^{final}\times g
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
TPS_{dec}^{cluster}=N_{inst}^{biz}\times TPS_{dec}^{inst}
$$

业务集群 prefill 吞吐：

$$
TPS_{pre}^{cluster}=N_{inst}^{biz}\times TPS_{pre}^{inst}
$$

### 13.2 可持续并发能力

先定义 decode-active 可持续并发：

$$
C_{dec}^{sus}=\frac{TPS_{dec}^{cluster}}{R_{dec,req}^{p95}}
$$

再折算总活跃并发：

$$
C_{total,dec}^{sus}=\frac{C_{dec}^{sus}}{r_{dec,peak}}
$$

Prefill-active 可持续并发：

$$
C_{pre}^{sus}=\frac{TPS_{pre}^{cluster}}{R_{pre,req}^{p95}}
$$

再折算总活跃并发：

$$
C_{total,pre}^{sus}=\frac{C_{pre}^{sus}}{r_{pre,peak}}
$$

最终可持续并发能力：

$$
C^{sus}=\min\left(C_{total,dec}^{sus},C_{total,pre}^{sus}\right)
$$

### 13.3 单请求生成速度

$$
V_{gen}\approx\frac{TPS_{dec}^{cluster}}{C_{peak}\times r_{dec,peak}}
$$

### 13.4 平均时延近似

平均 TTFT 近似：

$$
TTFT_{avg}^{est}\approx\frac{S_{in,avg}}{TPS_{pre}^{cluster}/\left(C_{avg}\times r_{pre,peak}\right)}
$$

平均 E2E 近似：

$$
E2E_{avg}^{est}\approx \frac{S_{in,avg}}{TPS_{pre}^{cluster}/\left(C_{avg}\times r_{pre,peak}\right)}+\frac{S_{out,avg}}{TPS_{dec}^{cluster}/\left(C_{avg}\times r_{dec,avg}\right)}
$$

说明：

- 若只有峰值 prefill-active 比例而缺少平均口径，平均时延反推中可暂用 $r_{pre,peak}$ 代替，结果偏保守。
- 该式是工程近似与 sanity check，不是队列论严格推导。

### 13.5 保守 P95 时延近似

$$
TTFT_{p95}^{upper}\approx\frac{S_{in,p95}}{TPS_{pre}^{cluster}/\left(C_{peak}\times r_{pre,peak}\right)}
$$

$$
E2E_{p95}^{upper}\approx \frac{S_{in,p95}}{TPS_{pre}^{cluster}/\left(C_{peak}\times r_{pre,peak}\right)}+\frac{S_{out,p95}}{TPS_{dec}^{cluster}/\left(C_{peak}\times r_{dec,peak}\right)}
$$

### 13.6 每日产能

$$
Q_{dec}^{day}=TPS_{dec}^{cluster}\times 86400
$$

$$
Q_{pre}^{day}=TPS_{pre}^{cluster}\times 86400
$$

---

## 14. 输出项建议

### 14.1 输入摘要

- 并发、TTFT、E2E 目标
- `request_shapes` 摘要
- Decode-active 比例与 Prefill-active 比例
- 模型 / GPU / 框架 / HA 选择
- 是否使用默认值及其清单

### 14.2 部署求解结果

- 单实例 GPU 数 $g$
- 是否分片 $is_{shard}$
- 部署模式 $mode_{deploy}$
- 显存可行性口径 $type_{mem}$
- 故障单元规模 $G_{fail}$
- 单实例 Prefill 吞吐 $TPS_{pre}^{inst}$
- 单实例 Decode 吞吐 $TPS_{dec}^{inst}$
- $N_{inst}^{mem}$ / $N_{inst}^{pre}$ / $N_{inst}^{dec}$

### 14.3 资源规模结果

- $N_{inst}^{biz}$
- $G_{biz}$
- $N_{inst}^{final}$
- $G_{final}$
- 若使用档位输入，建议同时输出保守 / 中性 / 乐观三档结果

### 14.4 效果反推结果

- $TPS_{pre}^{cluster}$
- $TPS_{dec}^{cluster}$
- $C^{sus}$
- $V_{gen}$
- $TTFT_{avg}^{est}$ / $TTFT_{p95}^{upper}$
- $E2E_{avg}^{est}$ / $E2E_{p95}^{upper}$
- $Q_{dec}^{day}$
- $Q_{pre}^{day}$

### 14.5 结果展示单位建议

为避免中间推导混入单位换算，建议仅在展示层做如下换算：

- $byte\rightarrow GiB$

$$
X_{GiB}=\frac{X_{byte}}{2^{30}}
$$

- $byte/s\rightarrow GiB/s$

$$
X_{GiB/s}=\frac{X_{byte/s}}{2^{30}}
$$

- $FLOP/s\rightarrow TFLOP/s$

$$
X_{TFLOPS}=\frac{X_{FLOP/s}}{10^{12}}
$$

显存与带宽展示默认采用二进制单位 $GiB$ / $GiB/s$；核心公式内部仍统一使用 $byte$ 与 $byte/s$，不在中间过程混入任何展示单位换算。
