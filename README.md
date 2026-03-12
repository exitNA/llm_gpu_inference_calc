# GPU Analysis

一个基于 Gradio 的单模型推理 GPU 资源估算器，用于交互式评估显存、吞吐和高可用采购规模。

## 运行

```bash
uv sync
uv run python main.py
```

默认启动 Gradio 页面，监听 `0.0.0.0:7860`。可选参数：

```bash
uv run python main.py --host 127.0.0.1 --port 7860
uv run python main.py --dry-run
```

`--dry-run` 会使用内置示例参数执行一次估算并直接输出文本结果，不启动 Web UI。

## 交互输入

- 模型参数：参数量、层数、隐藏维度、权重精度、KV 精度、显存开销
- 流量参数：并发、decode/prefill 吞吐目标、请求长度分布
- GPU 参数：单卡显存、可用比例、带宽、实测吞吐、单价
- 高可用参数：模式、副本数、故障冗余、是否双区部署

## 输出结果

- 核心指标表：平均/P95 长度、权重显存、KV Cache、总显存、吞吐估算
- 请求分布与 KV Cache 明细
- 原始 JSON 结果，便于继续集成或导出
