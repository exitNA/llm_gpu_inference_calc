---
name: feature-doc-driven-dev
description: 文档驱动开发技能。用于新增或修改 GPU sizing 方法说明、方案文档、README、示例参数或输出说明时触发；先统一术语、公式、输入输出定义和示例，再落代码或改实现，确保 `docs/*` 与 `main.py` 一致。
---

# 文档驱动开发

## 读取顺序

1. `docs/llm_gpu_inference_calc.md`
2. `README.md`
3. `main.py`
4. `references/workflow.md`
5. `references/doc-templates.md`

## 核心原则

- 先写清定义，再写实现细节。
- 同一概念只保留一套主表述，重复说明要么删除，要么引用主文档。
- 文档里的参数、默认值、公式、单位和代码保持一致。
- 方案文档关注方法与假设；代码文档关注可执行输入输出；不要互相复制大段内容。

## 执行流程

1. 明确主文档：方法说明优先放 `docs/llm_gpu_inference_calc.md`，使用说明优先放 `README.md`。
2. 建立改动面矩阵：术语、公式、参数表、示例、代码输出。
3. 先更新主文档，再同步次级文档和代码。
4. 如果代码行为变了，补齐文档中的输入输出定义和示例。
5. 删除已经失效的旧段落、旧标题和重复解释。
6. 交付前核对文档与代码的一致性。

## 工具与资源

- 流程约束：`references/workflow.md`
- 文档模板：`references/doc-templates.md`
- 完整性清单：`../_shared/references/change-completeness-checklist.md`
