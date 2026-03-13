---
name: server-dev
description: Python 计算逻辑开发技能。用于修改 `main.py`、新增 GPU sizing 公式、调整输入输出参数、修正文档中的计算说明或同步示例时触发；先确认假设、单位和 source-of-truth，再同步更新代码与文档，避免公式、参数和示例结果失配。
---

# Python 计算逻辑开发

## 读取顺序

1. `docs/llm_gpu_inference_calc.md`
2. `main.py`
3. `../_shared/references/change-completeness-checklist.md`
4. `../_shared/references/quality-gates.md`

## 原则

- 代码是可执行真源，文档是解释真源；两者必须同步。
- 先确认公式、单位、默认值和适用边界，再改实现。
- 改动优先消除重复或矛盾说明，不保留旧公式和旧参数解释。
- 任何新输入项都必须补默认值、校验规则和输出影响。
- 数学近似可以存在，但必须在代码或文档中写清近似前提。

## 实施流程

1. 识别改动类型：`formula`、`parameter`、`output`、`docs-only`。
2. 明确 source-of-truth：计算逻辑落在 `main.py`，说明文字落在 `docs/*`。
3. 按 `change-completeness-checklist.md` 列出受影响项：数据类、计算函数、打印输出、文档公式、示例参数。
4. 先改根因位置，再删掉过期常量、注释、段落和示例。
5. 若输出含新增字段，同步更新结果结构和文档中的字段定义。
6. 完成后执行最小验证，并记录仍未覆盖的假设。

## 硬门禁

- 不允许只改文档不改代码，或只改代码不改文档，导致定义分叉。
- 不允许引入隐式单位转换；GiB、GB、token/s、ms/token 必须写清。
- 不允许保留与当前实现冲突的旧示例、旧默认值或旧公式。
- 不允许新增未校验的输入字段。
