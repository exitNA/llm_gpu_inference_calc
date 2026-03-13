---
name: code-review
description: 本仓库代码与文档评审技能。用于用户请求“review/代码审查/文档审查/一致性检查/风险评估”时触发；优先识别公式错误、代码与文档不一致、边界条件缺陷、无效参数、输出回归和验证缺口，并按严重级别输出 findings。
---

# 代码与文档评审

## 读取顺序

1. 变更文件
2. `main.py`
3. `docs/llm_gpu_inference_calc.md`
4. `references/severity-rubric.md`
5. `references/review-output-template.md`

## 重点

- 正确性高于风格。
- 文档和代码的矛盾视为真实风险，不当作纯文案问题。
- 若用户同时要求修复，先给 findings，再进入改动。

## 审查流程

1. 确认审查范围：是公式、参数、输出格式还是文档结构改动。
2. 先核对定义一致性：变量名、单位、默认值、公式和示例是否一致。
3. 再检查行为风险：边界条件、非法输入、舍入方式、P95 取值逻辑、高可用分支。
4. 检查验证充分性：是否运行了最小编译检查或示例计算。
5. 形成 findings：按严重级别排序，给出位置、影响和修复方向。
6. 若无阻塞问题，明确写出剩余风险或未验证项。

## 输出要求

- 严格使用 `references/review-output-template.md` 的结构。
- findings 必须放在最前，按严重级别排序。
- 每条 finding 必须包含：`Severity`、`Location`、`Issue`、`Impact`、`Fix direction`。
- `Location` 必须使用可点击文件路径并带行号。
- 若运行过验证命令，需写明命令与结论；未运行则明确说明原因。
