---
name: server-code-style
description: Python 代码规范技能。用于编写、重构、评审本仓库的 Python 脚本、计算函数或命令行示例时触发；约束命名、类型、异常处理、函数职责和公式可读性，适用于 `main.py` 及后续工具脚本。
---

# Python 代码规范

## 范围

- 本技能只负责 Python 实现质量。
- 涉及公式、参数含义、示例和文档同步时，组合 `../server-dev/SKILL.md` 或 `../feature-doc-driven-dev/SKILL.md`。

## 规则

- 模块、函数、变量使用 `snake_case`；数据结构类型使用 `PascalCase`；常量使用 `UPPER_SNAKE_CASE`。
- 所有公开函数写参数和返回类型；只有边界层允许少量 `Any`。
- 优先使用 `dataclass` 或清晰的 `dict` 结构承载结果，避免混合返回值。
- 一个函数只做一件事：校验、计算、聚合、打印尽量拆开。
- 公式相关注释只写假设、单位和近似来源，不复述代码。
- 禁止裸 `except`、静默吞错和隐式回退；输入非法时抛出明确异常。
- 避免魔法数字；精度、比例、默认参数和单位转换应集中定义。
- 输出和示例允许使用中文，但标识符和接口键保持英文，便于后续复用。

## 共享基线

- 通用约束以 `../_shared/references/common-code-baseline.md` 为准。
- 交付前检查以 `../_shared/references/quality-gates.md` 为准。
