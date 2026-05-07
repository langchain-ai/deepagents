# 电商助手 - 技能系统

## 概述

本项目集成了 DeepAgents 的技能系统，完全兼容 Claude 的技能模式。技能系统允许 Agent 使用专业的、结构化的工作流来完成特定任务。

## 项目结构

```
skills/
├── base/                          # 基础技能（通用技能）
└── ecommerce/                     # 电商专用技能
    ├── product-publish/
    │   └── SKILL.md              # 商品发布技能
    ├── good-review/
    │   └── SKILL.md              # 好评管理技能
    └── data-collection/
        └── SKILL.md              # 数据采集技能
```

## 技能文件格式

每个技能目录包含一个 `SKILL.md` 文件，格式如下：

```markdown
---
name: skill-name
description: 技能描述
compatibility: Python 3.10+, Playwright
allowed-tools: tool1, tool2
license: MIT
metadata:
  category: category-name
  platform: platform1, platform2
  last-updated: 2026-01-01
---

# 技能名称

## 何时使用
说明什么时候使用这个技能

## 执行步骤
详细的步骤说明

## 相关技能
列出相关的技能
```

## 可用技能

### 1. product-publish (商品发布)
- **描述**: 在各大电商平台自动发布商品
- **平台**: 抖音、拼多多、淘宝
- **工具**: navigate, click_element, input_text, take_screenshot

### 2. good-review (好评管理)
- **描述**: 管理电商平台的好评，包括自动回复、追评和好评分析
- **平台**: 抖音、拼多多、淘宝
- **工具**: navigate, click_element, input_text, take_screenshot

### 3. data-collection (数据采集)
- **描述**: 采集电商平台的订单、销售和推广数据
- **平台**: 抖音、拼多多、淘宝
- **工具**: navigate, click_element, take_screenshot

## 在代码中使用

### Agent 集成

在 `backend/agent/core.py` 中已经集成了技能系统：

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    system_prompt=system_prompt,
    tools=tools,
    skills=[skills_dir],  # 技能目录
    backend=FilesystemBackend(root_dir=project_root)
)
```

### 系统提示词集成

Agent 的系统提示词中已经包含了使用技能的指导：
1. 首先检查是否有相关技能
2. 阅读技能的完整说明
3. 按照技能指导完成任务
4. 使用技能推荐的工具

## 测试

运行技能系统测试：

```bash
cd /workspace/examples/ecommerce_agent
python test_skills.py
```

## 添加新技能

1. 在 `skills/` 目录下创建新的技能目录，目录名与技能名一致
2. 创建 `SKILL.md` 文件，按照格式填写
3. 可选：添加辅助脚本或工具
4. 在 Agent 的系统提示词中更新技能列表

## 技能系统特点

1. **渐进式披露**: 先显示技能元数据，需要时再读取完整内容
2. **多源支持**: 可以从多个目录加载技能
3. **工具权限**: 每个技能可以指定允许使用的工具
4. **元数据**: 包含描述、兼容性、许可等信息
5. **版本管理**: 支持技能版本和更新记录

## 参考资料

- [DeepAgents 技能系统文档](file:///workspace/libs/deepagents/deepagents/middleware/skills.py)
- [Agent Skills 规范](https://agentskills.io/specification)
