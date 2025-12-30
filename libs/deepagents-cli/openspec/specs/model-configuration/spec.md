# model-configuration Specification

## Purpose
TBD - created by archiving change add-flexible-model-providers. Update Purpose after archive.
## Requirements
### Requirement: 自定义 API 端点配置

系统 **必须 (SHALL)** 支持通过环境变量配置自定义 API 端点，以接入兼容 OpenAI 或 Anthropic API 格式的第三方服务。

#### Scenario: 使用环境变量配置 OpenAI 兼容端点

- **GIVEN** 用户设置了环境变量 `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- **AND** 用户设置了环境变量 `OPENAI_API_KEY=sk-xxx`
- **WHEN** 用户运行 `deepagents --model deepseek-chat`
- **THEN** 系统使用 `https://api.deepseek.com/v1` 作为 API 端点
- **AND** 系统成功初始化 ChatOpenAI 模型

#### Scenario: 使用环境变量配置 Anthropic 兼容端点

- **GIVEN** 用户设置了环境变量 `ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/paas/v4`
- **AND** 用户设置了环境变量 `ANTHROPIC_API_KEY=xxx`
- **WHEN** 用户运行 `deepagents --model glm-4`
- **THEN** 系统使用 `https://open.bigmodel.cn/api/paas/v4` 作为 API 端点
- **AND** 系统成功初始化 ChatAnthropic 模型

#### Scenario: 未配置自定义端点时使用默认端点

- **GIVEN** 用户未设置 `OPENAI_BASE_URL` 环境变量
- **AND** 用户设置了 `OPENAI_API_KEY`
- **WHEN** 用户运行 `deepagents --model gpt-4o`
- **THEN** 系统使用 OpenAI 官方 API 端点（默认行为）

---

### Requirement: CLI 参数覆盖端点配置

系统 **必须 (SHALL)** 支持通过 CLI 参数 `--base-url` 覆盖环境变量中的端点配置。

#### Scenario: CLI 参数覆盖环境变量

- **GIVEN** 用户设置了环境变量 `OPENAI_BASE_URL=https://api.openai.com/v1`
- **WHEN** 用户运行 `deepagents --model llama-3.1-70b --base-url https://api.groq.com/openai/v1`
- **THEN** 系统使用 `https://api.groq.com/openai/v1` 作为 API 端点（而非环境变量配置的端点）

#### Scenario: 仅使用 CLI 参数配置端点

- **GIVEN** 用户未设置任何 `*_BASE_URL` 环境变量
- **WHEN** 用户运行 `deepagents --model custom-model --base-url https://my-api.com/v1 --provider openai`
- **THEN** 系统使用 `https://my-api.com/v1` 作为 API 端点

---

### Requirement: 显式指定模型提供商

系统 **必须 (SHALL)** 支持通过 CLI 参数 `--provider` 显式指定模型提供商，用于无法自动识别的模型名称。

#### Scenario: 使用 --provider 指定 OpenAI 提供商

- **GIVEN** 用户有一个自定义模型名称 `my-fine-tuned-model`
- **WHEN** 用户运行 `deepagents --model my-fine-tuned-model --provider openai`
- **THEN** 系统使用 OpenAI 兼容的方式初始化模型
- **AND** 不因模型名称无法识别而报错

#### Scenario: --provider 与自动检测冲突时优先使用 --provider

- **GIVEN** 模型名称 `claude-custom` 会被自动识别为 anthropic
- **WHEN** 用户运行 `deepagents --model claude-custom --provider openai`
- **THEN** 系统使用 OpenAI 兼容的方式初始化模型（用户显式指定优先）

---

### Requirement: 基于配置的提供商推断

系统 **必须 (SHALL)** 在无法从模型名称自动识别提供商时，根据已配置的 API 端点推断提供商。

#### Scenario: 根据 OPENAI_BASE_URL 推断提供商

- **GIVEN** 用户设置了 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`
- **AND** 用户未设置其他提供商的配置
- **WHEN** 用户运行 `deepagents --model any-custom-model`
- **THEN** 系统使用 OpenAI 兼容提供商

#### Scenario: 根据 ANTHROPIC_BASE_URL 推断提供商

- **GIVEN** 用户设置了 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_API_KEY`
- **AND** 用户未设置其他提供商的配置
- **WHEN** 用户运行 `deepagents --model glm-4`
- **THEN** 系统使用 Anthropic 兼容提供商

#### Scenario: 多提供商配置时需要显式指定

- **GIVEN** 用户同时设置了 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY`
- **AND** 模型名称无法自动识别（如 `custom-model`）
- **WHEN** 用户运行 `deepagents --model custom-model` 不带 `--provider`
- **THEN** 系统按优先级选择（OpenAI 优先）或提示用户使用 `--provider` 指定

---

### Requirement: 启动信息展示端点配置

系统 **必须 (SHALL)** 在启动界面清晰展示当前使用的 API 端点配置。

#### Scenario: 展示自定义端点信息

- **GIVEN** 用户配置了 `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- **WHEN** 用户启动 deepagents
- **THEN** 启动界面显示类似 `✓ Model: OpenAI (custom) → 'deepseek-chat' @ https://api.deepseek.com/v1`

#### Scenario: 默认端点不显示 URL

- **GIVEN** 用户未配置自定义端点
- **WHEN** 用户启动 deepagents
- **THEN** 启动界面显示类似 `✓ Model: OpenAI → 'gpt-4o'`（不显示 URL）

---

### Requirement: 向后兼容性

系统 **必须 (SHALL)** 保持与现有配置方式的完全向后兼容。

#### Scenario: 不设置新环境变量时行为不变

- **GIVEN** 用户仅设置了 `OPENAI_API_KEY`（未设置任何 `*_BASE_URL`）
- **WHEN** 用户运行 `deepagents --model gpt-4o`
- **THEN** 系统行为与变更前完全一致
- **AND** 使用 OpenAI 官方 API 端点

#### Scenario: 现有环境变量继续生效

- **GIVEN** 用户设置了 `OPENAI_API_KEY`、`OPENAI_MODEL`
- **AND** 未设置任何新增的环境变量
- **WHEN** 用户运行 `deepagents`（不带参数）
- **THEN** 系统使用 `OPENAI_MODEL` 指定的模型和 OpenAI 官方端点

