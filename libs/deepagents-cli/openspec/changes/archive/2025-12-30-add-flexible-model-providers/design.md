# Design: 灵活的模型提供商配置

## Context

DeepAgents CLI 当前硬编码使用各 LLM 提供商的官方 API 端点。随着开源模型和第三方 API 服务的兴起，用户需要接入：

1. **国产大模型服务**：智普 AI (GLM)、DeepSeek、月之暗面 (Moonshot) 等
2. **开源模型托管服务**：Together AI、Groq、Fireworks AI 等
3. **本地模型服务**：Ollama、vLLM、LocalAI 等
4. **云厂商兼容端点**：Azure OpenAI、AWS Bedrock 等

这些服务大多提供兼容 OpenAI 或 Anthropic API 格式的接口，只需配置 base_url 即可接入。

## Goals / Non-Goals

**Goals**:
- 支持通过环境变量配置自定义 API 端点
- 支持通过 CLI 参数临时覆盖端点配置
- 扩展模型名称自动检测，覆盖更多常见模型
- 提供 `--provider` 参数作为 fallback，处理无法自动识别的情况

**Non-Goals**:
- 不实现多端点负载均衡或 failover
- 不实现 API 密钥的加密存储
- 不实现非标准 API 格式的适配（必须兼容 OpenAI 或 Anthropic 格式）

## Decisions

### Decision 1: 使用环境变量作为主要配置方式

**选择**：新增 `OPENAI_BASE_URL`、`ANTHROPIC_BASE_URL`、`GOOGLE_BASE_URL` 环境变量

**原因**：
- 与现有的 `*_API_KEY` 配置方式一致
- 符合 12-factor app 原则
- LangChain 官方也使用类似的环境变量名（如 `OPENAI_API_BASE`，已废弃）
- 便于在 CI/CD 和容器环境中配置

**替代方案**：
- 配置文件：增加复杂度，不符合 CLI 工具的简洁原则
- 仅 CLI 参数：每次启动都需要输入，不够便捷

### Decision 2: CLI 参数覆盖环境变量

**选择**：`--base-url` 参数优先于环境变量

**原因**：
- 允许临时使用不同端点，无需修改环境变量
- 调试和测试更方便
- 符合「CLI 参数 > 环境变量 > 默认值」的配置优先级惯例

### Decision 3: 扩展模型名称检测

**选择**：在 `_detect_provider()` 中添加更多模式匹配

新增的模式：
| 模式 | 提供商 | 说明 |
|------|--------|------|
| `deepseek-*` | openai | DeepSeek API 兼容 OpenAI 格式 |
| `llama-*`, `llama3*` | openai | Meta Llama 系列（通过 Together/Groq 等） |
| `mistral-*` | openai | Mistral AI（通过兼容端点） |
| `qwen-*` | openai | 通义千问（通过兼容端点） |
| `glm-*` | anthropic | 智普 GLM 系列（支持 Anthropic 协议） |

**原因**：
- 减少用户配置负担，常见模型开箱即用
- 保持向后兼容，不影响现有检测逻辑

### Decision 4: 添加 `--provider` 参数作为 fallback

**选择**：当模型名称无法自动识别时，用户可通过 `--provider` 显式指定

**原因**：
- 不可能穷举所有模型名称
- 用户自定义模型名称无法预测
- 提供最大灵活性

**用法**：
```bash
# 模型名 "my-fine-tuned-gpt" 无法自动识别
deepagents --model my-fine-tuned-gpt --provider openai --base-url https://my-api.com
```

## Risks / Trade-offs

### Risk 1: API 兼容性问题

**风险**：某些「兼容」API 实际上并非 100% 兼容，可能导致功能异常

**缓解**：
- 文档中明确说明「需要真正兼容 OpenAI/Anthropic API 格式的服务」
- 不提供针对特定服务的 workaround，保持代码简洁

### Risk 2: 配置错误难以排查

**风险**：用户配置错误的 base_url 可能导致难以理解的错误信息

**缓解**：
- 在启动时显示当前使用的端点配置
- 连接失败时提供有意义的错误提示

### Risk 3: 安全风险

**风险**：用户可能误将请求发送到恶意服务

**缓解**：
- 启动时明确显示 base_url 配置
- 不自动将请求重定向到第三方服务

## Migration Plan

无需迁移，本变更完全向后兼容。

## Open Questions

1. **是否需要支持 per-provider 的模型名称映射？**
   - 例如：DeepSeek 使用 `deepseek-chat` 而非 `gpt-4`
   - 当前决定：不需要，用户直接使用目标服务的模型名称即可

2. **是否需要支持多个 API key 配置（如备用 key）？**
   - 当前决定：不在本次实现范围内，可作为后续增强

3. **Google Generative AI 是否支持 base_url？**
   - 需要验证 `langchain-google-genai` 是否支持自定义端点
   - 如不支持，该功能仅对 OpenAI 和 Anthropic 可用
