# Change: 增加灵活的模型提供商配置支持

## Why

当前 DeepAgents CLI 仅支持官方的 OpenAI、Anthropic 和 Google API 端点，无法接入：
- 兼容 OpenAI API 格式的第三方服务（如 DeepSeek、Together AI、Groq、vLLM、Ollama、Azure OpenAI）
- 兼容 Anthropic/Claude API 格式的第三方服务（如智普 AI 的 Anthropic 协议接口）

用户需要自定义 API 端点和密钥来接入这些服务，以获得更灵活的模型选择、更低的成本或满足特定的合规要求。

## What Changes

### 1. 环境变量扩展

新增以下环境变量支持自定义 API 端点：
- `OPENAI_BASE_URL`: OpenAI 兼容 API 的自定义端点（如 `https://api.deepseek.com/v1`）
- `ANTHROPIC_BASE_URL`: Anthropic 兼容 API 的自定义端点（如 `https://open.bigmodel.cn/api/anthropic`）
- `GOOGLE_BASE_URL`: Google Generative AI 的自定义端点

### 2. CLI 参数扩展

新增 `--base-url` 参数，用于运行时指定 API 端点：
```bash
deepagents --model deepseek-chat --base-url https://api.deepseek.com/v1
```

### 3. 模型名称检测增强

扩展 `_detect_provider()` 函数，支持更多模型名称模式：
- OpenAI 兼容: `gpt-*`, `o1-*`, `o3-*`, `deepseek-*`, `llama-*`, `mistral-*`, `qwen-*`, `doubao-*`, `baichuan-*`
- Anthropic 兼容: `claude-*`, `glm-*`（智普 GLM 系列通过 Anthropic 协议）

**新增 `--provider` 参数**：当模型名称无法自动识别时，显式指定提供商：
```bash
deepagents --model my-custom-model --provider openai --base-url https://my-api.com
```

### 4. Settings 类扩展

在 `Settings` 数据类中添加：
```python
openai_base_url: str | None
anthropic_base_url: str | None
google_base_url: str | None
```

### 5. create_model() 函数修改

将检测到的 base_url 传递给对应的 ChatModel 构造函数：
```python
# OpenAI
ChatOpenAI(model=model_name, base_url=base_url)

# Anthropic
ChatAnthropic(model_name=model_name, base_url=base_url)

# Google
ChatGoogleGenerativeAI(model=model_name, base_url=base_url)
```

## Impact

- Affected specs: `model-configuration`（新建）
- Affected code: 
  - `deepagents_cli/config.py`: Settings 类、create_model() 函数
  - `deepagents_cli/main.py`: CLI 参数解析
  - `tests/unit_tests/test_config.py`: 新增 base_url 配置测试

## 向后兼容性

- **完全向后兼容**: 所有新增配置都是可选的
- 不设置新环境变量时，行为与当前版本完全一致
- 现有的 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY` 继续按原有逻辑工作

## 使用示例

### 示例 1: 使用 DeepSeek API

```bash
export OPENAI_API_KEY="your-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
deepagents --model deepseek-chat
```

### 示例 2: 使用智普 AI（Anthropic 协议）

```bash
export ANTHROPIC_API_KEY="your-zhipu-key"
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
deepagents --model glm-4
```

### 示例 3: CLI 参数覆盖

```bash
# 环境变量中配置的是官方 OpenAI，临时使用 Groq
deepagents --model llama-3.1-70b --base-url https://api.groq.com/openai/v1 --provider openai
```

### 示例 4: 本地 Ollama

```bash
export OPENAI_API_KEY="ollama"  # Ollama 不需要真实 key
export OPENAI_BASE_URL="http://localhost:11434/v1"
deepagents --model llama3.2
```
