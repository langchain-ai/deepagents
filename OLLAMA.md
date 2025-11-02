# Using Ollama with Deep Agents

Deep Agents now supports running with local LLM models via [Ollama](https://ollama.ai/). This allows you to run Deep Agents completely offline using models like Llama, Mistral, Qwen, and many others.

## Prerequisites

1. **Install Ollama**: Download and install Ollama from [https://ollama.ai](https://ollama.ai)

2. **Pull a model**: Download a model to use with Deep Agents
   ```bash
   # Example: Pull the qwen2.5-coder model (recommended for coding tasks)
   ollama pull qwen2.5-coder:14b

   # Or pull other models
   ollama pull llama3.1:8b
   ollama pull mistral:latest
   ```

3. **Verify Ollama is running**: Ollama should start automatically. You can verify it's running by checking:
   ```bash
   ollama list
   ```

## Configuration

To use Ollama with the Deep Agents CLI, set the following environment variables:

```bash
# Required: The Ollama server URL
export OLLAMA_BASE_URL=http://localhost:11434

# Optional: The model to use (defaults to llama2)
export OLLAMA_MODEL=qwen2.5-coder:14b
```

You can also add these to your `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:14b
```

## Usage with the CLI

Once configured, simply run the Deep Agents CLI as usual:

```bash
deepagents
```

The CLI will automatically detect your Ollama configuration and use it instead of OpenAI or Anthropic.

## Usage in Python

You can use Ollama models programmatically with Deep Agents:

```python
import os
from langchain_ollama import ChatOllama
from deepagents import create_deep_agent

# Configure the Ollama model
model = ChatOllama(
    model="qwen2.5-coder:14b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# Create a deep agent with the Ollama model
agent = create_deep_agent(
    model=model,
    tools=[...],  # Your tools here
    system_prompt="Your custom instructions..."
)

# Use the agent
result = agent.invoke({"messages": [{"role": "user", "content": "Your task here"}]})
```

## Recommended Models

For coding tasks with Deep Agents:
- **qwen2.5-coder:14b** - Excellent for coding tasks, good balance of speed and quality
- **qwen2.5-coder:7b** - Faster, good for simpler tasks
- **deepseek-coder:33b** - Very capable but requires more resources

For general tasks:
- **llama3.1:8b** - Fast and capable general-purpose model
- **mistral:latest** - Good balance of speed and quality
- **qwen3:14b** - Strong reasoning capabilities

## Model Priority

The Deep Agents CLI checks for model configurations in this order:
1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Anthropic** (if `ANTHROPIC_API_KEY` is set)
3. **Ollama** (if `OLLAMA_BASE_URL` is set)

To ensure Ollama is used, make sure `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are not set.

## Performance Tips

1. **Use appropriate model sizes**: Larger models (14b+) provide better quality but require more RAM and are slower
2. **Adjust temperature**: Lower temperatures (0.1-0.3) work better for deterministic tasks like coding
3. **GPU acceleration**: If you have a GPU, Ollama will automatically use it for faster inference
4. **Context length**: Be aware of your model's context window - some tasks may require models with larger context windows

## Troubleshooting

### Ollama not found
```bash
# Check if Ollama is installed
which ollama

# Check if Ollama service is running
ollama list
```

### Model not available
```bash
# Pull the model
ollama pull qwen2.5-coder:14b
```

### Connection refused
Make sure Ollama is running and accessible at the configured base URL (default: http://localhost:11434)

### Out of memory
Try using a smaller model variant:
```bash
ollama pull qwen2.5-coder:7b
export OLLAMA_MODEL=qwen2.5-coder:7b
```

## Example: Complete Setup

```bash
# 1. Install Ollama (visit https://ollama.ai)

# 2. Pull a coding model
ollama pull qwen2.5-coder:14b

# 3. Configure environment
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5-coder:14b

# 4. Run Deep Agents CLI
deepagents

# You should see: "Using Ollama model: qwen2.5-coder:14b at http://localhost:11434"
```

## Benefits of Using Ollama

- **Privacy**: Your code and data never leave your machine
- **Cost**: No API costs, runs completely free on your hardware
- **Offline**: Works without internet connection
- **Customization**: Fine-tune models for your specific use cases
- **Speed**: For small models, can be faster than API calls (no network latency)

## Limitations

- **Resource intensive**: Requires significant RAM (8GB+ for smaller models)
- **Quality trade-offs**: Local models may not match the quality of GPT-4 or Claude for complex tasks
- **Context windows**: Some local models have smaller context windows than cloud models
- **Setup required**: Need to install and manage Ollama and models locally
