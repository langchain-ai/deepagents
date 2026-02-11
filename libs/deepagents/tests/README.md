# Deep Agents SDK Tests

## API Keys

### Required

- **`ANTHROPIC_API_KEY`** - Required for integration tests using `ChatAnthropic`

### Optional

- **`LANGSMITH_API_KEY`** or **`LANGCHAIN_API_KEY`** - Enables LangSmith tracing for test runs

## Test Utilities

Shared test utilities are in `tests/utils.py`:

- Mock tools (`get_weather`, `get_soccer_scores`, etc.)
- Middleware classes (`ResearchMiddleware`, `WeatherToolMiddleware`, etc.)
- Assertion helpers (`assert_all_deepagent_qualities`)
