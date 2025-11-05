# 修复 langchain-mcp-adapters 中工具 schema 的 bug

## 问题描述
当 MCP 工具没有参数时，`tool.inputSchema` 可能缺少 `properties` 字段，导致 OpenAI API 返回错误：
```
Invalid schema for function 'check_login_status': In context=(), object schema missing properties.
```

## 修复位置
文件：`langchain_mcp_adapters/tools.py`
函数：`convert_mcp_tool_to_langchain_tool`

## 修复代码

在 `convert_mcp_tool_to_langchain_tool` 函数中，创建 `StructuredTool` 之前添加 schema 修复逻辑：

```python
def convert_mcp_tool_to_langchain_tool(
    session: ClientSession | None,
    tool: MCPTool,
    *,
    connection: Connection | None = None,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
) -> BaseTool:
    # ... 前面的代码保持不变 ...
    
    # 修复 schema：确保符合 OpenAI Function Calling 要求
    input_schema = tool.inputSchema
    if input_schema is not None:
        # 如果 schema 是 dict 类型，确保有 properties 字段
        if isinstance(input_schema, dict):
            if "properties" not in input_schema:
                # 确保有 type 字段
                if "type" not in input_schema:
                    input_schema["type"] = "object"
                # 添加空的 properties 字段
                input_schema["properties"] = {}
        # 如果 input_schema 是 BaseModel 类型，检查其 JSON schema
        elif hasattr(input_schema, "model_json_schema"):
            schema_dict = input_schema.model_json_schema()
            if "properties" not in schema_dict:
                # 创建新的 BaseModel 确保有 properties
                from pydantic import create_model
                FixedSchema = create_model(
                    f"{tool.name}_Args",
                )
                input_schema = FixedSchema
    
    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=input_schema,  # 使用修复后的 schema
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
    )
```

## 或者更简单的修复方法

如果 `input_schema` 是 dict，直接修复：

```python
    # 修复 schema 以确保符合 OpenAI Function Calling 要求
    input_schema = tool.inputSchema
    if input_schema is not None and isinstance(input_schema, dict):
        # 确保 schema 有 properties 字段（OpenAI 要求）
        if "properties" not in input_schema:
            if "type" not in input_schema:
                input_schema["type"] = "object"
            input_schema["properties"] = {}
    
    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=input_schema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
    )
```

## 应用修复

1. Fork 仓库：`git@github.com:kilingzhang/langchain-mcp-adapters.git`
2. 在 `langchain_mcp_adapters/tools.py` 文件的 `convert_mcp_tool_to_langchain_tool` 函数中应用上述修复
3. 提交并推送修复
4. 更新项目依赖使用修复后的版本

