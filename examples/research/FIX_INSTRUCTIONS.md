# 修复 langchain-mcp-adapters 工具 schema bug

## 修复位置
文件：`langchain_mcp_adapters/tools.py`
函数：`convert_mcp_tool_to_langchain_tool`

## 修复代码

在 `convert_mcp_tool_to_langchain_tool` 函数中，找到创建 `StructuredTool` 的部分（大约第 387-395 行），替换为：

```python
    # 修复 schema 以确保符合 OpenAI Function Calling 要求
    # OpenAI 要求即使函数没有参数，schema 也必须包含 'properties' 字段
    input_schema = tool.inputSchema
    if input_schema is not None and isinstance(input_schema, dict):
        # 确保 schema 有 properties 字段（OpenAI 要求）
        if "properties" not in input_schema:
            if "type" not in input_schema:
                input_schema["type"] = "object"
            input_schema["properties"] = {}
    
    meta = getattr(tool, "meta", None)
    base = tool.annotations.model_dump() if tool.annotations is not None else {}
    meta = {"_meta": meta} if meta is not None else {}
    metadata = {**base, **meta} or None

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=input_schema,  # 使用修复后的 schema
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata,
    )
```

## 修复说明

这个修复确保：
1. 当 `input_schema` 是 dict 类型时，检查是否有 `properties` 字段
2. 如果没有 `properties` 字段，添加 `{"type": "object", "properties": {}}`
3. 这样即使工具没有参数，schema 也符合 OpenAI Function Calling 的要求

## 应用步骤

1. 在 fork 的仓库中编辑 `langchain_mcp_adapters/tools.py`
2. 找到 `convert_mcp_tool_to_langchain_tool` 函数
3. 在创建 `StructuredTool` 之前添加上述修复代码
4. 提交并推送到你的 fork
5. 在本项目的 `requirements.txt` 中已经更新为使用你的 fork

