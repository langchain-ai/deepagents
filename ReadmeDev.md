libs/cli/
├── pyproject.toml             # 项目元数据与依赖配置
├── Makefile                   # 包含常用的开发命令 (如 make run-dev)
├── deepagents_cli/            # 源代码根目录
│   ├── __main__.py            # python -m deepagents_cli 入口
│   ├── main.py                # CLI 命令分发与主逻辑
│   ├── app.py                 # Textual App 类，负责 UI 整体架构
│   ├── agent.py               # 封装 LangGraph Agent 的创建逻辑
│   ├── config.py              # 全局配置、Console 设置、Settings 加载
│   ├── sessions.py            # 会话与 Thread 管理 (SQLite)
│   ├── ui.py                  # CLI 辅助展示组件 (如 Help 文本)
│   ├── tools.py               # 基础工具定义 (搜索、HTTP)
│   ├── widgets/               # 自定义 Textual 组件
│   │   ├── chat_input.py      # 输入框 (支持多行与模式切换)
│   │   ├── messages.py        # 消息气泡渲染
│   │   ├── approval.py        # 工具调用确认按钮
│   │   ├── diff.py            # 代码差异预览
│   │   └── status.py          # 底部状态栏 (Token 统计等)
│   ├── mcp/                   # Model Context Protocol 协议集成
│   ├── skills/                # 插件化技能系统逻辑
│   └── integrations/          # 外部沙箱集成 (Daytona, Modal, Runloop)
└── tests/                     # 单元测试与集成测试