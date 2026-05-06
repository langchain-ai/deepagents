# 电商自动化助手 - 快速启动指南

## 项目介绍

这是一个基于 DeepAgents 框架的多平台电商自动化运营工具，支持抖音、拼多多、淘宝等平台。

## 技术栈

- **AI Agent框架**: DeepAgents (替代原来的 LangGraph + LangChain)
- **浏览器自动化**: Playwright
- **后端**: FastAPI + APScheduler
- **前端**: Vue3 + Element Plus
- **数据库**: SQLite + Chroma (向量数据库)

## 快速开始

### 1. 安装依赖

#### 后端依赖
```bash
cd /workspace/examples/ecommerce_agent
uv sync
```

#### 前端依赖
```bash
cd frontend
npm install
```

### 2. 安装 Playwright 浏览器

```bash
playwright install
```

### 3. 启动服务

#### 启动后端
```bash
cd /workspace/examples/ecommerce_agent
uv run python -m backend.main
```

后端将在 http://localhost:8000 启动

#### 启动前端 (开发模式)
```bash
cd frontend
npm run dev
```

前端将在 http://localhost:5173 启动

### 4. 开始使用

1. 打开浏览器访问 http://localhost:5173
2. 添加店铺（支持抖音/拼多多/淘宝）
3. 创建任务（商品发布/好评管理/数据采集/运营分析）
4. 查看任务执行进度

## 项目结构

```
ecommerce_agent/
├── backend/
│   ├── agent/           # Agent 核心逻辑
│   ├── browser/         # 浏览器管理
│   ├── database/        # 数据库模型
│   ├── scheduler/       # 定时任务
│   ├── knowledge/       # 知识库
│   ├── api/             # API 路由
│   ├── config.py        # 配置文件
│   └── main.py          # FastAPI 主入口
├── frontend/            # Vue3 前端
├── data/                # 数据目录
├── configs/             # 配置文件
├── scripts/             # 脚本目录
└── README.md            # 项目说明
```

## 核心功能

- ✅ 多店铺管理与隔离
- ✅ DOM 元素模块化配置
- ✅ 防检测反风控系统
- ✅ 基于 DeepAgents 的智能代理
- ✅ 定时任务调度
- ✅ 知识库与经验库
- ✅ 任务进度可视化
- ✅ 支持打包为 EXE

## 后续开发

本项目已经搭建好了完整的框架结构，后续可以根据具体需求：

1. 完善具体平台的 DOM 元素选择器
2. 实现具体的业务流程逻辑
3. 添加更多平台支持
4. 完善错误处理和恢复机制
5. 添加更多功能模块

## 打包 EXE

```bash
cd /workspace/examples/ecommerce_agent
uv run python scripts/build.py
```

打包后的文件将位于 dist/ 目录。
