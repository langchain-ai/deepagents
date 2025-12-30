# Tasks: 增加灵活的模型提供商配置支持

## 1. Settings 类扩展

- [x] 1.1 在 `Settings` 数据类中添加 `openai_base_url`、`anthropic_base_url`、`google_base_url` 字段
- [x] 1.2 在 `Settings.from_environment()` 中读取对应的环境变量
- [x] 1.3 添加 `has_custom_openai_base_url` 等便捷属性

## 2. 模型提供商检测增强

- [x] 2.1 保留官方模型名称检测（gpt/claude/gemini），移除第三方模型硬编码
  - 设计变更：不再硬编码 `deepseek-*`、`llama-*`、`glm-*` 等模式
  - 用户通过 `--provider` 参数或环境变量配置自行控制
- [x] 2.2 添加 `_infer_provider_from_config()` 函数，根据已配置的 API key 推断提供商
- [x] 2.3 确保无法识别时返回 None，配合 `--provider` 参数使用

## 3. create_model() 函数修改

- [x] 3.1 添加 `provider_override` 和 `base_url_override` 参数
- [x] 3.2 在创建 ChatOpenAI 时传入 `base_url` 参数
- [x] 3.3 在创建 ChatAnthropic 时传入 `base_url` 参数
- [x] 3.4 Google 暂不支持 base_url（langchain-google-genai 不支持）
- [x] 3.5 优先使用 CLI 传入的 base_url，其次使用环境变量配置

## 4. CLI 参数扩展

- [x] 4.1 在 `parse_args()` 中添加 `--base-url` 参数
- [x] 4.2 在 `parse_args()` 中添加 `--provider` 参数（choices: openai, anthropic, google）
- [x] 4.3 将参数传递给 `main()` 和 `create_model()`
- [x] 4.4 更新帮助文本，说明新参数用途

## 5. 启动信息展示

- [x] 5.1 当使用自定义 base_url 时，在启动界面显示端点信息
- [x] 5.2 格式: `✓ Model: OpenAI (custom) → 'model-name'` + `Endpoint: https://...`

## 6. 文档更新

- [ ] 6.1 更新 README.md 的「Model Configuration」部分
- [ ] 6.2 添加使用第三方 API 的示例
- [ ] 6.3 添加本地模型（Ollama）的配置示例

## 7. 测试

- [x] 7.1 添加 `test_settings_base_url()` 测试环境变量读取
- [x] 7.2 添加 `test_detect_provider()` 测试模型名称检测
- [x] 7.3 添加 `test_infer_provider_from_config()` 测试提供商推断
- [x] 7.4 实际验证 OpenAI 和 Anthropic 兼容端点连接

## 验收标准

1. ✅ 设置 `OPENAI_BASE_URL` 后，能成功连接到第三方 API（已验证：one-api.yafex.cn）
2. ✅ 设置 `ANTHROPIC_BASE_URL` 后，能成功连接到智普 AI 的 Anthropic 协议端点
3. ✅ `--base-url` 和 `--provider` 参数能覆盖环境变量配置
4. ✅ 不设置任何新配置时，行为与当前版本完全一致（向后兼容）
5. ✅ 所有新增测试通过（83 个单元测试全部通过）

## 待完成

- 文档更新（README.md）可作为后续工作
