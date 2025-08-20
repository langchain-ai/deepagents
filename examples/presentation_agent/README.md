# 🎯 Presentation Agent System

Hệ thống tạo slide thuyết trình tự động sử dụng AI agents, từ topic đầu vào tạo ra presentation HTML hoàn chỉnh.

## 🚀 Tính năng chính

- **Tự động nghiên cứu**: Tìm kiếm thông tin từ internet
- **Tạo outline thông minh**: Cấu trúc presentation logic 
- **Thiết kế slide chuyên nghiệp**: Nội dung ngắn gọn, hấp dẫn
- **Speaker notes chi tiết**: Hỗ trợ thuyết trình tự tin
- **HTML presentation**: Không cần phần mềm, chạy trên browser
- **Responsive design**: Tương thích mobile và desktop

## 📁 Cấu trúc Project

```
presentation_agent/
├── presentation_agent.py      # Agent chính
├── test_presentation_agent.py # File test
├── requirements.txt           # Dependencies
├── templates/
│   └── presentation_template.html  # Template HTML
└── README.md                 # Hướng dẫn này
```

## 🛠️ Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Tavily API Key:**
```bash
export TAVILY_API_KEY="your_tavily_api_key_here"
```

## 🎮 Cách sử dụng

### 1. Basic Usage

```python
from presentation_agent import presentation_agent

# Tạo presentation
result = await presentation_agent.ainvoke({
    "messages": [
        {
            "role": "user", 
            "content": "Tạo presentation về 'Machine Learning cơ bản' với 15 slides"
        }
    ]
})
```

### 2. Chạy Test

```bash
python test_presentation_agent.py
```

### 3. Output Files

Sau khi chạy thành công, bạn sẽ có:

- `topic.txt` - Topic gốc
- `presentation_outline.json` - Outline structured
- `slides_content.json` - Nội dung slides
- `presentation.html` - HTML presentation hoàn chỉnh
- `speaker_guide.txt` - Hướng dẫn thuyết trình

## 🎨 Features của HTML Presentation

### Điều khiển cơ bản:
- **→ / Space**: Slide tiếp theo
- **← / Backspace**: Slide trước  
- **S**: Bật/tắt speaker notes
- **F**: Chế độ toàn màn hình
- **?**: Hiện trợ giúp

### Tính năng:
- ✅ Navigation với keyboard/mouse
- ✅ Progress bar
- ✅ Slide counter
- ✅ Speaker notes view
- ✅ Responsive mobile support
- ✅ Touch/swipe support
- ✅ Print-friendly CSS
- ✅ Auto-hide cursor

## 🏗️ Kiến trúc Hệ thống

### Main Agent
**Presentation Orchestrator** - Điều phối toàn bộ quy trình

### Sub-Agents

1. **Research Planning Agent**
   - Nghiên cứu topic
   - Tạo outline presentation
   - Tools: `internet_search`

2. **Slide Creator Agent**  
   - Tạo nội dung slides
   - Viết speaker notes
   - Tools: `internet_search`, `image_search`

3. **Content Critique Agent**
   - Review và cải thiện content
   - Đảm bảo consistency

4. **HTML Generator Agent**
   - Convert content thành HTML
   - Apply styling và interactions

## 🔄 Workflow

```
Topic Input
    ↓
Research & Planning (5-10 min)
    ↓  
Outline Creation
    ↓
Parallel Slide Creation (10-20 min)
    ↓
Content Review & Polish (5-10 min)
    ↓
HTML Generation (5 min)
    ↓
Final Presentation
```

## 🎯 Ví dụ Topics

### Business & Technology
- "Digital Transformation trong doanh nghiệp"
- "Startup ecosystem Việt Nam"
- "Cybersecurity trends 2024"

### Education & Training  
- "Python cho người mới bắt đầu"
- "Data Science fundamentals"
- "Agile project management"

### Research & Analysis
- "Climate change impacts"
- "AI ethics và society"
- "Cryptocurrency market analysis"

## 🛠️ Customization

### 1. Thay đổi Template

Edit `templates/presentation_template.html` để:
- Đổi color scheme
- Thay đổi fonts
- Customize animations
- Thêm branding

### 2. Extend Sub-Agents

Thêm sub-agents mới:

```python
custom_agent = {
    "name": "custom-agent",
    "description": "Custom functionality",
    "prompt": "Your custom prompt",
    "tools": ["custom_tool"]
}
```

### 3. Thêm Tools

```python
def custom_tool(param: str):
    """Custom tool description"""
    # Your implementation
    return result
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **TAVILY_API_KEY not found**
   - Đảm bảo đã set environment variable
   - Check API key còn hạn sử dụng

2. **Agent timeout**
   - Tăng recursion_limit trong config
   - Chia nhỏ topic phức tạp

3. **HTML không hiển thị đúng**
   - Check browser console for errors
   - Verify slides_content.json format

### Debug mode:

```python
agent = create_deep_agent(
    tools=[internet_search, image_search],
    instructions=presentation_instructions,
    subagents=subagents,
).with_config({
    "recursion_limit": 20,
    "verbose": True
})
```

## 📝 TODO & Roadmap

- [ ] Tích hợp image generation (DALL-E/Midjourney)
- [ ] Export to PowerPoint/PDF
- [ ] Voice narration synthesis  
- [ ] Interactive elements (polls, quizzes)
- [ ] Template gallery
- [ ] Collaboration features
- [ ] Analytics tracking

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file

## 🆘 Support

- Issues: GitHub Issues
- Docs: README.md
- Examples: test_presentation_agent.py

---

**Happy Presenting! 🎉**
