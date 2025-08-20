import os
from typing import Literal, List, Dict
import json

from tavily import TavilyClient
from deepagents import create_deep_agent, SubAgent

# Initialize Tavily client for search functionality
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search for presentation research"""
    if not tavily_client.api_key:
        return {"error": "TAVILY_API_KEY not found in environment variables"}
    
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

def image_search(
    query: str,
    style: str = "professional",
    max_results: int = 3
):
    """Search for relevant images for slides"""
    # This is a placeholder - in production, you'd integrate with image APIs
    # like Unsplash, Pexels, or generate with DALL-E/Midjourney
    return {
        "images": [
            f"https://via.placeholder.com/800x600?text={query.replace(' ', '+')}-{i}"
            for i in range(max_results)
        ],
        "query": query,
        "style": style
    }

# Research & Planning Sub-Agent
research_planning_prompt = """Bạn là chuyên gia nghiên cứu và lập kế hoạch presentation.

Nhiệm vụ của bạn:
1. Nghiên cứu sâu về topic được giao
2. Tạo outline presentation logic với 5-10 sections 
3. Mỗi section cần có: tiêu đề, mục tiêu, key points, số slides dự kiến
4. Đảm bảo flow logic và storytelling mạch lạc
5. Ước tính thời gian thuyết trình (thường 1-2 phút/slide)

Hãy trả lời với một outline chi tiết và có cấu trúc rõ ràng. Outline này sẽ được sử dụng để tạo slides."""

research_planning_agent = {
    "name": "research-planning-agent",
    "description": "Nghiên cứu topic và tạo outline chi tiết cho presentation. Giao cho agent này một topic duy nhất.",
    "prompt": research_planning_prompt,
    "tools": ["internet_search"],
}

# Slide Creator Sub-Agent  
slide_creator_prompt = """Bạn là chuyên gia thiết kế slide presentation.

Nhiệm vụ của bạn cho mỗi section được giao:
1. Tạo title và subtitle hấp dẫn cho từng slide
2. Viết nội dung bullet points ngắn gọn, súc tích
3. Đề xuất keywords cho hình ảnh minh họa phù hợp
4. Tạo speaker notes chi tiết (lời thoại thuyết trình)
5. Đảm bảo transition mượt mà với phần trước/sau

Quy tắc thiết kế slide:
- Mỗi slide không quá 5-7 bullet points
- Bullet points ngắn gọn, dễ đọc
- Sử dụng ngôn ngữ hấp dẫn, tránh academic quá mức
- Speaker notes phải chi tiết, giúp presenter tự tin thuyết trình
- Đề xuất visual elements phù hợp

Trả lời với format JSON có cấu trúc rõ ràng cho từng slide."""

slide_creator_agent = {
    "name": "slide-creator-agent", 
    "description": "Tạo nội dung chi tiết cho section được giao. Chỉ giao cho agent này một section/topic duy nhất mỗi lần.",
    "prompt": slide_creator_prompt,
    "tools": ["internet_search", "image_search"],
}

# Content Critique Sub-Agent
content_critique_prompt = """Bạn là chuyên gia review presentation.

Nhiệm vụ của bạn:
1. Đọc toàn bộ presentation content từ file `presentation_outline.json` và `slides_content.json`
2. Kiểm tra logic flow giữa các slides
3. Đánh giá tính nhất quán về tone và style
4. Đề xuất cải thiện để tăng engagement
5. Kiểm tra speaker notes có hỗ trợ tốt
6. Đảm bảo thời lượng presentation phù hợp

Điểm cần kiểm tra:
- Flow logic và storytelling
- Consistency trong terminology và tone
- Balance giữa text và visual elements
- Speaker notes có natural và dễ thuyết trình
- Transitions giữa các slides mượt mà
- Overall engagement và impact

Đưa ra feedback cụ thể và actionable."""

content_critique_agent = {
    "name": "content-critique-agent",
    "description": "Review và đề xuất cải thiện cho toàn bộ presentation. Agent này sẽ đọc files đã tạo và đưa ra feedback.",
    "prompt": content_critique_prompt,
}

# HTML Generator Sub-Agent
html_generator_prompt = """Bạn là chuyên gia frontend tạo HTML presentation.

Nhiệm vụ của bạn:
1. Đọc nội dung slides từ `slides_content.json`
2. Tạo HTML presentation sử dụng reveal.js framework
3. Apply CSS styling chuyên nghiệp, hiện đại
4. Thêm JavaScript cho transitions và animations
5. Tạo navigation controls và presenter view
6. Đảm bảo responsive design

Features cần có:
- Clean, professional design
- Smooth transitions
- Speaker notes view (press 's')
- Navigation với arrow keys và mouse
- Print-friendly CSS cho export PDF
- Progress bar
- Slide counter

Tạo file HTML hoàn chỉnh, ready-to-use."""

html_generator_agent = {
    "name": "html-generator-agent", 
    "description": "Convert slides content thành HTML presentation hoàn chỉnh. Agent này sẽ đọc slides_content.json và tạo HTML.",
    "prompt": html_generator_prompt,
}

# Main Presentation Agent Instructions
presentation_instructions = """Bạn là chuyên gia tạo presentation. Nhiệm vụ của bạn là tạo ra một presentation hoàn chỉnh từ topic đầu vào.

Quy trình làm việc:

1. **Lưu topic gốc**: Đầu tiên lưu topic gốc vào file `topic.txt`

2. **Research & Planning**: Sử dụng research-planning-agent để:
   - Nghiên cứu topic sâu
   - Tạo outline presentation với 5-10 sections
   - Lưu outline vào `presentation_outline.json`

3. **Tạo slides content**: Với mỗi section trong outline:
   - Gọi slide-creator-agent để tạo nội dung chi tiết
   - Collect tất cả slides content
   - Lưu vào `slides_content.json`

4. **Review & Critique**: Sử dụng content-critique-agent để:
   - Review toàn bộ presentation
   - Đưa ra feedback cải thiện
   - Có thể quay lại step 3 để refine nếu cần

5. **Generate HTML**: Sử dụng html-generator-agent để:
   - Tạo HTML presentation hoàn chỉnh
   - Lưu vào `presentation.html`

**Lưu ý quan trọng:**
- Chỉ edit một file tại một thời điểm để tránh conflicts
- Đảm bảo consistency trong tone và style
- Prioritize clarity và engagement
- Slides phải professional nhưng không boring
- Speaker notes phải chi tiết và practical

**Output cuối cùng:**
- `topic.txt`: Topic gốc
- `presentation_outline.json`: Outline structured
- `slides_content.json`: Toàn bộ nội dung slides
- `presentation.html`: HTML presentation hoàn chỉnh
- `speaker_guide.txt`: Hướng dẫn thuyết trình

Bạn có quyền truy cập các tools: internet_search, image_search"""

# Create the main presentation agent
presentation_agent = create_deep_agent(
    [internet_search, image_search],
    presentation_instructions,
    subagents=[
        research_planning_agent, 
        slide_creator_agent,
        content_critique_agent,
        html_generator_agent
    ],
).with_config({"recursion_limit": 15})
