"""
Test script cho Presentation Agent System
"""

import os
import asyncio
from presentation_agent import presentation_agent

async def test_presentation_creation():
    """Test tạo presentation về AI trong giáo dục"""
    
    # Test topic
    topic = "Ứng dụng AI trong giáo dục: Cơ hội và thách thức"
    
    print(f"🎯 Tạo presentation về: {topic}")
    print("=" * 50)
    
    try:
        # Invoke presentation agent
        result = await presentation_agent.ainvoke({
            "messages": [
                {
                    "role": "user", 
                    "content": f"Hãy tạo một presentation hoàn chỉnh về topic: '{topic}'. Presentation nên có khoảng 15-20 slides và thời lượng 20-25 phút."
                }
            ]
        })
        
        print("✅ Presentation đã được tạo thành công!")
        print("\n📁 Files được tạo:")
        
        # List files created
        files_to_check = [
            "topic.txt",
            "presentation_outline.json", 
            "slides_content.json",
            "presentation.html",
            "speaker_guide.txt"
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                print(f"   ✓ {file}")
            else:
                print(f"   ✗ {file} (không tìm thấy)")
        
        # Show final result
        print(f"\n💬 Agent response: {result['messages'][-1]['content']}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

async def test_simple_topic():
    """Test với topic đơn giản hơn"""
    
    topic = "Python cho người mới bắt đầu"
    
    print(f"\n🎯 Test đơn giản với topic: {topic}")
    print("=" * 50)
    
    try:
        result = await presentation_agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Tạo presentation ngắn về '{topic}' với 8-10 slides cho workshop 15 phút."
                }
            ]
        })
        
        print("✅ Test hoàn thành!")
        
    except Exception as e:
        print(f"❌ Lỗi trong test: {e}")

def main():
    """Main function để chạy tests"""
    print("🚀 Khởi chạy Presentation Agent Test")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  Cảnh báo: TAVILY_API_KEY không được set trong environment")
        print("   Bạn cần set API key để agent hoạt động đầy đủ")
    
    # Run tests
    asyncio.run(test_presentation_creation())
    # asyncio.run(test_simple_topic())

if __name__ == "__main__":
    main()
