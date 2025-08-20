"""
Demo tạo presentation mẫu với static data
Không cần API key để chạy demo này
"""

import json
import os
from datetime import datetime

def create_demo_presentation():
    """Tạo một presentation demo về 'Python cơ bản'"""
    
    # Topic demo
    topic = "Python Programming cơ bản cho người mới bắt đầu"
    
    # Tạo demo outline
    outline = {
        "title": topic,
        "estimated_duration": "20-25 phút",
        "target_audience": "Người mới bắt đầu lập trình",
        "sections": [
            {
                "id": 1,
                "title": "Giới thiệu về Python",
                "slides_count": 2,
                "duration": "3 phút",
                "objectives": ["Hiểu Python là gì", "Ưu điểm của Python"]
            },
            {
                "id": 2, 
                "title": "Cài đặt Python Environment",
                "slides_count": 2,
                "duration": "4 phút",
                "objectives": ["Cài đặt Python", "Setup IDE"]
            },
            {
                "id": 3,
                "title": "Syntax cơ bản",
                "slides_count": 3,
                "duration": "6 phút", 
                "objectives": ["Variables", "Data types", "Basic operations"]
            },
            {
                "id": 4,
                "title": "Control Flow",
                "slides_count": 3,
                "duration": "6 phút",
                "objectives": ["If/else", "Loops", "Functions"]
            },
            {
                "id": 5,
                "title": "Thực hành & Kết luận",
                "slides_count": 2,
                "duration": "6 phút",
                "objectives": ["Bài tập demo", "Next steps"]
            }
        ]
    }
    
    # Tạo demo slides content
    slides_content = {
        "title": topic,
        "total_slides": 12,
        "slides": [
            {
                "id": 1,
                "title": "Python Programming cơ bản",
                "subtitle": "Cho người mới bắt đầu",
                "content": [
                    "Ngôn ngữ lập trình dễ học",
                    "Syntax đơn giản, dễ hiểu", 
                    "Cộng đồng lớn và hỗ trợ tốt",
                    "Ứng dụng rộng rãi: Web, AI, Data Science"
                ],
                "image_suggestions": ["python-logo", "programming-workspace"],
                "speaker_notes": "Chào mừng các bạn đến với workshop Python cơ bản. Hôm nay chúng ta sẽ tìm hiểu về Python - một ngôn ngữ lập trình rất phổ biến và dễ học. Python được thiết kế với triết lý 'đơn giản là tốt nhất', giúp người mới bắt đầu có thể tiếp cận lập trình một cách dễ dàng."
            },
            {
                "id": 2,
                "title": "Tại sao chọn Python?",
                "content": [
                    "✅ Syntax gần với ngôn ngữ tự nhiên",
                    "✅ Học curve thoải mái cho beginners",
                    "✅ Libraries và frameworks phong phú",
                    "✅ Job opportunities cao",
                    "✅ Open source và miễn phí"
                ],
                "speaker_notes": "Python có nhiều ưu điểm vượt trội. Đầu tiên là syntax rất gần với ngôn ngữ tự nhiên, ví dụ thay vì viết những câu lệnh phức tạp, bạn có thể viết 'if name == John' rất dễ hiểu. Thứ hai, Python có học curve thoải mái, nghĩa là bạn có thể bắt đầu viết code đơn giản ngay từ ngày đầu."
            },
            {
                "id": 3,
                "title": "Cài đặt Python",
                "content": [
                    "Tải từ python.org",
                    "Chọn phiên bản Python 3.x",
                    "Tick 'Add Python to PATH'",
                    "Verify: python --version"
                ],
                "image_suggestions": ["python-installation", "command-line"],
                "speaker_notes": "Để bắt đầu với Python, trước tiên chúng ta cần cài đặt Python trên máy tính. Các bạn vào trang python.org, tải phiên bản mới nhất của Python 3. Quan trọng là phải tick vào ô 'Add Python to PATH' để có thể chạy Python từ command line. Sau khi cài xong, mở terminal và gõ 'python --version' để kiểm tra."
            },
            {
                "id": 4,
                "title": "Chọn IDE/Editor",
                "content": [
                    "🆓 VS Code (khuyên dùng)",
                    "🆓 PyCharm Community",
                    "🆓 Jupyter Notebook", 
                    "🆓 Sublime Text",
                    "💰 PyCharm Professional"
                ],
                "speaker_notes": "Để viết code Python hiệu quả, bạn nên chọn một IDE hoặc editor phù hợp. Tôi khuyên dùng VS Code vì nó miễn phí, nhẹ, và có nhiều extension hỗ trợ Python rất tốt. PyCharm cũng là lựa chọn tuyệt vời với nhiều tính năng advanced. Jupyter Notebook rất phù hợp cho data science và machine learning."
            },
            {
                "id": 5,
                "title": "Variables và Data Types",
                "content": [
                    "name = 'John'  # String",
                    "age = 25       # Integer", 
                    "height = 1.75  # Float",
                    "is_student = True  # Boolean",
                    "fruits = ['apple', 'banana']  # List"
                ],
                "speaker_notes": "Python có các kiểu dữ liệu cơ bản. String để lưu text, Integer cho số nguyên, Float cho số thập phân, Boolean cho True/False, và List để lưu nhiều giá trị. Điều đặc biệt là Python tự động nhận biết kiểu dữ liệu, bạn không cần khai báo như int, string."
            },
            {
                "id": 6,
                "title": "Basic Operations",
                "content": [
                    "# Arithmetic",
                    "result = 10 + 5  # 15",
                    "# String operations", 
                    "full_name = first + ' ' + last",
                    "# List operations",
                    "fruits.append('orange')"
                ],
                "speaker_notes": "Python hỗ trợ các phép toán cơ bản như cộng, trừ, nhân, chia. Với string, bạn có thể nối chuỗi bằng dấu +. Với list, có thể thêm phần tử bằng append(), xóa bằng remove(). Các operations này rất intuitive và dễ nhớ."
            },
            {
                "id": 7,
                "title": "Print và Input",
                "content": [
                    "# Output",
                    "print('Hello World!')",
                    "print(f'Tôi {age} tuổi')",
                    "# Input",
                    "name = input('Tên bạn là gì? ')",
                    "age = int(input('Bạn bao nhiêu tuổi? '))"
                ],
                "speaker_notes": "Print() để hiển thị output ra màn hình. F-string (với f'') là cách hiện đại để format string trong Python. Input() để nhận dữ liệu từ user, lưu ý input() luôn trả về string nên cần convert nếu muốn số."
            },
            {
                "id": 8,
                "title": "Conditional Statements",
                "content": [
                    "if age >= 18:",
                    "    print('Bạn đã trưởng thành')",
                    "elif age >= 13:",
                    "    print('Bạn là teenager')", 
                    "else:",
                    "    print('Bạn còn nhỏ')"
                ],
                "speaker_notes": "If/elif/else giúp chương trình đưa ra quyết định. Lưu ý Python sử dụng indentation (thụt lề) thay vì dấu {} để phân biệt block code. Thường dùng 4 spaces cho mỗi level indentation."
            },
            {
                "id": 9,
                "title": "Loops - For Loop",
                "content": [
                    "# Loop through list",
                    "for fruit in fruits:",
                    "    print(fruit)",
                    "# Loop with range", 
                    "for i in range(5):",
                    "    print(f'Số {i}')"
                ],
                "speaker_notes": "For loop để lặp qua các phần tử. Có thể loop qua list, string, hoặc dùng range() để tạo dãy số. Range(5) tạo ra 0,1,2,3,4. For loop rất mạnh mẽ và được dùng thường xuyên trong Python."
            },
            {
                "id": 10,
                "title": "Functions",
                "content": [
                    "def greet(name):",
                    "    return f'Xin chào {name}!'",
                    "",
                    "# Gọi function",
                    "message = greet('John')",
                    "print(message)  # Xin chào John!"
                ],
                "speaker_notes": "Functions giúp tái sử dụng code và tổ chức chương trình tốt hơn. Định nghĩa với def, có thể có parameters và return value. Functions là building blocks quan trọng của programming."
            },
            {
                "id": 11,
                "title": "Bài tập thực hành",
                "content": [
                    "Viết chương trình:",
                    "1. Nhập tên và tuổi",
                    "2. Kiểm tra tuổi >= 18", 
                    "3. In thông báo phù hợp",
                    "4. Bonus: Tính năm sinh"
                ],
                "speaker_notes": "Bây giờ chúng ta sẽ thực hành với một bài tập đơn giản. Các bạn hãy viết chương trình nhập tên và tuổi, sau đó kiểm tra xem người đó đã trưởng thành chưa và in thông báo. Ai làm nhanh có thể thêm tính năng tính năm sinh."
            },
            {
                "id": 12,
                "title": "Next Steps",
                "content": [
                    "🎯 Học tiếp: OOP, Libraries",
                    "🎯 Thực hành: HackerRank, LeetCode",
                    "🎯 Projects: Web app, Data analysis",
                    "🎯 Community: Python Vietnam",
                    "📚 Resources: python.org, realpython.com"
                ],
                "speaker_notes": "Chúc mừng các bạn đã hoàn thành workshop Python cơ bản! Để tiếp tục học Python, tôi khuyên các bạn nên học về OOP, thư viện như pandas, requests. Hãy thực hành thường xuyên trên các trang như HackerRank. Và quan trọng nhất là bắt đầu làm projects thực tế. Cảm ơn các bạn đã tham gia!"
            }
        ]
    }
    
    # Lưu files
    with open('topic.txt', 'w', encoding='utf-8') as f:
        f.write(topic)
    
    with open('presentation_outline.json', 'w', encoding='utf-8') as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    
    with open('slides_content.json', 'w', encoding='utf-8') as f:
        json.dump(slides_content, f, ensure_ascii=False, indent=2)
    
    # Tạo speaker guide
    speaker_guide = f"""
# 🎤 Speaker Guide - {topic}

## 📊 Overview
- **Thời lượng**: {outline['estimated_duration']}
- **Audience**: {outline['target_audience']}
- **Tổng slides**: {slides_content['total_slides']}

## 🎯 Objectives
Sau workshop này, học viên sẽ:
- Hiểu được Python là gì và tại sao nên học
- Biết cách cài đặt và setup environment
- Nắm được syntax cơ bản của Python
- Có thể viết chương trình Python đơn giản

## ⏰ Timeline
{chr(10).join([f"- {section['title']}: {section['duration']}" for section in outline['sections']])}

## 💡 Presentation Tips

### Slide 1-2: Introduction (3 phút)
- Tạo không khí thân thiện, hỏi kinh nghiệm lập trình của audience
- Nhấn mạnh Python dễ học, đừng lo lắng nếu chưa có kinh nghiệm

### Slide 3-4: Setup (4 phút)  
- Demo live cài đặt nếu có thể
- Chuẩn bị link download sẵn
- Có backup plan nếu internet chậm

### Slide 5-7: Syntax (6 phút)
- Code live, đừng chỉ đọc slides
- Khuyến khích audience follow along
- Giải thích tại sao Python không cần declare types

### Slide 8-10: Control Flow (6 phút)
- Nhấn mạnh indentation quan trọng trong Python
- Cho ví dụ thực tế, dễ relate
- Demo loops với examples vui nhộn

### Slide 11: Practice (6 phút)
- Cho audience time coding
- Walk around, help cá nhân
- Share solutions và discuss

### Slide 12: Wrap-up (2 phút)
- Encourage practice regularly
- Share resources và community links
- Q&A session

## 🚨 Common Questions & Answers

**Q: Python có khó học không?**
A: Python được thiết kế để dễ học. Nếu bạn có thể đọc tiếng Anh, bạn có thể hiểu Python code.

**Q: Tôi cần background math để học Python không?**
A: Không nhất thiết. Tùy vào mục đích sử dụng. Web development thì ít math, AI/Data Science thì cần nhiều hơn.

**Q: Python có chậm không?**
A: Python có slower execution nhưng development speed nhanh. Và có nhiều cách optimize khi cần.

**Q: Nên học Python version nào?**
A: Python 3.x, hiện tại là 3.11+. Python 2 đã deprecated.

## 📱 Emergency Contacts
- Technical Support: [your-contact]
- Backup slides: [backup-location]

---
*Generated by Presentation Agent System*
*{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open('speaker_guide.txt', 'w', encoding='utf-8') as f:
        f.write(speaker_guide)
    
    # Tạo HTML presentation
    create_demo_html(slides_content)
    
    print("✅ Demo presentation đã được tạo thành công!")
    print("\n📁 Files được tạo:")
    print("   ✓ topic.txt")
    print("   ✓ presentation_outline.json")
    print("   ✓ slides_content.json") 
    print("   ✓ speaker_guide.txt")
    print("   ✓ presentation.html")
    print(f"\n🎉 Mở presentation.html trong browser để xem kết quả!")

def create_demo_html(slides_content):
    """Tạo HTML presentation từ slides content"""
    
    # Đọc template
    template_path = 'templates/presentation_template.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Tạo slides HTML
    slides_html = ""
    speaker_notes = []
    
    for slide in slides_content['slides']:
        slide_html = f'<div class="slide" id="slide-{slide["id"]}">\n'
        
        if slide.get('subtitle'):
            slide_html += f'    <h1>{slide["title"]}</h1>\n'
            slide_html += f'    <p style="font-size: 1.8em; color: #7f8c8d; margin-bottom: 40px;">{slide["subtitle"]}</p>\n'
        else:
            slide_html += f'    <h2>{slide["title"]}</h2>\n'
        
        if slide.get('content'):
            slide_html += '    <ul>\n'
            for item in slide['content']:
                slide_html += f'        <li>{item}</li>\n'
            slide_html += '    </ul>\n'
        
        slide_html += '</div>\n\n'
        slides_html += slide_html
        
        # Speaker notes
        speaker_notes.append(slide.get('speaker_notes', ''))
    
    # Replace template variables
    html_content = template.replace('{{title}}', slides_content['title'])
    html_content = html_content.replace('{{total_slides}}', str(slides_content['total_slides']))
    html_content = html_content.replace('{{slides_content}}', slides_html)
    html_content = html_content.replace('{{speaker_notes}}', json.dumps(speaker_notes, ensure_ascii=False))
    
    # Lưu HTML
    with open('presentation.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    create_demo_presentation()
