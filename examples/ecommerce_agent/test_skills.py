#!/usr/bin/env python3
"""
测试技能系统
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deepagents.middleware.skills import SkillsMiddleware
from deepagents.backends.filesystem import FilesystemBackend


def test_skills_system():
    """测试技能系统"""
    print("=" * 60)
    print("测试技能系统")
    print("=" * 60)
    
    # 1. 检查技能目录
    skills_dir = project_root / "skills"
    print(f"\n[1] 检查技能目录: {skills_dir}")
    
    if not skills_dir.exists():
        print(f"❌ 技能目录不存在: {skills_dir}")
        return False
    
    print("✅ 技能目录存在")
    
    # 2. 列出技能目录结构
    print("\n[2] 技能目录结构:")
    for root, dirs, files in os.walk(skills_dir):
        level = root.replace(str(skills_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # 3. 检查各个技能的 SKILL.md
    print("\n[3] 检查技能文件:")
    ecommerce_dir = skills_dir / "ecommerce"
    if ecommerce_dir.exists():
        skill_names = ["product-publish", "good-review", "data-collection"]
        for skill_name in skill_names:
            skill_dir = ecommerce_dir / skill_name
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                print(f"✅ {skill_name}: {skill_file}")
            else:
                print(f"❌ {skill_name}: 缺失 SKILL.md")
    
    # 4. 测试 SkillsMiddleware
    print("\n[4] 测试 SkillsMiddleware:")
    try:
        backend = FilesystemBackend(root_dir=str(skills_dir))
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/base/",
                "/ecommerce/",
            ]
        )
        print("✅ SkillsMiddleware 创建成功")
    except Exception as e:
        print(f"❌ SkillsMiddleware 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 技能系统测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_skills_system()
    sys.exit(0 if success else 1)
