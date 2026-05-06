import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import DOMElement


class ElementManager:
    """DOM 元素管理器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_dir = settings.CONFIG_DIR / "elements"
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        for platform in ["douyin", "pinduoduo", "taobao"]:
            platform_dir = self.config_dir / platform
            platform_dir.mkdir(parents=True, exist_ok=True)
    
    def get_element(
        self,
        platform: str,
        page: str,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """获取元素配置"""
        # 先从数据库查找
        element = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name,
            DOMElement.is_active == True
        ).first()
        
        if element:
            return {
                "platform": element.platform,
                "page": element.page,
                "name": element.name,
                "selectors": element.selectors,
                "description": element.description,
                "version": element.version
            }
        
        # 从文件查找
        return self._load_element_from_file(platform, page, name)
    
    def _load_element_from_file(
        self,
        platform: str,
        page: str,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """从文件加载元素配置"""
        file_path = self.config_dir / platform / f"{page}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if name in data.get("elements", {}):
                    return {
                        "platform": platform,
                        "page": page,
                        "name": name,
                        "selectors": data["elements"][name]["selectors"],
                        "description": data["elements"][name].get("description", ""),
                        "version": data["elements"][name].get("version", 1)
                    }
        return None
    
    def save_element(
        self,
        platform: str,
        page: str,
        name: str,
        selectors: List[Dict[str, str]],
        description: str = ""
    ) -> DOMElement:
        """保存元素配置"""
        # 查找现有元素
        existing = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name
        ).first()
        
        if existing:
            existing.selectors = selectors
            existing.description = description
            existing.version += 1
            existing.is_active = True
            element = existing
        else:
            element = DOMElement(
                platform=platform,
                page=page,
                name=name,
                selectors=selectors,
                description=description,
                version=1
            )
            self.db.add(element)
        
        self.db.commit()
        self.db.refresh(element)
        return element
    
    def get_page_elements(
        self,
        platform: str,
        page: str
    ) -> List[Dict[str, Any]]:
        """获取页面所有元素"""
        elements = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.is_active == True
        ).all()
        
        result = []
        for elem in elements:
            result.append({
                "platform": elem.platform,
                "page": elem.page,
                "name": elem.name,
                "selectors": elem.selectors,
                "description": elem.description,
                "version": elem.version
            })
        return result
    
    def load_platform_config(self, platform: str) -> Dict[str, Any]:
        """加载平台配置"""
        file_path = self.config_dir / f"{platform}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def import_from_file(self, file_path: str):
        """从文件导入元素配置"""
        path = Path(file_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                platform = data.get("platform")
                page = data.get("page")
                if platform and page:
                    for name, elem_data in data.get("elements", {}).items():
                        self.save_element(
                            platform=platform,
                            page=page,
                            name=name,
                            selectors=elem_data.get("selectors", []),
                            description=elem_data.get("description", "")
                        )
    
    def export_to_file(self, platform: str, page: str, output_path: str):
        """导出元素配置到文件"""
        elements = self.get_page_elements(platform, page)
        data = {
            "platform": platform,
            "page": page,
            "elements": {
                elem["name"]: {
                    "selectors": elem["selectors"],
                    "description": elem["description"],
                    "version": elem["version"]
                }
                for elem in elements
            }
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 示例元素选择器
DEFAULT_ELEMENTS = {
    "douyin": {
        "login": {
            "username_input": {
                "selectors": [
                    {"type": "css", "value": "input[name='username']"},
                    {"type": "xpath", "value": "//input[@name='username']"}
                ],
                "description": "用户名输入框"
            },
            "password_input": {
                "selectors": [
                    {"type": "css", "value": "input[name='password']"},
                    {"type": "xpath", "value": "//input[@name='password']"}
                ],
                "description": "密码输入框"
            },
            "login_button": {
                "selectors": [
                    {"type": "css", "value": "button[type='submit']"},
                    {"type": "xpath", "value": "//button[@type='submit']"}
                ],
                "description": "登录按钮"
            }
        },
        "publish": {
            "publish_button": {
                "selectors": [
                    {"type": "css", "value": ".publish-btn"},
                    {"type": "text", "value": "发布商品"}
                ],
                "description": "发布商品按钮"
            },
            "title_input": {
                "selectors": [
                    {"type": "css", "value": "input[name='title']"}
                ],
                "description": "商品标题输入框"
            }
        }
    },
    "pinduoduo": {
        "login": {
            "username_input": {
                "selectors": [{"type": "css", "value": "#username"}],
                "description": "用户名输入框"
            }
        }
    },
    "taobao": {
        "login": {
            "username_input": {
                "selectors": [{"type": "css", "value": "#fm-login-id"}],
                "description": "用户名输入框"
            }
        }
    }
}


def init_default_elements(db: Session):
    """初始化默认元素配置"""
    manager = ElementManager(db)
    for platform, pages in DEFAULT_ELEMENTS.items():
        for page, elements in pages.items():
            for name, elem_data in elements.items():
                manager.save_element(
                    platform=platform,
                    page=page,
                    name=name,
                    selectors=elem_data.get("selectors", []),
                    description=elem_data.get("description", "")
                )
