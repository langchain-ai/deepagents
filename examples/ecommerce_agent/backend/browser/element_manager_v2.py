from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database.models import DOMElement
from playwright.async_api import Page, Locator
import asyncio


class ElementSelector:
    """元素选择器"""
    
    def __init__(self, selector_type: str, value: str, description: str = ""):
        self.selector_type = selector_type
        self.value = value
        self.description = description
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "type": self.selector_type,
            "value": self.value,
            "description": self.description
        }
    
    @staticmethod
    def from_dict(data: Dict[str, str]) -> "ElementSelector":
        return ElementSelector(
            selector_type=data.get("type", "css"),
            value=data.get("value", ""),
            description=data.get("description", "")
        )


class SmartElementLocator:
    """智能元素定位器（支持多选择器降级）"""
    
    def __init__(self, selectors: List[ElementSelector]):
        self.selectors = selectors
        self.version = 1
        self.last_working_selector: Optional[ElementSelector] = None
    
    def add_selector(self, selector: ElementSelector):
        """添加备用选择器"""
        self.selectors.append(selector)
    
    async def find_element(self, page: Page) -> Optional[Locator]:
        """查找元素，尝试所有选择器"""
        for i, selector in enumerate(self.selectors):
            try:
                locator = await self._try_selector(page, selector)
                if locator:
                    self.last_working_selector = selector
                    return locator
            except Exception as e:
                print(f"选择器 {selector.selector_type}:{selector.value} 失败: {e}")
                continue
        
        return None
    
    async def _try_selector(self, page: Page, selector: ElementSelector) -> Optional[Locator]:
        """尝试使用单个选择器"""
        if selector.selector_type == "css":
            return page.locator(selector.value).first
        elif selector.selector_type == "xpath":
            return page.locator(f"xpath={selector.value}").first
        elif selector.selector_type == "text":
            return page.get_by_text(selector.value, exact=False).first
        elif selector.selector_type == "role":
            return page.get_by_role(selector.value).first
        elif selector.selector_type == "label":
            return page.get_by_label(selector.value).first
        
        return None
    
    async def click(self, page: Page, timeout: int = 5000):
        """点击元素"""
        locator = await self.find_element(page)
        if locator:
            await locator.click(timeout=timeout)
            return True
        return False
    
    async def fill(self, page: Page, text: str, timeout: int = 5000):
        """填写元素"""
        locator = await self.find_element(page)
        if locator:
            await locator.fill(text, timeout=timeout)
            return True
        return False
    
    async def get_text(self, page: Page, timeout: int = 5000) -> Optional[str]:
        """获取文本"""
        locator = await self.find_element(page)
        if locator:
            return await locator.text_content(timeout=timeout)
        return None


class ElementVersionManager:
    """元素版本管理器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_version(
        self,
        platform: str,
        page: str,
        name: str,
        selectors: List[Dict[str, str]],
        description: str = "",
        reason: str = ""
    ) -> DOMElement:
        """创建新版本"""
        # 查找现有元素
        existing = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name
        ).first()
        
        if existing:
            # 创建新版本
            new_element = DOMElement(
                platform=platform,
                page=page,
                name=name,
                selectors=selectors,
                description=description,
                version=existing.version + 1,
                is_active=True
            )
            self.db.add(new_element)
            
            # 停用旧版本
            existing.is_active = False
            self.db.commit()
            self.db.refresh(new_element)
            
            # 记录版本历史（可以扩展为单独的版本历史表）
            return new_element
        else:
            # 创建第一个版本
            new_element = DOMElement(
                platform=platform,
                page=page,
                name=name,
                selectors=selectors,
                description=description,
                version=1,
                is_active=True
            )
            self.db.add(new_element)
            self.db.commit()
            self.db.refresh(new_element)
            return new_element
    
    def rollback_to_version(
        self,
        platform: str,
        page: str,
        name: str,
        target_version: int
    ) -> Optional[DOMElement]:
        """回滚到指定版本"""
        # 查找目标版本
        target = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name,
            DOMElement.version == target_version
        ).first()
        
        if target:
            # 创建新版本复制目标版本
            new_element = DOMElement(
                platform=platform,
                page=page,
                name=name,
                selectors=target.selectors,
                description=target.description,
                version=target.version + 1,
                is_active=True
            )
            self.db.add(new_element)
            
            # 停用所有旧版本
            self.db.query(DOMElement).filter(
                DOMElement.platform == platform,
                DOMElement.page == page,
                DOMElement.name == name
            ).update({"is_active": False})
            
            self.db.commit()
            self.db.refresh(new_element)
            return new_element
        
        return None
    
    def get_version_history(
        self,
        platform: str,
        page: str,
        name: str
    ) -> List[Dict[str, Any]]:
        """获取版本历史"""
        elements = self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name
        ).order_by(DOMElement.version.desc()).all()
        
        return [
            {
                "version": elem.version,
                "selectors": elem.selectors,
                "description": elem.description,
                "is_active": elem.is_active,
                "created_at": elem.created_at
            }
            for elem in elements
        ]
    
    def get_current_version(
        self,
        platform: str,
        page: str,
        name: str
    ) -> Optional[DOMElement]:
        """获取当前活跃版本"""
        return self.db.query(DOMElement).filter(
            DOMElement.platform == platform,
            DOMElement.page == page,
            DOMElement.name == name,
            DOMElement.is_active == True
        ).first()


class ElementTestTool:
    """元素测试工具"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def test_element(
        self,
        platform: str,
        page: str,
        name: str,
        page_obj: Page
    ) -> Dict[str, Any]:
        """测试元素是否能被找到"""
        from backend.browser.elements import ElementManager
        
        element_manager = ElementManager(self.db)
        element_data = element_manager.get_element(platform, page, name)
        
        if not element_data:
            return {
                "success": False,
                "message": f"元素不存在: {platform}-{page}-{name}"
            }
        
        selectors = element_data.get("selectors", [])
        if not selectors:
            return {
                "success": False,
                "message": "元素没有配置选择器"
            }
        
        # 尝试每个选择器
        results = []
        for selector_data in selectors:
            selector = ElementSelector.from_dict(selector_data)
            try:
                locator = await self._try_selector(page_obj, selector)
                if locator:
                    is_visible = await locator.is_visible()
                    results.append({
                        "selector_type": selector.selector_type,
                        "selector_value": selector.value,
                        "success": True,
                        "visible": is_visible,
                        "message": "找到元素"
                    })
                else:
                    results.append({
                        "selector_type": selector.selector_type,
                        "selector_value": selector.value,
                        "success": False,
                        "message": "未找到元素"
                    })
            except Exception as e:
                results.append({
                    "selector_type": selector.selector_type,
                    "selector_value": selector.value,
                    "success": False,
                    "message": f"错误: {str(e)}"
                })
        
        # 返回第一个成功的
        working = next((r for r in results if r["success"]), None)
        
        return {
            "success": working is not None,
            "element_name": name,
            "test_results": results,
            "working_selector": working,
            "message": "测试完成" if working else "所有选择器都失败"
        }
    
    async def _try_selector(self, page: Page, selector: ElementSelector) -> Optional[Locator]:
        """尝试使用单个选择器"""
        try:
            if selector.selector_type == "css":
                locator = page.locator(selector.value).first
                await locator.wait_for(state="attached", timeout=1000)
                return locator
            elif selector.selector_type == "xpath":
                locator = page.locator(f"xpath={selector.value}").first
                await locator.wait_for(state="attached", timeout=1000)
                return locator
        except:
            return None
        
        return None
