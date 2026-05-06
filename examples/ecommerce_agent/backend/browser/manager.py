import asyncio
import uuid
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import Store, BrowserTab
from backend.browser.anti_detect import anti_detect


class BrowserManager:
    """浏览器管理器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.contexts: Dict[int, BrowserContext] = {}  # store_id -> context
        self.tabs: Dict[int, Page] = {}  # store_id -> page
        self.tab_info: Dict[int, Dict[str, Any]] = {}
    
    async def start(self):
        """启动浏览器"""
        if self.playwright is None:
            self.playwright = await async_playwright().start()
            
            launch_options = {
                "headless": settings.HEADLESS,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                ]
            }
            self.browser = await self.playwright.chromium.launch(**launch_options)
    
    async def stop(self):
        """停止浏览器"""
        for store_id in list(self.tabs.keys()):
            await self.close_tab(store_id)
        
        for store_id in list(self.contexts.keys()):
            await self.close_context(store_id)
        
        if self.browser:
            await self.browser.close()
            self.browser = None
        
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
    
    async def get_context(self, store: Store) -> BrowserContext:
        """获取或创建浏览器上下文（Profile）"""
        if store.id in self.contexts:
            return self.contexts[store.id]
        
        # 创建 Profile 目录
        profile_path = settings.PROFILE_DIR / f"store_{store.id}"
        profile_path.mkdir(parents=True, exist_ok=True)
        
        # 加载或生成指纹
        fingerprint = anti_detect.load_fingerprint(store.id, profile_path)
        if fingerprint is None:
            fingerprint = anti_detect.generate_fingerprint()
            anti_detect.save_fingerprint(store.id, fingerprint, profile_path)
        
        # 创建上下文
        context_options = {
            "user_data_dir": str(profile_path),
            "viewport": fingerprint["viewport"],
            "user_agent": fingerprint["user_agent"],
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
            "permissions": ["geolocation"],
            "geolocation": {"latitude": 39.9042, "longitude": 116.4074},  # 北京
        }
        
        context = await self.browser.new_context(**context_options)
        
        # 应用 stealth 插件
        await self._apply_stealth(context)
        
        self.contexts[store.id] = context
        return context
    
    async def _apply_stealth(self, context: BrowserContext):
        """应用 stealth 技术"""
        # 注入脚本隐藏自动化特征
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en']
            });
            
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
    
    async def get_tab(self, store: Store) -> Page:
        """获取或创建 Tab"""
        if store.id in self.tabs:
            page = self.tabs[store.id]
            # 更新最后使用时间
            await self._update_tab_last_used(store.id)
            return page
        
        context = await self.get_context(store)
        
        # 创建新页面
        page = await context.new_page()
        tab_id = str(uuid.uuid4())
        
        # 保存 Tab 信息
        self.tabs[store.id] = page
        self.tab_info[store.id] = {
            "tab_id": tab_id,
            "created_at": datetime.utcnow(),
            "last_used_at": datetime.utcnow()
        }
        
        # 保存到数据库
        db_tab = BrowserTab(
            store_id=store.id,
            tab_id=tab_id,
            url="",
            is_active=True
        )
        self.db.add(db_tab)
        self.db.commit()
        
        return page
    
    async def close_tab(self, store_id: int):
        """关闭 Tab"""
        if store_id in self.tabs:
            try:
                await self.tabs[store_id].close()
            except:
                pass
            del self.tabs[store_id]
        
        if store_id in self.tab_info:
            del self.tab_info[store_id]
        
        # 更新数据库
        self.db.query(BrowserTab).filter(
            BrowserTab.store_id == store_id
        ).update({"is_active": False})
        self.db.commit()
    
    async def close_context(self, store_id: int):
        """关闭上下文"""
        if store_id in self.contexts:
            try:
                await self.contexts[store_id].close()
            except:
                pass
            del self.contexts[store_id]
    
    async def _update_tab_last_used(self, store_id: int):
        """更新 Tab 最后使用时间"""
        if store_id in self.tab_info:
            self.tab_info[store_id]["last_used_at"] = datetime.utcnow()
        
        self.db.query(BrowserTab).filter(
            BrowserTab.store_id == store_id
        ).update({"last_used_at": datetime.utcnow()})
        self.db.commit()
    
    async def cleanup_idle_tabs(self):
        """清理闲置 Tab"""
        now = datetime.utcnow()
        idle_timeout = timedelta(minutes=settings.TAB_IDLE_TIMEOUT_MINUTES)
        
        to_close = []
        for store_id, info in self.tab_info.items():
            if now - info["last_used_at"] > idle_timeout:
                to_close.append(store_id)
        
        for store_id in to_close:
            await self.close_tab(store_id)
    
    async def take_screenshot(self, store_id: int, name: str = "") -> str:
        """截图"""
        if store_id not in self.tabs:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{store_id}_{timestamp}_{name}.png"
        filepath = settings.SCREENSHOT_DIR / filename
        
        await self.tabs[store_id].screenshot(path=str(filepath))
        return str(filepath)
    
    async def navigate(self, store: Store, url: str) -> Page:
        """导航到 URL"""
        page = await self.get_tab(store)
        await page.goto(url, wait_until="networkidle")
        
        # 更新 Tab URL
        self.db.query(BrowserTab).filter(
            BrowserTab.store_id == store.id
        ).update({"url": url})
        self.db.commit()
        
        return page


# 全局浏览器管理器实例
_browser_manager: Optional[BrowserManager] = None


def get_browser_manager(db: Session) -> BrowserManager:
    """获取浏览器管理器"""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager(db)
    return _browser_manager
