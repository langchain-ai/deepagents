import asyncio
from typing import Dict, Callable, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import BrowserTab


class TabRecycler:
    """闲置 Tab 回收器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.idle_timeout_minutes = settings.TAB_IDLE_TIMEOUT_MINUTES
        self.check_interval_seconds = 60  # 每分钟检查一次
        self.is_running = False
        self.recycler_task: Optional[asyncio.Task] = None
        self.active_tabs: Dict[int, datetime] = {}
    
    async def start(self):
        """启动回收器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.recycler_task = asyncio.create_task(self._recycle_loop())
        print(f"Tab 回收器已启动，超时时间: {self.idle_timeout_minutes} 分钟")
    
    async def stop(self):
        """停止回收器"""
        self.is_running = False
        if self.recycler_task:
            self.recycler_task.cancel()
            try:
                await self.recycler_task
            except asyncio.CancelledError:
                pass
        print("Tab 回收器已停止")
    
    async def _recycle_loop(self):
        """回收循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                await self._check_and_recycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"回收循环错误: {e}")
    
    async def _check_and_recycle(self):
        """检查并回收闲置 Tab"""
        now = datetime.utcnow()
        idle_threshold = now - timedelta(minutes=self.idle_timeout_minutes)
        
        # 从数据库获取所有活跃 Tab
        tabs = self.db.query(BrowserTab).filter(
            BrowserTab.is_active == True
        ).all()
        
        recycled_count = 0
        for tab in tabs:
            if tab.last_used_at and tab.last_used_at < idle_threshold:
                # 检查内存中的 Tab 是否还在使用
                if tab.store_id not in self.active_tabs:
                    continue
                
                last_used = self.active_tabs.get(tab.store_id)
                if last_used and last_used < idle_threshold:
                    # 回收 Tab
                    await self._recycle_tab(tab.store_id)
                    recycled_count += 1
        
        if recycled_count > 0:
            print(f"已回收 {recycled_count} 个闲置 Tab")
    
    async def _recycle_tab(self, store_id: int):
        """回收单个 Tab"""
        from backend.browser.manager import get_browser_manager
        
        try:
            browser_manager = get_browser_manager(self.db)
            await browser_manager.close_tab(store_id)
            
            # 更新数据库
            self.db.query(BrowserTab).filter(
                BrowserTab.store_id == store_id
            ).update({"is_active": False})
            self.db.commit()
            
            # 从活跃列表移除
            if store_id in self.active_tabs:
                del self.active_tabs[store_id]
            
            print(f"已回收店铺 {store_id} 的闲置 Tab")
        except Exception as e:
            print(f"回收 Tab 失败: {e}")
    
    def mark_tab_active(self, store_id: int):
        """标记 Tab 为活跃"""
        self.active_tabs[store_id] = datetime.utcnow()
    
    def mark_tab_idle(self, store_id: int):
        """标记 Tab 为闲置"""
        if store_id in self.active_tabs:
            del self.active_tabs[store_id]
    
    async def force_recycle(self, store_id: int):
        """强制回收指定 Tab"""
        await self._recycle_tab(store_id)
    
    def get_active_tab_count(self) -> int:
        """获取活跃 Tab 数量"""
        return len(self.active_tabs)
    
    def get_idle_tabs(self) -> list:
        """获取闲置 Tab 列表"""
        now = datetime.utcnow()
        idle_threshold = now - timedelta(minutes=self.idle_timeout_minutes)
        
        idle_tabs = []
        for store_id, last_used in self.active_tabs.items():
            if last_used < idle_threshold:
                idle_tabs.append({
                    "store_id": store_id,
                    "last_used": last_used,
                    "idle_time_minutes": int((now - last_used).total_seconds() / 60)
                })
        
        return idle_tabs
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "active_tabs": len(self.active_tabs),
            "idle_tabs": len(self.get_idle_tabs()),
            "total_tabs": self.db.query(BrowserTab).filter(
                BrowserTab.is_active == True
            ).count(),
            "idle_timeout_minutes": self.idle_timeout_minutes
        }


class TabPool:
    """Tab 连接池"""
    
    def __init__(self, db: Session, max_tabs_per_store: int = 1):
        self.db = db
        self.max_tabs_per_store = max_tabs_per_store
        self.tab_pools: Dict[int, list] = {}  # store_id -> [page1, page2]
    
    async def get_tab(self, store_id: int):
        """获取 Tab"""
        from backend.browser.manager import get_browser_manager
        
        if store_id not in self.tab_pools:
            self.tab_pools[store_id] = []
        
        if self.tab_pools[store_id]:
            # 返回池中的 Tab
            return self.tab_pools[store_id].pop()
        
        # 创建新 Tab
        browser_manager = get_browser_manager(self.db)
        store = self.db.query(Store).get(store_id)
        if store:
            page = await browser_manager.get_tab(store)
            return page
        
        return None
    
    async def return_tab(self, store_id: int, page):
        """归还 Tab"""
        if store_id not in self.tab_pools:
            self.tab_pools[store_id] = []
        
        if len(self.tab_pools[store_id]) < self.max_tabs_per_store:
            self.tab_pools[store_id].append(page)
        else:
            # 关闭多余的 Tab
            try:
                await page.close()
            except:
                pass
    
    async def close_all(self):
        """关闭所有 Tab"""
        from backend.browser.manager import get_browser_manager
        
        browser_manager = get_browser_manager(self.db)
        
        for store_id, pages in self.tab_pools.items():
            for page in pages:
                try:
                    await page.close()
                except:
                    pass
        
        self.tab_pools.clear()


# 全局实例
_tab_recycler: Optional[TabRecycler] = None


def get_tab_recycler(db: Session) -> TabRecycler:
    """获取 Tab 回收器"""
    global _tab_recycler
    if _tab_recycler is None:
        _tab_recycler = TabRecycler(db)
    return _tab_recycler


# 导入 Store
from backend.database.models import Store
