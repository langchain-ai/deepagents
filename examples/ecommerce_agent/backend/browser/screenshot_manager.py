from typing import Optional, List
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import OperationLog
import asyncio


class ScreenshotManager:
    """截图管理器"""
    
    def __init__(self, db: Session, store_id: int, task_id: Optional[int] = None):
        self.db = db
        self.store_id = store_id
        self.task_id = task_id
        self.screenshot_dir = settings.SCREENSHOT_DIR
        self.retention_days = settings.SCREENSHOT_RETENTION_DAYS
    
    async def capture_screenshot(
        self,
        name: str = "auto",
        capture_type: str = "step"
    ) -> str:
        """捕获截图"""
        from backend.browser.manager import get_browser_manager
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{capture_type}_{self.store_id}_{name}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        
        try:
            browser_manager = get_browser_manager(self.db)
            page = browser_manager.tabs.get(self.store_id)
            
            if page:
                await page.screenshot(path=str(filepath))
                
                # 记录到日志
                await self._log_screenshot(filename, name, capture_type)
                
                return str(filepath)
            else:
                return ""
        except Exception as e:
            print(f"截图失败: {e}")
            return ""
    
    async def capture_key_step_screenshot(self, step_name: str) -> str:
        """捕获关键步骤截图"""
        return await self.capture_screenshot(
            name=step_name,
            capture_type="key_step"
        )
    
    async def capture_error_screenshot(self, error_msg: str) -> str:
        """捕获错误截图"""
        return await self.capture_screenshot(
            name=error_msg[:20],  # 截取前20个字符作为文件名
            capture_type="error"
        )
    
    async def capture_periodic_screenshot(self) -> str:
        """定期自动截图"""
        return await self.capture_screenshot(
            name="periodic",
            capture_type="periodic"
        )
    
    async def _log_screenshot(
        self,
        filename: str,
        name: str,
        capture_type: str
    ):
        """记录截图日志"""
        screenshot_path = str(self.screenshot_dir / filename)
        
        log = OperationLog(
            store_id=self.store_id,
            task_id=self.task_id,
            operation_type=f"screenshot_{capture_type}",
            status="success",
            message=f"截图: {name}",
            screenshot_path=screenshot_path
        )
        self.db.add(log)
        self.db.commit()
    
    def cleanup_old_screenshots(self):
        """清理过期截图"""
        cutoff_date = datetime.now().timestamp() - (self.retention_days * 86400)
        
        for file in self.screenshot_dir.glob(f"*{self.store_id}*.png"):
            if file.stat().st_mtime < cutoff_date:
                try:
                    file.unlink()
                    print(f"已删除过期截图: {file.name}")
                except Exception as e:
                    print(f"删除截图失败: {e}")
    
    def get_screenshots(self, limit: int = 50) -> List[Path]:
        """获取截图列表"""
        screenshots = sorted(
            self.screenshot_dir.glob(f"*{self.store_id}*.png"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        return screenshots[:limit]


class RecordingManager:
    """录屏管理器"""
    
    def __init__(self, db: Session, store_id: int, task_id: Optional[int] = None):
        self.db = db
        self.store_id = store_id
        self.task_id = task_id
        self.recording_dir = settings.RECORDING_DIR
        self.recording_duration = 30  # 30秒
        self.is_recording = False
        self.current_recording: Optional[str] = None
    
    async def start_recording(self) -> str:
        """开始录屏"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{self.store_id}_{timestamp}.webm"
        filepath = self.recording_dir / filename
        
        self.is_recording = True
        self.current_recording = str(filepath)
        
        return self.current_recording
    
    async def stop_recording(self) -> Optional[str]:
        """停止录屏"""
        if self.is_recording and self.current_recording:
            recording_path = self.current_recording
            self.is_recording = False
            self.current_recording = None
            
            # 记录日志
            log = OperationLog(
                store_id=self.store_id,
                task_id=self.task_id,
                operation_type="recording",
                status="success",
                message="录屏完成",
                recording_path=recording_path
            )
            self.db.add(log)
            self.db.commit()
            
            return recording_path
        
        return None
    
    async def capture_error_recording(self, duration: int = 30) -> Optional[str]:
        """捕获错误发生时的录屏"""
        await self.start_recording()
        
        # 等待指定时长
        await asyncio.sleep(duration)
        
        return await self.stop_recording()
    
    def cleanup_old_recordings(self):
        """清理过期录屏"""
        from backend.config import settings
        retention_days = settings.RECORDING_RETENTION_DAYS
        cutoff_date = datetime.now().timestamp() - (retention_days * 86400)
        
        for file in self.recording_dir.glob(f"*{self.store_id}*.webm"):
            if file.stat().st_mtime < cutoff_date:
                try:
                    file.unlink()
                    print(f"已删除过期录屏: {file.name}")
                except Exception as e:
                    print(f"删除录屏失败: {e}")
