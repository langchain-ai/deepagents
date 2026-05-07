from typing import Dict, Any, Optional, Callable
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import Task, OperationLog
import asyncio


class ExceptionLevel:
    """异常等级"""
    LIGHT = "light"       # 轻度异常：重试当前步骤
    MEDIUM = "medium"     # 中度异常：刷新页面，重试任务
    HEAVY = "heavy"       # 重度异常：关闭Tab，重新打开


class GlobalExceptionHandler:
    """全局异常处理器"""
    
    def __init__(self, db: Session, task: Task):
        self.db = db
        self.task = task
        self.retry_count = 0
        self.max_light_retries = 3
        self.max_medium_retries = 2
        self.error_history = []
        self.exception_callbacks = []
    
    def register_callback(self, callback: Callable):
        """注册异常回调"""
        self.exception_callbacks.append(callback)
    
    async def handle_exception(
        self,
        exception: Exception,
        level: str = ExceptionLevel.LIGHT,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理异常"""
        error_info = {
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "level": level,
            "timestamp": datetime.utcnow(),
            "context": context or {},
            "retry_count": self.retry_count
        }
        
        self.error_history.append(error_info)
        
        # 记录错误日志
        await self._log_exception(error_info)
        
        # 根据异常等级执行恢复策略
        if level == ExceptionLevel.LIGHT:
            return await self._handle_light_exception(exception, error_info)
        elif level == ExceptionLevel.MEDIUM:
            return await self._handle_medium_exception(exception, error_info)
        elif level == ExceptionLevel.HEAVY:
            return await self._handle_heavy_exception(exception, error_info)
        
        return {"status": "failed", "action": "unknown_level"}
    
    async def _handle_light_exception(
        self,
        exception: Exception,
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理轻度异常：重试当前步骤"""
        if self.retry_count >= self.max_light_retries:
            # 超过重试次数，升级为中度异常
            return await self.handle_exception(
                exception,
                level=ExceptionLevel.MEDIUM,
                context={"upgraded_from": "light", "retry_count": self.retry_count}
            )
        
        self.retry_count += 1
        
        # 等待一段时间后重试
        wait_time = 2 ** self.retry_count  # 指数退避
        await asyncio.sleep(wait_time)
        
        return {
            "status": "retrying",
            "action": "retry_step",
            "retry_count": self.retry_count,
            "wait_time": wait_time
        }
    
    async def _handle_medium_exception(
        self,
        exception: Exception,
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理中度异常：刷新页面，重试任务"""
        if self.retry_count >= self.max_medium_retries:
            # 超过重试次数，升级为重度异常
            return await self.handle_exception(
                exception,
                level=ExceptionLevel.HEAVY,
                context={"upgraded_from": "medium", "retry_count": self.retry_count}
            )
        
        self.retry_count += 1
        
        # 回调执行刷新操作
        for callback in self.exception_callbacks:
            if hasattr(callback, '__name__') and 'refresh' in callback.__name__:
                await callback()
        
        return {
            "status": "retrying",
            "action": "refresh_and_retry",
            "retry_count": self.retry_count
        }
    
    async def _handle_heavy_exception(
        self,
        exception: Exception,
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理重度异常：关闭Tab，重新打开Profile"""
        # 更新任务状态为失败
        self.task.status = "failed"
        self.task.error_message = str(exception)
        self.db.commit()
        
        # 回调执行关闭和重启操作
        for callback in self.exception_callbacks:
            if hasattr(callback, '__name__') and 'restart' in callback.__name__:
                await callback()
        
        return {
            "status": "failed",
            "action": "restart_profile",
            "error": str(exception)
        }
    
    async def _log_exception(self, error_info: Dict[str, Any]):
        """记录异常日志"""
        log = OperationLog(
            store_id=self.task.store_id,
            task_id=self.task.id,
            operation_type="exception",
            status="failed",
            message=f"[{error_info['level']}] {error_info['message']}",
            error_detail=str(error_info)
        )
        self.db.add(log)
        self.db.commit()
    
    def should_pause_task(self) -> bool:
        """检查是否应该暂停任务"""
        return len([e for e in self.error_history if e["level"] == ExceptionLevel.HEAVY]) >= 5
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return {
            "total_errors": len(self.error_history),
            "light_errors": len([e for e in self.error_history if e["level"] == ExceptionLevel.LIGHT]),
            "medium_errors": len([e for e in self.error_history if e["level"] == ExceptionLevel.MEDIUM]),
            "heavy_errors": len([e for e in self.error_history if e["level"] == ExceptionLevel.HEAVY]),
            "should_pause": self.should_pause_task(),
            "recent_errors": self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }


class ExceptionHandlerFactory:
    """异常处理器工厂"""
    
    @staticmethod
    def create_handler(db: Session, task: Task) -> GlobalExceptionHandler:
        """创建异常处理器"""
        return GlobalExceptionHandler(db, task)
    
    @staticmethod
    def is_recoverable(exception: Exception) -> bool:
        """判断异常是否可恢复"""
        # 可以恢复的异常类型
        recoverable_types = [
            "TimeoutError",
            "NetworkError",
            "ElementNotFoundError",
            "PageLoadError",
            "BrowserCrashError"
        ]
        
        exception_type = type(exception).__name__
        return exception_type in recoverable_types or "timeout" in str(exception).lower()
    
    @staticmethod
    def determine_exception_level(exception: Exception) -> str:
        """判断异常等级"""
        error_msg = str(exception).lower()
        
        if "timeout" in error_msg or "not found" in error_msg:
            return ExceptionLevel.LIGHT
        elif "network" in error_msg or "load" in error_msg:
            return ExceptionLevel.MEDIUM
        elif "crash" in error_msg or "fatal" in error_msg:
            return ExceptionLevel.HEAVY
        
        return ExceptionLevel.LIGHT


exception_handler_factory = ExceptionHandlerFactory()
