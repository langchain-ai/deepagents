from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import Task
import json


class CheckpointManager:
    """断点续跑管理器"""
    
    def __init__(self, db: Session, task: Task):
        self.db = db
        self.task = task
    
    def save_checkpoint(
        self,
        step_index: int,
        step_name: str,
        step_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """保存断点"""
        checkpoint = {
            "step_index": step_index,
            "step_name": step_name,
            "step_data": step_data,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": self.task.id,
            "store_id": self.task.store_id,
            "progress": self.task.progress,
            "current_step": self.task.current_step
        }
        
        # 保存到任务记录
        self.task.checkpoint = checkpoint
        self.task.completed_steps = step_index
        self.task.progress = int((step_index / self.task.total_steps) * 100) if self.task.total_steps > 0 else 0
        self.db.commit()
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载断点"""
        return self.task.checkpoint
    
    def has_checkpoint(self) -> bool:
        """检查是否有断点"""
        return self.task.checkpoint is not None
    
    def clear_checkpoint(self):
        """清除断点"""
        self.task.checkpoint = None
        self.db.commit()
    
    async def resume_from_checkpoint(self) -> Dict[str, Any]:
        """从断点恢复"""
        checkpoint = self.load_checkpoint()
        
        if not checkpoint:
            return {
                "status": "no_checkpoint",
                "message": "没有可用的断点"
            }
        
        # 恢复任务状态
        self.task.status = "running"
        self.task.started_at = datetime.utcnow()
        self.db.commit()
        
        return {
            "status": "resuming",
            "checkpoint": checkpoint,
            "step_index": checkpoint["step_index"],
            "step_name": checkpoint["step_name"],
            "step_data": checkpoint["step_data"]
        }
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取断点摘要"""
        checkpoint = self.load_checkpoint()
        
        if not checkpoint:
            return {
                "has_checkpoint": False,
                "message": "没有断点"
            }
        
        return {
            "has_checkpoint": True,
            "step_index": checkpoint.get("step_index", 0),
            "step_name": checkpoint.get("step_name", ""),
            "progress": checkpoint.get("progress", 0),
            "timestamp": checkpoint.get("timestamp"),
            "metadata": checkpoint.get("metadata", {})
        }


class StepTracker:
    """步骤跟踪器"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, total_steps: int):
        self.checkpoint_manager = checkpoint_manager
        self.total_steps = total_steps
        self.current_step_index = 0
        self.step_history: List[Dict[str, Any]] = []
        self.checkpoint_interval = 5  # 每5步保存一次断点
    
    def start_step(self, step_name: str, step_data: Optional[Dict[str, Any]] = None):
        """开始步骤"""
        self.current_step_index += 1
        
        step_info = {
            "index": self.current_step_index,
            "name": step_name,
            "start_time": datetime.utcnow().isoformat(),
            "data": step_data or {}
        }
        
        self.step_history.append(step_info)
        
        # 更新任务进度
        self.checkpoint_manager.task.current_step = step_name
        self.checkpoint_manager.task.completed_steps = self.current_step_index - 1
        self.checkpoint_manager.db.commit()
        
        return step_info
    
    def complete_step(self, result: Optional[Dict[str, Any]] = None):
        """完成步骤"""
        if not self.step_history:
            return
        
        current_step = self.step_history[-1]
        current_step["end_time"] = datetime.utcnow().isoformat()
        current_step["status"] = "completed"
        current_step["result"] = result
        
        # 计算进度
        progress = int((self.current_step_index / self.total_steps) * 100)
        self.checkpoint_manager.task.progress = progress
        self.checkpoint_manager.db.commit()
        
        # 定期保存断点
        if self.current_step_index % self.checkpoint_interval == 0:
            self.save_checkpoint()
    
    def fail_step(self, error: str):
        """步骤失败"""
        if not self.step_history:
            return
        
        current_step = self.step_history[-1]
        current_step["end_time"] = datetime.utcnow().isoformat()
        current_step["status"] = "failed"
        current_step["error"] = error
        
        # 保存断点
        self.save_checkpoint()
    
    def save_checkpoint(self):
        """保存断点"""
        if not self.step_history:
            return
        
        current_step = self.step_history[-1]
        
        self.checkpoint_manager.save_checkpoint(
            step_index=self.current_step_index,
            step_name=current_step["name"],
            step_data={
                "step_history": self.step_history,
                "current_step_data": current_step.get("data", {})
            },
            metadata={
                "total_steps": self.total_steps,
                "progress": self.checkpoint_manager.task.progress
            }
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """获取进度"""
        return {
            "current_step": self.current_step_index,
            "total_steps": self.total_steps,
            "progress_percent": int((self.current_step_index / self.total_steps) * 100),
            "completed_steps": [s for s in self.step_history if s.get("status") == "completed"],
            "failed_step": next((s for s in self.step_history if s.get("status") == "failed"), None)
        }
    
    def get_remaining_steps(self) -> List[Dict[str, Any]]:
        """获取剩余步骤"""
        completed_indices = [s["index"] for s in self.step_history if s.get("status") == "completed"]
        failed_index = next((s["index"] for s in self.step_history if s.get("status") == "failed"), None)
        
        remaining = []
        for i in range(1, self.total_steps + 1):
            if i > max(completed_indices or [0]) and (failed_index is None or i >= failed_index):
                remaining.append({"index": i})
        
        return remaining
