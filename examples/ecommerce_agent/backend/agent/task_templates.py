from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database.models import Task, ScheduledTask


class TaskTemplate:
    """任务模板"""
    
    def __init__(
        self,
        name: str,
        task_type: str,
        description: str = "",
        steps: List[Dict[str, Any]] = None,
        schedule: Optional[str] = None  # cron 表达式
    ):
        self.name = name
        self.task_type = task_type
        self.description = description
        self.steps = steps or []
        self.schedule = schedule
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "description": self.description,
            "steps": self.steps,
            "schedule": self.schedule
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskTemplate":
        return TaskTemplate(
            name=data["name"],
            task_type=data["task_type"],
            description=data.get("description", ""),
            steps=data.get("steps", []),
            schedule=data.get("schedule")
        )


class TaskTemplateManager:
    """任务模板管理器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_template(
        self,
        name: str,
        task_type: str,
        description: str = "",
        steps: List[Dict[str, Any]] = None,
        schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建任务模板"""
        template = TaskTemplate(
            name=name,
            task_type=task_type,
            description=description,
            steps=steps,
            schedule=schedule
        )
        
        # 这里可以存储到数据库或文件
        # 暂时使用内存存储
        if not hasattr(self, "_templates"):
            self._templates = {}
        
        template_id = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._templates[template_id] = template
        
        return {
            "id": template_id,
            **template.to_dict()
        }
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板"""
        if hasattr(self, "_templates") and template_id in self._templates:
            template = self._templates[template_id]
            return {
                "id": template_id,
                **template.to_dict()
            }
        return None
    
    def list_templates(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有模板"""
        if not hasattr(self, "_templates"):
            return []
        
        templates = []
        for tid, template in self._templates.items():
            if task_type is None or template.task_type == task_type:
                templates.append({
                    "id": tid,
                    **template.to_dict()
                })
        
        return templates
    
    def apply_template_to_store(
        self,
        template_id: str,
        store_id: int
    ) -> Optional[Task]:
        """应用模板到店铺"""
        template_data = self.get_template(template_id)
        if not template_data:
            return None
        
        # 创建任务
        task = Task(
            store_id=store_id,
            task_type=template_data["task_type"],
            name=f"{template_data['name']}-{store_id}",
            status="pending",
            total_steps=len(template_data.get("steps", []))
        )
        self.db.add(task)
        self.db.commit()
        
        return task


class BatchOperationManager:
    """批量操作管理器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.batch_tasks: Dict[str, List[int]] = {}  # batch_id -> [task_ids]
    
    def batch_create_tasks(
        self,
        store_ids: List[int],
        task_type: str,
        name: Optional[str] = None
    ) -> List[int]:
        """批量创建任务"""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task_ids = []
        
        for store_id in store_ids:
            task = Task(
                store_id=store_id,
                task_type=task_type,
                name=name or f"{task_type}-{store_id}",
                status="pending",
                total_steps=100
            )
            self.db.add(task)
            self.db.commit()
            task_ids.append(task.id)
        
        self.batch_tasks[batch_id] = task_ids
        return task_ids
    
    def batch_start_tasks(self, task_ids: List[int]):
        """批量启动任务"""
        self.db.query(Task).filter(
            Task.id.in_(task_ids)
        ).update(
            {Task.status: "running", Task.started_at: datetime.utcnow()},
            synchronize_session=False
        )
        self.db.commit()
    
    def batch_stop_tasks(self, task_ids: List[int]):
        """批量停止任务"""
        self.db.query(Task).filter(
            Task.id.in_(task_ids)
        ).update(
            {Task.status: "paused"},
            synchronize_session=False
        )
        self.db.commit()
    
    def batch_delete_tasks(self, task_ids: List[int]):
        """批量删除任务"""
        self.db.query(Task).filter(
            Task.id.in_(task_ids)
        ).delete(synchronize_session=False)
        self.db.commit()
    
    def get_batch_status(self, task_ids: List[int]) -> Dict[str, Any]:
        """获取批量任务状态"""
        tasks = self.db.query(Task).filter(
            Task.id.in_(task_ids)
        ).all()
        
        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "paused": 0
        }
        
        for task in tasks:
            if task.status in status_counts:
                status_counts[task.status] += 1
        
        return {
            "total": len(tasks),
            "status": status_counts
        }
    
    def batch_create_scheduled_tasks(
        self,
        store_ids: List[int],
        task_type: str,
        cron_expression: str,
        name: Optional[str] = None
    ) -> List[int]:
        """批量创建定时任务"""
        scheduled_task_ids = []
        
        for store_id in store_ids:
            scheduled_task = ScheduledTask(
                store_id=store_id,
                task_type=task_type,
                name=name or f"{task_type}-{store_id}",
                cron_expression=cron_expression,
                is_active=True
            )
            self.db.add(scheduled_task)
            self.db.commit()
            scheduled_task_ids.append(scheduled_task.id)
        
        return scheduled_task_ids
    
    def batch_start_scheduled_tasks(self, scheduled_task_ids: List[int]):
        """批量启用定时任务"""
        self.db.query(ScheduledTask).filter(
            ScheduledTask.id.in_(scheduled_task_ids)
        ).update(
            {ScheduledTask.is_active: True},
            synchronize_session=False
        )
        self.db.commit()
    
    def batch_stop_scheduled_tasks(self, scheduled_task_ids: List[int]):
        """批量停用定时任务"""
        self.db.query(ScheduledTask).filter(
            ScheduledTask.id.in_(scheduled_task_ids)
        ).update(
            {ScheduledTask.is_active: False},
            synchronize_session=False
        )
        self.db.commit()


# 预定义的任务模板
DEFAULT_TASK_TEMPLATES = {
    "daily_good_review": TaskTemplate(
        name="每日好评管理",
        task_type="good_review",
        description="自动回复今日好评",
        steps=[
            {"action": "navigate", "target": "reviews_page"},
            {"action": "filter", "filter": "good_reviews"},
            {"action": "iterate", "target": "review_list"},
            {"action": "reply", "template": "谢谢亲的好评！"}
        ],
        schedule="0 9 * * *"  # 每天早上9点
    ).to_dict(),
    
    "daily_data_fetch": TaskTemplate(
        name="每日数据采集",
        task_type="fetch_data",
        description="自动采集今日销售数据",
        steps=[
            {"action": "navigate", "target": "data_center"},
            {"action": "select", "target": "today"},
            {"action": "extract", "data": ["orders", "sales", "visitors"]},
            {"action": "save", "target": "database"}
        ],
        schedule="0 22 * * *"  # 每天晚上10点
    ).to_dict(),
    
    "weekly_analyze": TaskTemplate(
        name="每周运营分析",
        task_type="analyze",
        description="分析本周运营数据并给出建议",
        steps=[
            {"action": "fetch", "data": "weekly_stats"},
            {"action": "analyze", "metrics": ["orders", "sales", "conversion"]},
            {"action": "generate", "output": "report"},
            {"action": "notify", "channel": "im"}
        ],
        schedule="0 10 * * 1"  # 每周一早上10点
    ).to_dict()
}


def get_default_templates() -> List[Dict[str, Any]]:
    """获取默认模板"""
    return list(DEFAULT_TASK_TEMPLATES.values())
