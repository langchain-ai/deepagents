import asyncio
from typing import Optional
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from backend.database.models import ScheduledTask, Task, Store
try:
    from backend.agent.core import ECommerceAgent
except ImportError:
    ECommerceAgent = None


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.scheduler = AsyncIOScheduler()
        self.agent_instances = {}
    
    def start(self):
        """启动调度器"""
        self.scheduler.start()
        self._load_scheduled_tasks()
    
    def shutdown(self):
        """关闭调度器"""
        self.scheduler.shutdown()
    
    def _load_scheduled_tasks(self):
        """加载数据库中的定时任务"""
        tasks = self.db.query(ScheduledTask).filter(
            ScheduledTask.is_active == True
        ).all()
        
        for task in tasks:
            self._add_job(task)
    
    def _add_job(self, scheduled_task: ScheduledTask):
        """添加定时任务"""
        job_id = f"scheduled_{scheduled_task.id}"
        
        # 解析 cron 表达式
        parts = scheduled_task.cron_expression.split()
        if len(parts) != 5:
            return
        
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4]
        )
        
        self.scheduler.add_job(
            self._execute_scheduled_task,
            trigger=trigger,
            id=job_id,
            args=[scheduled_task.id],
            replace_existing=True
        )
    
    async def _execute_scheduled_task(self, scheduled_task_id: int):
        """执行定时任务"""
        scheduled_task = self.db.query(ScheduledTask).get(scheduled_task_id)
        if not scheduled_task or not scheduled_task.is_active:
            return
        
        store = self.db.query(Store).get(scheduled_task.store_id)
        if not store:
            return
        
        # 创建任务记录
        task = Task(
            store_id=store.id,
            task_type=scheduled_task.task_type,
            name=scheduled_task.name or f"定时任务-{scheduled_task.task_type}",
            status="pending",
            total_steps=100
        )
        self.db.add(task)
        self.db.commit()
        
        # 更新最后运行时间
        scheduled_task.last_run_at = datetime.utcnow()
        self.db.commit()
        
        # 执行任务
        try:
            agent = ECommerceAgent(self.db, store)
            await agent.initialize()
            await agent.execute_task(task)
        except Exception as e:
            print(f"定时任务执行失败: {e}")
    
    def add_scheduled_task(
        self,
        store_id: int,
        task_type: str,
        cron_expression: str,
        name: str = ""
    ) -> ScheduledTask:
        """添加定时任务"""
        task = ScheduledTask(
            store_id=store_id,
            task_type=task_type,
            name=name,
            cron_expression=cron_expression,
            is_active=True
        )
        self.db.add(task)
        self.db.commit()
        
        self._add_job(task)
        return task
    
    def update_scheduled_task(
        self,
        task_id: int,
        cron_expression: Optional[str] = None,
        name: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[ScheduledTask]:
        """更新定时任务"""
        task = self.db.query(ScheduledTask).get(task_id)
        if not task:
            return None
        
        if cron_expression is not None:
            task.cron_expression = cron_expression
        
        if name is not None:
            task.name = name
        
        if is_active is not None:
            task.is_active = is_active
        
        self.db.commit()
        
        # 重新调度
        job_id = f"scheduled_{task.id}"
        
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        if task.is_active:
            self._add_job(task)
        
        return task
    
    def delete_scheduled_task(self, task_id: int):
        """删除定时任务"""
        task = self.db.query(ScheduledTask).get(task_id)
        if not task:
            return
        
        # 从调度器中移除
        job_id = f"scheduled_{task.id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        # 从数据库删除
        self.db.delete(task)
        self.db.commit()


# 全局调度器实例
_task_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler(db: Session) -> TaskScheduler:
    """获取任务调度器"""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = TaskScheduler(db)
    return _task_scheduler
