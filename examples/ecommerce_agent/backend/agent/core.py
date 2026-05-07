import asyncio
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from deepagents import create_deep_agent
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.backends.filesystem import FilesystemBackend
from backend.config import settings
from backend.database.models import Store, Task, OperationLog
from backend.browser.manager import get_browser_manager
from backend.browser.elements import ElementManager
from backend.browser.anti_detect import anti_detect
from backend.database.vector_store import vector_store


class ECommerceAgent:
    """电商 Agent"""
    
    def __init__(self, db: Session, store: Store):
        self.db = db
        self.store = store
        self.agent = None
        self.browser_manager = get_browser_manager(db)
        self.element_manager = ElementManager(db)
        self.current_task: Optional[Task] = None
    
    async def initialize(self):
        """初始化 Agent"""
        # 创建 DeepAgents Agent
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        skills_dir = os.path.join(project_root, "skills")
        
        self.agent = create_deep_agent(
            system_prompt=self._get_system_prompt(),
            tools=[
                self._get_element,
                self._click_element,
                self._input_text,
                self._navigate,
                self._take_screenshot,
                self._log_operation,
                self._search_knowledge,
                self._search_experience,
            ],
            skills=[skills_dir],
            backend=FilesystemBackend(root_dir=project_root)
        )
        # 启动浏览器
        await self.browser_manager.start()
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return f"""你是一个专业的电商运营助手，帮助用户在各大电商平台进行自动化操作。

当前店铺：{self.store.name}
平台：{self.store.platform}

你的主要任务包括：
1. 商品发布：自动填写商品信息并发布
2. 好评管理：自动回复好评、追评
3. 数据采集：采集订单、销售、推广数据
4. 运营分析：分析数据并提供优化建议

操作原则：
- 遵循平台规则，不要进行违规操作
- 保持人类行为模式，避免被检测为机器人
- 操作前先搜索知识库和经验库，参考历史经验
- 每个操作都要记录日志，关键步骤截图
- 遇到异常及时暂停，等待人工介入
- 优先使用技能系统中的专业技能完成任务

可用工具：
- get_element: 获取页面元素
- click_element: 点击元素
- input_text: 输入文本
- navigate: 导航到 URL
- take_screenshot: 截图
- log_operation: 记录操作日志
- search_knowledge: 搜索知识库
- search_experience: 搜索经验库

技能系统：
你拥有专业的电商技能库，在执行任务时：
1. 首先检查是否有相关技能
2. 阅读技能的完整说明
3. 按照技能指导完成任务
4. 使用技能推荐的工具

可用技能包括：
- product-publish: 商品发布
- good-review: 好评管理
- data-collection: 数据采集
"""
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行任务"""
        self.current_task = task
        
        # 更新任务状态
        task.status = "running"
        task.started_at = datetime.utcnow()
        self.db.commit()
        
        try:
            # 检查是否有断点
            if task.checkpoint:
                result = await self._resume_from_checkpoint(task.checkpoint)
            else:
                # 根据任务类型执行
                if task.task_type == "publish":
                    result = await self._publish_product()
                elif task.task_type == "good_review":
                    result = await self._manage_good_reviews()
                elif task.task_type == "fetch_data":
                    result = await self._fetch_data()
                elif task.task_type == "analyze":
                    result = await self._analyze_data()
                else:
                    raise Exception(f"未知任务类型: {task.task_type}")
            
            # 任务完成
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = result
            task.progress = 100
            self.db.commit()
            
            return result
            
        except Exception as e:
            # 任务失败
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            self.db.commit()
            
            # 记录错误日志
            await self._log_operation(
                operation_type="error",
                status="failed",
                message=str(e)
            )
            
            raise
    
    async def _publish_product(self) -> Dict[str, Any]:
        """发布商品"""
        # 搜索相关经验
        experiences = vector_store.search_experience(
            query=f"{self.store.platform} 商品发布",
            platform=self.store.platform
        )
        
        # 记录日志
        await self._log_operation(
            operation_type="publish",
            status="running",
            message="开始发布商品"
        )
        
        # 这里添加具体的发布逻辑
        # 实际实现需要根据具体平台的页面结构编写
        
        # 模拟进度更新
        await self._update_progress(20, "导航到发布页面")
        
        # 增加计数
        anti_detect.increment_product_count(self.store.id)
        
        return {
            "status": "success",
            "message": "商品发布完成"
        }
    
    async def _manage_good_reviews(self) -> Dict[str, Any]:
        """管理好评"""
        await self._update_progress(30, "开始管理好评")
        
        return {
            "status": "success",
            "message": "好评管理完成"
        }
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """采集数据"""
        await self._update_progress(50, "开始采集数据")
        
        return {
            "status": "success",
            "message": "数据采集完成"
        }
    
    async def _analyze_data(self) -> Dict[str, Any]:
        """分析数据"""
        await self._update_progress(70, "开始分析数据")
        
        return {
            "status": "success",
            "message": "数据分析完成"
        }
    
    async def _resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """从断点恢复"""
        # 实现断点续跑逻辑
        return {"status": "resumed"}
    
    async def _update_progress(self, progress: int, current_step: str):
        """更新进度"""
        if self.current_task:
            self.current_task.progress = progress
            self.current_task.current_step = current_step
            self.current_task.completed_steps = progress
            self.db.commit()
    
    # 工具函数
    async def _get_element(self, platform: str, page: str, name: str) -> Dict[str, Any]:
        """获取元素"""
        element = self.element_manager.get_element(platform, page, name)
        return element or {}
    
    async def _click_element(self, element_name: str):
        """点击元素"""
        anti_detect.human_like_click_delay(self.store.id)
        
        page = await self.browser_manager.get_tab(self.store)
        # 这里需要实现元素查找和点击逻辑
        # 实际实现需要根据元素选择器查找并点击
        
        await self._log_operation(
            operation_type="click",
            status="success",
            message=f"点击元素: {element_name}"
        )
    
    async def _input_text(self, element_name: str, text: str):
        """输入文本"""
        page = await self.browser_manager.get_tab(self.store)
        
        # 使用人类打字模式
        actions = anti_detect.type_with_delay(text)
        for action in actions:
            if action["type"] == "pause":
                await asyncio.sleep(action["duration"])
        
        await self._log_operation(
            operation_type="input",
            status="success",
            message=f"输入文本: {element_name}"
        )
    
    async def _navigate(self, url: str):
        """导航到 URL"""
        await self.browser_manager.navigate(self.store, url)
        
        await self._log_operation(
            operation_type="navigate",
            status="success",
            message=f"导航到: {url}"
        )
    
    async def _take_screenshot(self, name: str = "") -> str:
        """截图"""
        screenshot_path = await self.browser_manager.take_screenshot(self.store.id, name)
        
        await self._log_operation(
            operation_type="screenshot",
            status="success",
            message=f"截图: {name}",
            screenshot_path=screenshot_path
        )
        
        return screenshot_path
    
    async def _log_operation(
        self,
        operation_type: str,
        status: str,
        message: str,
        screenshot_path: str = "",
        error_detail: str = ""
    ):
        """记录操作日志"""
        log = OperationLog(
            store_id=self.store.id,
            task_id=self.current_task.id if self.current_task else None,
            operation_type=operation_type,
            status=status,
            message=message,
            screenshot_path=screenshot_path,
            error_detail=error_detail
        )
        self.db.add(log)
        self.db.commit()
    
    async def _search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识库"""
        return vector_store.search_knowledge(query)
    
    async def _search_experience(self, query: str) -> List[Dict[str, Any]]:
        """搜索经验库"""
        return vector_store.search_experience(query, platform=self.store.platform)
