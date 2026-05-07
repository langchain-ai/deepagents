from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from backend.config import settings

# 创建引擎
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Store(Base):
    """店铺表"""
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, comment="店铺名称")
    platform = Column(String(50), nullable=False, comment="平台：douyin/pinduoduo/taobao")
    username = Column(String(100), comment="账号")
    password = Column(String(255), comment="密码（加密）")
    profile_path = Column(String(255), comment="Profile 路径")
    fingerprint = Column(JSON, comment="浏览器指纹配置")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    tasks = relationship("Task", back_populates="store")
    logs = relationship("OperationLog", back_populates="store")
    daily_data = relationship("DailyData", back_populates="store")


class Task(Base):
    """任务表"""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    task_type = Column(String(50), nullable=False, comment="任务类型：publish/good_review/fetch_data/analyze")
    name = Column(String(100), comment="任务名称")
    status = Column(String(20), default="pending", comment="状态：pending/running/completed/failed/paused")
    progress = Column(Integer, default=0, comment="进度 0-100")
    current_step = Column(String(255), comment="当前步骤")
    total_steps = Column(Integer, default=0, comment="总步骤数")
    completed_steps = Column(Integer, default=0, comment="已完成步骤数")
    error_message = Column(Text, comment="错误信息")
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    result = Column(JSON, comment="任务结果")
    checkpoint = Column(JSON, comment="断点续跑数据")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    store = relationship("Store", back_populates="tasks")
    logs = relationship("OperationLog", back_populates="task")


class ScheduledTask(Base):
    """定时任务表"""
    __tablename__ = "scheduled_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    task_type = Column(String(50), nullable=False, comment="任务类型")
    name = Column(String(100), comment="任务名称")
    cron_expression = Column(String(100), nullable=False, comment="Cron 表达式")
    is_active = Column(Boolean, default=True, comment="是否启用")
    last_run_at = Column(DateTime, comment="上次运行时间")
    next_run_at = Column(DateTime, comment="下次运行时间")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OperationLog(Base):
    """操作日志表"""
    __tablename__ = "operation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"))
    task_id = Column(Integer, ForeignKey("tasks.id"))
    operation_type = Column(String(50), nullable=False, comment="操作类型")
    status = Column(String(20), default="success", comment="状态：success/failed/warning")
    message = Column(Text, comment="日志消息")
    screenshot_path = Column(String(255), comment="截图路径")
    recording_path = Column(String(255), comment="录屏路径")
    error_detail = Column(Text, comment="错误详情")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    store = relationship("Store", back_populates="logs")
    task = relationship("Task", back_populates="logs")


class DailyData(Base):
    """每日数据表"""
    __tablename__ = "daily_data"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    date = Column(String(10), nullable=False, comment="日期 YYYY-MM-DD")
    orders_count = Column(Integer, default=0, comment="订单数")
    sales_amount = Column(Float, default=0, comment="销售额")
    visitors_count = Column(Integer, default=0, comment="访客数")
    conversion_rate = Column(Float, default=0, comment="转化率")
    products_published = Column(Integer, default=0, comment="发布商品数")
    promotion_data = Column(JSON, comment="推广数据")
    ai_analysis = Column(Text, comment="AI 分析结论")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关系
    store = relationship("Store", back_populates="daily_data")
    
    __table_args__ = (
        {'sqlite_autoincrement': True}
    )


class DOMElement(Base):
    """DOM 元素表"""
    __tablename__ = "dom_elements"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False, comment="平台")
    page = Column(String(100), nullable=False, comment="页面")
    name = Column(String(100), nullable=False, comment="元素名称")
    selectors = Column(JSON, nullable=False, comment="选择器列表（CSS/XPath/text）")
    description = Column(String(255), comment="描述")
    version = Column(Integer, default=1, comment="版本号")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeItem(Base):
    """知识项表"""
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), nullable=False, comment="类型：product/rule/template")
    title = Column(String(255), nullable=False, comment="标题")
    content = Column(Text, nullable=False, comment="内容")
    extra_data = Column(JSON, comment="元数据")
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperienceItem(Base):
    """经验项表"""
    __tablename__ = "experience_items"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"))
    platform = Column(String(50), comment="平台")
    scenario = Column(String(100), comment="场景")
    content = Column(Text, nullable=False, comment="经验内容")
    is_success = Column(Boolean, default=True, comment="是否成功案例")
    extra_data = Column(JSON, comment="元数据")
    created_at = Column(DateTime, default=datetime.utcnow)


class BrowserTab(Base):
    """浏览器 Tab 表"""
    __tablename__ = "browser_tabs"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    tab_id = Column(String(100), nullable=False, comment="Tab ID")
    url = Column(String(500), comment="当前 URL")
    is_active = Column(Boolean, default=True, comment="是否活跃")
    last_used_at = Column(DateTime, default=datetime.utcnow, comment="最后使用时间")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """初始化数据库"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
