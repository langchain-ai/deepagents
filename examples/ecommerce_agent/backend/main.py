import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import init_db, get_db, Store, Task
from backend.agent.core import ECommerceAgent
from backend.scheduler.scheduler import get_task_scheduler
from backend.browser.manager import get_browser_manager
from backend.browser.elements import init_default_elements


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 初始化数据库
    init_db()
    
    # 初始化默认元素
    db = next(get_db())
    try:
        init_default_elements(db)
    except:
        pass
    
    # 启动调度器
    scheduler = get_task_scheduler(db)
    scheduler.start()
    
    yield
    
    # 关闭调度器
    scheduler.shutdown()
    
    # 关闭浏览器
    browser_manager = get_browser_manager(db)
    await browser_manager.stop()


app = FastAPI(
    title="E-Commerce Agent API",
    version="0.1.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/stores")
async def get_stores(db: Session = Depends(get_db)):
    """获取店铺列表"""
    stores = db.query(Store).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "platform": s.platform,
            "is_active": s.is_active,
            "created_at": s.created_at
        }
        for s in stores
    ]


@app.post("/api/stores")
async def create_store(
    name: str,
    platform: str,
    username: str = "",
    password: str = "",
    db: Session = Depends(get_db)
):
    """创建店铺"""
    store = Store(
        name=name,
        platform=platform,
        username=username,
        password=password,
        is_active=True
    )
    db.add(store)
    db.commit()
    db.refresh(store)
    return {
        "id": store.id,
        "name": store.name,
        "platform": store.platform
    }


@app.get("/api/tasks")
async def get_tasks(
    store_id: int = None,
    db: Session = Depends(get_db)
):
    """获取任务列表"""
    query = db.query(Task)
    if store_id:
        query = query.filter(Task.store_id == store_id)
    
    tasks = query.order_by(Task.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "store_id": t.store_id,
            "name": t.name,
            "task_type": t.task_type,
            "status": t.status,
            "progress": t.progress,
            "current_step": t.current_step,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "completed_at": t.completed_at
        }
        for t in tasks
    ]


@app.post("/api/tasks")
async def create_task(
    store_id: int,
    task_type: str,
    name: str = "",
    db: Session = Depends(get_db)
):
    """创建任务"""
    store = db.query(Store).get(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    task = Task(
        store_id=store_id,
        task_type=task_type,
        name=name or f"{task_type}-{store_id}",
        status="pending",
        total_steps=100
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 异步执行任务
    async def execute_task():
        agent = ECommerceAgent(db, store)
        await agent.initialize()
        await agent.execute_task(task)
    
    asyncio.create_task(execute_task())
    
    return {
        "id": task.id,
        "status": "submitted"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
