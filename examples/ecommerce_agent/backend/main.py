import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
from backend.config import settings
from backend.database.models import init_db, get_db, Store, Task, ScheduledTask, DOMElement, OperationLog, DailyData
from backend.agent.core import ECommerceAgent
from backend.scheduler.scheduler import get_task_scheduler
from backend.browser.manager import get_browser_manager
from backend.browser.elements import init_default_elements


# Pydantic Models for Request/Response
class StoreCreate(BaseModel):
    name: str
    platform: str
    username: str = ""
    password: str = ""
    is_active: bool = True


class TaskCreate(BaseModel):
    store_id: int
    task_type: str
    name: str = ""


class DOMElementCreate(BaseModel):
    platform: str
    page: str
    name: str
    selector: str
    selector_type: str = "css"
    description: str = ""
    version: int = 1
    is_active: bool = True


class DOMElementUpdate(BaseModel):
    name: Optional[str] = None
    selector: Optional[str] = None
    selector_type: Optional[str] = None
    description: Optional[str] = None
    version: Optional[int] = None
    is_active: Optional[bool] = None


class ScheduledTaskCreate(BaseModel):
    store_id: int
    task_type: str
    name: str = ""
    cron_expression: str
    is_active: bool = True


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


# ==================== Stores API ====================
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


@app.get("/api/stores/{store_id}")
async def get_store(store_id: int = Path(...), db: Session = Depends(get_db)):
    """获取单个店铺详情"""
    store = db.query(Store).get(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    return {
        "id": store.id,
        "name": store.name,
        "platform": store.platform,
        "is_active": store.is_active,
        "created_at": store.created_at,
        "updated_at": store.updated_at
    }


@app.post("/api/stores")
async def create_store(
    store_data: StoreCreate,
    db: Session = Depends(get_db)
):
    """创建店铺"""
    store = Store(
        name=store_data.name,
        platform=store_data.platform,
        username=store_data.username,
        password=store_data.password,
        is_active=store_data.is_active
    )
    db.add(store)
    db.commit()
    db.refresh(store)
    return {
        "id": store.id,
        "name": store.name,
        "platform": store.platform
    }


@app.delete("/api/stores/{store_id}")
async def delete_store(store_id: int, db: Session = Depends(get_db)):
    """删除店铺"""
    store = db.query(Store).get(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    db.delete(store)
    db.commit()
    return {"status": "success", "message": "Store deleted"}


# ==================== Tasks API ====================
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


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: int, db: Session = Depends(get_db)):
    """获取任务详情"""
    task = db.query(Task).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": task.id,
        "store_id": task.store_id,
        "name": task.name,
        "task_type": task.task_type,
        "status": task.status,
        "progress": task.progress,
        "current_step": task.current_step,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "result": task.result,
        "error_message": task.error_message
    }


@app.post("/api/tasks")
async def create_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db)
):
    """创建任务"""
    store = db.query(Store).get(task_data.store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    task = Task(
        store_id=task_data.store_id,
        task_type=task_data.task_type,
        name=task_data.name or f"{task_data.task_type}-{task_data.store_id}",
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


@app.put("/api/tasks/{task_id}/pause")
async def pause_task(task_id: int, db: Session = Depends(get_db)):
    """暂停任务"""
    task = db.query(Task).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = "pending"
    db.commit()
    return {"status": "success"}


@app.put("/api/tasks/{task_id}/resume")
async def resume_task(task_id: int, db: Session = Depends(get_db)):
    """恢复任务"""
    task = db.query(Task).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = "running"
    db.commit()
    return {"status": "success"}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: int, db: Session = Depends(get_db)):
    """删除任务"""
    task = db.query(Task).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"status": "success"}


# ==================== DOM Elements API ====================
@app.get("/api/dom-elements")
async def get_dom_elements(
    platform: str = None,
    page: str = None,
    db: Session = Depends(get_db)
):
    """获取DOM元素列表"""
    query = db.query(DOMElement)
    if platform:
        query = query.filter(DOMElement.platform == platform)
    if page:
        query = query.filter(DOMElement.page == page)
    
    elements = query.order_by(DOMElement.platform, DOMElement.page).all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "platform": e.platform,
            "page": e.page,
            "selector": e.selectors[0] if e.selectors and len(e.selectors) > 0 else "",
            "selector_type": "css",
            "description": e.description,
            "version": e.version,
            "is_active": e.is_active,
            "created_at": e.created_at,
            "updated_at": e.updated_at
        }
        for e in elements
    ]


@app.get("/api/dom-elements/{element_id}")
async def get_dom_element(element_id: int, db: Session = Depends(get_db)):
    """获取单个DOM元素"""
    element = db.query(DOMElement).get(element_id)
    if not element:
        raise HTTPException(status_code=404, detail="Element not found")
    return {
        "id": element.id,
        "name": element.name,
        "platform": element.platform,
        "page": element.page,
        "selector": element.selectors[0] if element.selectors and len(element.selectors) > 0 else "",
        "selector_type": "css",
        "description": element.description,
        "version": element.version,
        "is_active": element.is_active,
        "created_at": element.created_at,
        "updated_at": element.updated_at
    }


@app.post("/api/dom-elements")
async def create_dom_element(
    data: DOMElementCreate,
    db: Session = Depends(get_db)
):
    """创建DOM元素"""
    element = DOMElement(
        name=data.name,
        platform=data.platform,
        page=data.page,
        selectors=[data.selector],
        description=data.description,
        version=data.version,
        is_active=data.is_active
    )
    db.add(element)
    db.commit()
    db.refresh(element)
    return {
        "id": element.id,
        "name": element.name,
        "platform": element.platform
    }


@app.put("/api/dom-elements/{element_id}")
async def update_dom_element(
    element_id: int,
    data: DOMElementUpdate,
    db: Session = Depends(get_db)
):
    """更新DOM元素"""
    element = db.query(DOMElement).get(element_id)
    if not element:
        raise HTTPException(status_code=404, detail="Element not found")
    
    if data.name is not None:
        element.name = data.name
    if data.description is not None:
        element.description = data.description
    if data.version is not None:
        element.version = data.version
    if data.is_active is not None:
        element.is_active = data.is_active
    if data.selector is not None:
        element.selectors = [data.selector]
    
    db.commit()
    db.refresh(element)
    return {"id": element.id, "status": "success"}


@app.delete("/api/dom-elements/{element_id}")
async def delete_dom_element(element_id: int, db: Session = Depends(get_db)):
    """删除DOM元素"""
    element = db.query(DOMElement).get(element_id)
    if not element:
        raise HTTPException(status_code=404, detail="Element not found")
    db.delete(element)
    db.commit()
    return {"status": "success"}


# ==================== Scheduled Tasks API ====================
@app.get("/api/scheduled-tasks")
async def get_scheduled_tasks(
    store_id: int = None,
    db: Session = Depends(get_db)
):
    """获取定时任务列表"""
    query = db.query(ScheduledTask)
    if store_id:
        query = query.filter(ScheduledTask.store_id == store_id)
    
    tasks = query.order_by(ScheduledTask.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "store_id": t.store_id,
            "name": t.name,
            "task_type": t.task_type,
            "cron_expression": t.cron_expression,
            "is_active": t.is_active,
            "last_run_at": t.last_run_at,
            "next_run_at": t.next_run_at,
            "created_at": t.created_at
        }
        for t in tasks
    ]


@app.post("/api/scheduled-tasks")
async def create_scheduled_task(
    data: ScheduledTaskCreate,
    db: Session = Depends(get_db)
):
    """创建定时任务"""
    store = db.query(Store).get(data.store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    task = ScheduledTask(
        store_id=data.store_id,
        task_type=data.task_type,
        name=data.name or f"scheduled-{data.task_type}",
        cron_expression=data.cron_expression,
        is_active=data.is_active
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {"id": task.id, "status": "success"}


@app.put("/api/scheduled-tasks/{task_id}/pause")
async def pause_scheduled_task(task_id: int, db: Session = Depends(get_db)):
    """暂停定时任务"""
    task = db.query(ScheduledTask).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.is_active = False
    db.commit()
    return {"status": "success"}


@app.put("/api/scheduled-tasks/{task_id}/resume")
async def resume_scheduled_task(task_id: int, db: Session = Depends(get_db)):
    """恢复定时任务"""
    task = db.query(ScheduledTask).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.is_active = True
    db.commit()
    return {"status": "success"}


@app.delete("/api/scheduled-tasks/{task_id}")
async def delete_scheduled_task(task_id: int, db: Session = Depends(get_db)):
    """删除定时任务"""
    task = db.query(ScheduledTask).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"status": "success"}


# ==================== Orders API ====================
@app.get("/api/orders")
async def get_orders(
    store_id: int = None,
    db: Session = Depends(get_db)
):
    """获取订单列表（模拟数据）"""
    return [
        {
            "id": i,
            "order_id": f"ORD{10000+i}",
            "store_id": store_id or 1,
            "store_name": "测试店铺",
            "platform": ["douyin", "pinduoduo", "taobao"][i % 3],
            "status": ["pending_pay", "pending_ship", "shipped", "completed"][i % 4],
            "amount": round(50 + i * 10.5, 2),
            "item_count": 1 + i % 3,
            "buyer_name": f"买家{i}",
            "create_time": datetime.now().isoformat(),
            "pay_time": datetime.now().isoformat()
        }
        for i in range(10)
    ]


@app.get("/api/orders/{order_id}")
async def get_order(order_id: int):
    """获取订单详情"""
    return {
        "id": order_id,
        "order_id": f"ORD{10000+order_id}",
        "store_id": 1,
        "store_name": "测试店铺",
        "platform": "douyin",
        "status": "pending_ship",
        "amount": 99.9,
        "item_count": 1,
        "buyer_name": "测试买家",
        "create_time": datetime.now().isoformat(),
        "pay_time": datetime.now().isoformat()
    }


# ==================== Products API ====================
@app.get("/api/products")
async def get_products(
    store_id: int = None,
    db: Session = Depends(get_db)
):
    """获取商品列表（模拟数据）"""
    categories = ["数码产品", "服装鞋帽", "家居用品", "食品饮料", "美妆护肤"]
    return [
        {
            "id": i,
            "product_id": f"PRD{10000+i}",
            "store_id": store_id or 1,
            "store_name": "测试店铺",
            "platform": ["douyin", "pinduoduo", "taobao"][i % 3],
            "title": f"商品{i+1}",
            "category": categories[i % len(categories)],
            "price": round(29.9 + i * 15.5, 2),
            "original_price": round(49.9 + i * 20, 2),
            "stock": 100 - i * 5,
            "sales": 50 + i * 20,
            "status": ["online", "offline"][i % 2],
            "create_time": datetime.now().isoformat(),
            "update_time": datetime.now().isoformat()
        }
        for i in range(15)
    ]


@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    """获取商品详情"""
    return {
        "id": product_id,
        "product_id": f"PRD{10000+product_id}",
        "store_id": 1,
        "store_name": "测试店铺",
        "platform": "douyin",
        "title": f"商品{product_id+1}",
        "category": "数码产品",
        "price": 99.9,
        "original_price": 149.9,
        "stock": 50,
        "sales": 100,
        "status": "online",
        "create_time": datetime.now().isoformat(),
        "update_time": datetime.now().isoformat()
    }


# ==================== Analytics API ====================
@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """获取数据分析摘要"""
    return {
        "revenue": 128650.00,
        "orders": 856,
        "customers": 623,
        "conversion_rate": 3.8
    }


@app.get("/api/analytics/trends")
async def get_trends():
    """获取销售趋势数据"""
    return {
        "dates": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05"],
        "values": [15000, 18000, 22000, 25000, 28000]
    }


@app.get("/api/analytics/top-products")
async def get_top_products():
    """获取热销商品"""
    return [
        {"rank": 1, "name": "智能蓝牙耳机Pro", "sales": 320, "revenue": 63968},
        {"rank": 2, "name": "纯棉毛巾套装", "sales": 1200, "revenue": 47880},
        {"rank": 3, "name": "家用收纳箱套装", "sales": 890, "revenue": 53351},
        {"rank": 4, "name": "运动休闲T恤", "sales": 560, "revenue": 50344},
        {"rank": 5, "name": "护肤精华液", "sales": 280, "revenue": 83720}
    ]


@app.get("/api/analytics/platform-stats")
async def get_platform_stats():
    """获取各平台统计"""
    return [
        {"name": "抖音", "revenue": 45200, "orders": 280, "percentage": 35},
        {"name": "淘宝", "revenue": 32800, "orders": 220, "percentage": 25},
        {"name": "拼多多", "revenue": 28500, "orders": 310, "percentage": 22},
        {"name": "京东", "revenue": 15800, "orders": 80, "percentage": 12},
        {"name": "小红书", "revenue": 6350, "orders": 40, "percentage": 5}
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
