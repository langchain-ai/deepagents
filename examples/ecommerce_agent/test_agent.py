#!/usr/bin/env python3
"""测试电商Agent核心功能"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings
from backend.database.models import init_db, get_db, Store, Task
try:
    from backend.agent.core import ECommerceAgent
except ImportError:
    print("警告：deepagents 模块未安装，将跳过相关测试")
    ECommerceAgent = None
from backend.browser.elements import ElementManager, init_default_elements
from backend.browser.anti_detect import AntiDetectManager
from backend.agent.checkpoint import CheckpointManager, StepTracker
from backend.agent.exception_handler import GlobalExceptionHandler, ExceptionHandlerFactory
from backend.scheduler.scheduler import TaskScheduler
from backend.utils.encryption import CredentialManager
from backend.utils.resource_monitor import ResourceMonitor
from backend.utils.webhook import WebhookAlertManager, AlertEventTypes


def test_database():
    """测试数据库初始化"""
    print("测试数据库初始化...")
    try:
        init_db()
        print("✓ 数据库初始化成功")
        
        # 获取数据库会话
        db = next(get_db())
        
        # 测试创建店铺
        store = Store(
            name="测试店铺",
            platform="douyin",
            username="test_user",
            is_active=True
        )
        db.add(store)
        db.commit()
        
        # 测试创建任务
        task = Task(
            store_id=store.id,
            task_type="publish",
            name="测试任务",
            status="pending",
            total_steps=100
        )
        db.add(task)
        db.commit()
        
        print("✓ 店铺和任务创建成功")
        
        # 查询测试
        stores = db.query(Store).all()
        tasks = db.query(Task).all()
        print(f"✓ 查询成功：{len(stores)} 个店铺，{len(tasks)} 个任务")
        
        return db, store, task
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        return None, None, None


def test_element_manager(db):
    """测试元素管理器"""
    print("\n测试元素管理器...")
    try:
        manager = ElementManager(db)
        init_default_elements(db)
        
        # 测试获取元素
        element = manager.get_element("douyin", "login", "username_input")
        if element:
            print("✓ 元素获取成功")
            print(f"  元素: {element['name']}")
            print(f"  选择器: {element['selectors']}")
        else:
            print("✗ 元素获取失败")
    except Exception as e:
        print(f"✗ 元素管理器测试失败: {e}")


def test_anti_detect():
    """测试防检测模块"""
    print("\n测试防检测模块...")
    try:
        anti_detect = AntiDetectManager()
        
        # 测试生成指纹
        fingerprint = anti_detect.generate_fingerprint()
        print("✓ 指纹生成成功")
        print(f"  User-Agent: {fingerprint['user_agent'][:50]}...")
        print(f"  视口: {fingerprint['viewport']}")
        
        # 测试人类行为延迟
        anti_detect.human_like_click_delay(1)
        print("✓ 人类行为延迟测试通过")
        
        # 测试打字模拟
        actions = anti_detect.type_with_delay("测试文本")
        print(f"✓ 打字模拟测试通过，生成 {len(actions)} 个动作")
        
    except Exception as e:
        print(f"✗ 防检测模块测试失败: {e}")


def test_checkpoint(db, task):
    """测试断点续跑"""
    print("\n测试断点续跑...")
    try:
        checkpoint_manager = CheckpointManager(db, task)
        
        # 保存断点
        checkpoint_manager.save_checkpoint(
            step_index=5,
            step_name="测试步骤",
            step_data={"key": "value"},
            metadata={"test": True}
        )
        print("✓ 断点保存成功")
        
        # 加载断点
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint:
            print(f"✓ 断点加载成功")
            print(f"  当前步骤: {checkpoint['step_name']}")
            print(f"  步骤索引: {checkpoint['step_index']}")
        else:
            print("✗ 断点加载失败")
        
        # 测试步骤跟踪器
        step_tracker = StepTracker(checkpoint_manager, 10)
        step_tracker.start_step("第一步")
        step_tracker.complete_step({"result": "success"})
        progress = step_tracker.get_progress()
        print(f"✓ 步骤跟踪器测试通过，进度: {progress['progress_percent']}%")
        
    except Exception as e:
        print(f"✗ 断点续跑测试失败: {e}")


async def async_test_exception_handler(db, task):
    """异步测试异常处理"""
    try:
        handler = GlobalExceptionHandler(db, task)
        
        # 测试轻度异常处理
        result = await handler.handle_exception(Exception("测试异常"), level="light")
        print(f"✓ 轻度异常处理成功: {result['status']}")
        
        # 测试获取错误摘要
        summary = handler.get_error_summary()
        print(f"✓ 错误摘要获取成功: {summary['total_errors']} 个错误")
        
    except Exception as e:
        print(f"✗ 异常处理测试失败: {e}")


def test_exception_handler(db, task):
    """测试异常处理"""
    print("\n测试异常处理...")
    asyncio.run(async_test_exception_handler(db, task))


def test_encryption(db):
    """测试加密模块"""
    print("\n测试加密模块...")
    try:
        credential_manager = CredentialManager(db)
        
        # 测试加密解密
        encrypted = credential_manager.encryption.encrypt("测试密码")
        decrypted = credential_manager.encryption.decrypt(encrypted)
        if decrypted == "测试密码":
            print("✓ 加密解密测试通过")
        else:
            print("✗ 加密解密测试失败")
        
    except Exception as e:
        print(f"✗ 加密模块测试失败: {e}")


def test_resource_monitor():
    """测试资源监控"""
    print("\n测试资源监控...")
    try:
        monitor = ResourceMonitor()
        
        # 获取当前使用情况
        usage = monitor.get_current_usage()
        print("✓ 资源使用情况获取成功")
        print(f"  CPU: {usage['cpu']['percent']}%")
        print(f"  内存: {usage['memory']['percent']}%")
        print(f"  磁盘: {usage['disk']['percent']}%")
        
        # 检查阈值
        result = monitor.check_thresholds()
        print(f"✓ 阈值检查完成，是否有警报: {result['has_alerts']}")
        
    except Exception as e:
        print(f"✗ 资源监控测试失败: {e}")


def test_webhook(db):
    """测试Webhook告警"""
    print("\n测试Webhook告警...")
    try:
        alert_manager = WebhookAlertManager(db)
        
        # 添加测试Webhook
        alert_manager.add_webhook(
            webhook_id="test",
            url="http://localhost:8000/webhook/test",
            events=[AlertEventTypes.TASK_FAILED, AlertEventTypes.TASK_COMPLETED],
            name="测试Webhook"
        )
        
        webhooks = alert_manager.get_webhooks()
        if len(webhooks) > 0:
            print("✓ Webhook添加成功")
        else:
            print("✗ Webhook添加失败")
            
        # 获取告警历史
        history = alert_manager.get_alert_history()
        print(f"✓ 告警历史获取成功，记录数: {len(history)}")
        
    except Exception as e:
        print(f"✗ Webhook测试失败: {e}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("电商Agent系统测试")
    print("=" * 60)
    
    # 测试数据库
    db, store, task = test_database()
    
    if db and store and task:
        # 测试元素管理器
        test_element_manager(db)
        
        # 测试防检测模块
        test_anti_detect()
        
        # 测试断点续跑
        test_checkpoint(db, task)
        
        # 测试异常处理
        test_exception_handler(db, task)
        
        # 测试加密模块
        test_encryption(db)
        
        # 测试资源监控
        test_resource_monitor()
        
        # 测试Webhook告警
        test_webhook(db)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
