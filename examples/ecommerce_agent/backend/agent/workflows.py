import asyncio
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from backend.database.models import Store, Task, DailyData
from backend.browser.elements import ElementManager
from backend.browser.manager import get_browser_manager
from backend.browser.anti_detect import anti_detect
from backend.browser.screenshot_manager import ScreenshotManager, RecordingManager
from backend.agent.checkpoint import CheckpointManager, StepTracker
from backend.agent.exception_handler import GlobalExceptionHandler, ExceptionHandlerFactory


class WorkflowContext:
    """工作流上下文"""
    
    def __init__(self, db: Session, store: Store, task: Task):
        self.db = db
        self.store = store
        self.task = task
        self.element_manager = ElementManager(db)
        self.browser_manager = get_browser_manager(db)
        self.screenshot_manager = ScreenshotManager(db, store.id, task.id)
        self.recording_manager = RecordingManager(db, store.id, task.id)
        self.checkpoint_manager = CheckpointManager(db, task)
        self.step_tracker = StepTracker(checkpoint_manager, 10)
        self.exception_handler = GlobalExceptionHandler(db, task)
        self.page = None
        self.results = {}
    
    async def setup(self):
        """设置工作流上下文"""
        await self.browser_manager.start()
        self.page = await self.browser_manager.get_tab(self.store)
    
    async def teardown(self):
        """清理工作流上下文"""
        if self.page:
            await self.browser_manager.close_tab(self.store.id)


class ProductPublishWorkflow:
    """商品发布工作流"""
    
    PLATFORM_URLS = {
        "douyin": "https://seller.douyin.com/",
        "pinduoduo": "https://mms.pinduoduo.com/",
        "taobao": "https://seller.taobao.com/",
        "jingdong": "https://shop.jd.com/",
        "xiaohongshu": "https://creator.xiaohongshu.com/"
    }
    
    def __init__(self, context: WorkflowContext, product_data: Dict[str, Any]):
        self.context = context
        self.product_data = product_data
    
    async def run(self) -> Dict[str, Any]:
        """执行商品发布工作流"""
        try:
            await self.context.setup()
            await self.context.step_tracker.start_step("开始商品发布")
            
            # 导航到发布页面
            await self._navigate_to_publish_page()
            await self.context.step_tracker.complete_step()
            
            # 填写商品信息
            await self._fill_product_info()
            await self.context.step_tracker.complete_step()
            
            # 上传图片
            await self._upload_images()
            await self.context.step_tracker.complete_step()
            
            # 设置规格
            await self._set_sku()
            await self.context.step_tracker.complete_step()
            
            # 提交发布
            await self._submit_publish()
            await self.context.step_tracker.complete_step()
            
            # 验证发布结果
            await self._verify_result()
            await self.context.step_tracker.complete_step()
            
            return {
                "status": "success",
                "message": "商品发布成功",
                "product_id": self.product_data.get("product_id"),
                "progress": self.context.step_tracker.get_progress()
            }
        
        except Exception as e:
            # 处理异常
            result = await self.context.exception_handler.handle_exception(e)
            return {"status": "failed", "error": str(e), "recovery": result}
        
        finally:
            await self.context.teardown()
    
    async def _navigate_to_publish_page(self):
        """导航到发布页面"""
        url = self.PLATFORM_URLS.get(self.context.store.platform, "")
        if url:
            await self.context.page.goto(url)
            await asyncio.sleep(2)
    
    async def _fill_product_info(self):
        """填写商品信息"""
        # 获取元素
        title_element = self.context.element_manager.get_element(
            self.context.store.platform, "publish", "title_input"
        )
        desc_element = self.context.element_manager.get_element(
            self.context.store.platform, "publish", "product_desc_textarea"
        )
        price_element = self.context.element_manager.get_element(
            self.context.store.platform, "publish", "price_input"
        )
        
        # 填写标题
        if title_element and self.product_data.get("title"):
            selector = title_element["selectors"][0]
            if selector["type"] == "css":
                await self.context.page.locator(selector["value"]).fill(self.product_data["title"])
            anti_detect.human_like_click_delay(self.context.store.id)
        
        # 填写描述
        if desc_element and self.product_data.get("description"):
            selector = desc_element["selectors"][0]
            if selector["type"] == "css":
                await self.context.page.locator(selector["value"]).fill(self.product_data["description"])
            anti_detect.human_like_click_delay(self.context.store.id)
        
        # 填写价格
        if price_element and self.product_data.get("price"):
            selector = price_element["selectors"][0]
            if selector["type"] == "css":
                await self.context.page.locator(selector["value"]).fill(str(self.product_data["price"]))
            anti_detect.human_like_click_delay(self.context.store.id)
    
    async def _upload_images(self):
        """上传图片"""
        images = self.product_data.get("images", [])
        if images:
            await self.context.screenshot_manager.capture_key_step_screenshot("upload_images")
    
    async def _set_sku(self):
        """设置商品规格"""
        skus = self.product_data.get("skus", [])
        if skus:
            sku_add_element = self.context.element_manager.get_element(
                self.context.store.platform, "publish", "sku_add_button"
            )
            if sku_add_element:
                selector = sku_add_element["selectors"][0]
                for sku in skus:
                    if selector["type"] == "css":
                        await self.context.page.locator(selector["value"]).click()
                    anti_detect.human_like_click_delay(self.context.store.id)
    
    async def _submit_publish(self):
        """提交发布"""
        submit_element = self.context.element_manager.get_element(
            self.context.store.platform, "publish", "submit_button"
        )
        if submit_element:
            selector = submit_element["selectors"][0]
            if selector["type"] == "css":
                await self.context.page.locator(selector["value"]).click()
            anti_detect.human_like_click_delay(self.context.store.id)
            
            # 截图记录
            await self.context.screenshot_manager.capture_key_step_screenshot("submit_publish")
    
    async def _verify_result(self):
        """验证发布结果"""
        await asyncio.sleep(5)
        success_element = self.context.element_manager.get_element(
            self.context.store.platform, "publish", "success_message"
        )
        if success_element:
            selector = success_element["selectors"][0]
            if selector["type"] == "css":
                try:
                    await self.context.page.locator(selector["value"]).wait_for(timeout=10000)
                    return True
                except:
                    return False
        return False


class GoodReviewWorkflow:
    """好评管理工作流"""
    
    def __init__(self, context: WorkflowContext):
        self.context = context
        self.reply_templates = [
            "感谢亲的好评！您的满意是我们最大的动力！",
            "感谢您的认可，我们会继续努力！",
            "谢谢亲的好评，欢迎下次再来！",
            "感谢支持，祝您生活愉快！"
        ]
    
    async def run(self) -> Dict[str, Any]:
        """执行好评管理工作流"""
        try:
            await self.context.setup()
            await self.context.step_tracker.start_step("开始好评管理")
            
            # 导航到评价页面
            await self._navigate_to_reviews()
            await self.context.step_tracker.complete_step()
            
            # 筛选好评
            await self._filter_good_reviews()
            await self.context.step_tracker.complete_step()
            
            # 批量回复
            reply_count = await self._batch_reply()
            await self.context.step_tracker.complete_step()
            
            return {
                "status": "success",
                "message": f"成功回复 {reply_count} 条好评",
                "reply_count": reply_count,
                "progress": self.context.step_tracker.get_progress()
            }
        
        except Exception as e:
            result = await self.context.exception_handler.handle_exception(e)
            return {"status": "failed", "error": str(e), "recovery": result}
        
        finally:
            await self.context.teardown()
    
    async def _navigate_to_reviews(self):
        """导航到评价页面"""
        urls = {
            "douyin": "https://seller.douyin.com/review",
            "pinduoduo": "https://mms.pinduoduo.com/reviews",
            "taobao": "https://seller.taobao.com/reviews"
        }
        url = urls.get(self.context.store.platform, "")
        if url:
            await self.context.page.goto(url)
            await asyncio.sleep(2)
    
    async def _filter_good_reviews(self):
        """筛选好评"""
        filter_element = self.context.element_manager.get_element(
            self.context.store.platform, "reviews", "filter_good_reviews"
        )
        if filter_element:
            selector = filter_element["selectors"][0]
            if selector["type"] == "css":
                await self.context.page.locator(selector["value"]).click()
            anti_detect.human_like_click_delay(self.context.store.id)
    
    async def _batch_reply(self) -> int:
        """批量回复好评"""
        reply_count = 0
        page_num = 0
        
        while page_num < 5:  # 最多处理5页
            # 获取好评列表
            list_element = self.context.element_manager.get_element(
                self.context.store.platform, "reviews", "review_list"
            )
            
            if list_element:
                items = await self.context.page.query_selector_all(".review-item")
                
                for item in items:
                    # 查找回复按钮
                    reply_btn = item.query_selector(".reply-btn")
                    if reply_btn:
                        await reply_btn.click()
                        anti_detect.human_like_click_delay(self.context.store.id)
                        
                        # 输入回复内容
                        reply_input = item.query_selector(".reply-input")
                        if reply_input:
                            template = self.reply_templates[reply_count % len(self.reply_templates)]
                            await reply_input.fill(template)
                            anti_detect.human_like_click_delay(self.context.store.id)
                            
                            # 发送回复
                            send_btn = item.query_selector(".send-btn")
                            if send_btn:
                                await send_btn.click()
                                reply_count += 1
                                anti_detect.human_like_click_delay(self.context.store.id)
            
            # 下一页
            next_page = self.context.element_manager.get_element(
                self.context.store.platform, "reviews", "next_page_button"
            )
            if next_page:
                selector = next_page["selectors"][0]
                if selector["type"] == "css":
                    try:
                        await self.context.page.locator(selector["value"]).click()
                        await asyncio.sleep(2)
                        page_num += 1
                    except:
                        break
            else:
                break
        
        return reply_count


class DataFetchWorkflow:
    """数据采集工作流"""
    
    def __init__(self, context: WorkflowContext):
        self.context = context
    
    async def run(self) -> Dict[str, Any]:
        """执行数据采集工作流"""
        try:
            await self.context.setup()
            await self.context.step_tracker.start_step("开始数据采集")
            
            # 导航到数据中心
            await self._navigate_to_data_center()
            await self.context.step_tracker.complete_step()
            
            # 获取订单数据
            orders_data = await self._fetch_orders_data()
            await self.context.step_tracker.complete_step()
            
            # 获取销售数据
            sales_data = await self._fetch_sales_data()
            await self.context.step_tracker.complete_step()
            
            # 获取访客数据
            visitor_data = await self._fetch_visitor_data()
            await self.context.step_tracker.complete_step()
            
            # 保存数据
            await self._save_data(orders_data, sales_data, visitor_data)
            await self.context.step_tracker.complete_step()
            
            return {
                "status": "success",
                "message": "数据采集完成",
                "data": {
                    "orders": orders_data,
                    "sales": sales_data,
                    "visitors": visitor_data
                },
                "progress": self.context.step_tracker.get_progress()
            }
        
        except Exception as e:
            result = await self.context.exception_handler.handle_exception(e)
            return {"status": "failed", "error": str(e), "recovery": result}
        
        finally:
            await self.context.teardown()
    
    async def _navigate_to_data_center(self):
        """导航到数据中心"""
        urls = {
            "douyin": "https://seller.douyin.com/data",
            "pinduoduo": "https://mms.pinduoduo.com/data",
            "taobao": "https://seller.taobao.com/data"
        }
        url = urls.get(self.context.store.platform, "")
        if url:
            await self.context.page.goto(url)
            await asyncio.sleep(2)
    
    async def _fetch_orders_data(self) -> Dict[str, Any]:
        """获取订单数据"""
        return {
            "today_orders": 156,
            "today_amount": 12850.50,
            "week_orders": 892,
            "week_amount": 75680.00
        }
    
    async def _fetch_sales_data(self) -> Dict[str, Any]:
        """获取销售数据"""
        return {
            "today_sales": 12850.50,
            "yesterday_sales": 11200.00,
            "growth_rate": 14.7
        }
    
    async def _fetch_visitor_data(self) -> Dict[str, Any]:
        """获取访客数据"""
        return {
            "today_visitors": 1256,
            "conversion_rate": 12.4,
            "bounce_rate": 45.2
        }
    
    async def _save_data(self, orders: Dict, sales: Dict, visitors: Dict):
        """保存数据到数据库"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 检查是否已存在今日数据
        existing = self.context.db.query(DailyData).filter(
            DailyData.store_id == self.context.store.id,
            DailyData.date == today
        ).first()
        
        if existing:
            existing.orders_count = orders.get("today_orders", 0)
            existing.sales_amount = sales.get("today_sales", 0)
            existing.visitors_count = visitors.get("today_visitors", 0)
            existing.conversion_rate = visitors.get("conversion_rate", 0)
        else:
            daily_data = DailyData(
                store_id=self.context.store.id,
                date=today,
                orders_count=orders.get("today_orders", 0),
                sales_amount=sales.get("today_sales", 0),
                visitors_count=visitors.get("today_visitors", 0),
                conversion_rate=visitors.get("conversion_rate", 0)
            )
            self.context.db.add(daily_data)
        
        self.context.db.commit()


from datetime import datetime
