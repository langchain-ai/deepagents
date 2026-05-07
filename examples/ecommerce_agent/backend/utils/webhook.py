import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import httpx


class WebhookAlertManager:
    """Webhook 告警管理器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.webhooks: Dict[str, Dict[str, Any]] = {}  # webhook_id -> config
        self.alert_history: List[Dict[str, Any]] = []
    
    def add_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        name: str = "",
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_times: int = 3,
        timeout: int = 10
    ) -> bool:
        """添加 Webhook"""
        self.webhooks[webhook_id] = {
            "id": webhook_id,
            "name": name or webhook_id,
            "url": url,
            "events": events,
            "secret": secret,
            "headers": headers or {},
            "retry_times": retry_times,
            "timeout": timeout,
            "is_active": True,
            "created_at": datetime.now().isoformat()
        }
        return True
    
    def remove_webhook(self, webhook_id: str) -> bool:
        """移除 Webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    def update_webhook(
        self,
        webhook_id: str,
        **updates
    ) -> bool:
        """更新 Webhook"""
        if webhook_id not in self.webhooks:
            return False
        
        self.webhooks[webhook_id].update(updates)
        return True
    
    def get_webhooks(self) -> List[Dict[str, Any]]:
        """获取所有 Webhook"""
        return list(self.webhooks.values())
    
    def get_webhook(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """获取单个 Webhook"""
        return self.webhooks.get(webhook_id)
    
    async def send_alert(
        self,
        event_type: str,
        data: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """发送告警"""
        # 收集要发送的 Webhooks
        targets = []
        
        if webhook_id:
            webhook = self.webhooks.get(webhook_id)
            if webhook and webhook["is_active"] and event_type in webhook["events"]:
                targets.append(webhook)
        else:
            # 发送给所有订阅该事件的 Webhook
            for webhook in self.webhooks.values():
                if webhook["is_active"] and event_type in webhook["events"]:
                    targets.append(webhook)
        
        if not targets:
            return {
                "status": "no_targets",
                "message": "没有找到匹配的 Webhook"
            }
        
        # 构建消息
        message = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # 发送请求
        results = []
        for webhook in targets:
            result = await self._send_webhook(webhook, message)
            results.append({
                "webhook_id": webhook["id"],
                "result": result
            })
        
        # 记录历史
        self.alert_history.append({
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "targets": len(targets),
            "results": results
        })
        
        return {
            "status": "sent",
            "targets": len(targets),
            "results": results
        }
    
    async def _send_webhook(
        self,
        webhook: Dict[str, Any],
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """发送单个 Webhook 请求"""
        headers = webhook.get("headers", {}).copy()
        headers["Content-Type"] = "application/json"
        
        # 如果有密钥，添加签名
        if webhook.get("secret"):
            import hmac
            import hashlib
            signature = hmac.new(
                webhook["secret"].encode(),
                json.dumps(message).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        # 发送请求（带重试）
        for attempt in range(webhook["retry_times"]):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        webhook["url"],
                        json=message,
                        headers=headers,
                        timeout=webhook["timeout"]
                    )
                    
                    if response.status_code < 400:
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "attempt": attempt + 1
                        }
                    else:
                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "error": f"HTTP {response.status_code}",
                            "attempt": attempt + 1
                        }
            except Exception as e:
                if attempt == webhook["retry_times"] - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """测试 Webhook"""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {"success": False, "error": "Webhook not found"}
        
        # 发送测试消息
        test_message = {
            "event": "test",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "message": "这是一条测试消息"
            }
        }
        
        # 同步发送（简化版）
        import requests
        try:
            response = requests.post(
                webhook["url"],
                json=test_message,
                headers={"Content-Type": "application/json"},
                timeout=webhook["timeout"]
            )
            
            if response.status_code < 400:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response": response.text[:200]
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_alert_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取告警历史"""
        if event_type:
            return [
                alert for alert in self.alert_history[-limit:]
                if alert["event_type"] == event_type
            ]
        return self.alert_history[-limit:]
    
    def clear_alert_history(self):
        """清除告警历史"""
        self.alert_history.clear()


class AlertEventTypes:
    """告警事件类型"""
    
    # 任务相关
    TASK_STARTED = "task:started"
    TASK_COMPLETED = "task:completed"
    TASK_FAILED = "task:failed"
    TASK_PAUSED = "task:paused"
    
    # 店铺相关
    STORE_ONLINE = "store:online"
    STORE_OFFLINE = "store:offline"
    STORE_ERROR = "store:error"
    
    # 系统相关
    SYSTEM_WARNING = "system:warning"
    SYSTEM_ERROR = "system:error"
    SYSTEM_HEALTH = "system:health"
    
    # 资源相关
    RESOURCE_HIGH = "resource:high"
    RESOURCE_CRITICAL = "resource:critical"
    
    # 业务相关
    PRODUCT_PUBLISHED = "product:published"
    GOOD_REVIEW_REPLIED = "good_review:replied"
    DATA_FETCHED = "data:fetched"


class AlertTemplates:
    """告警模板"""
    
    @staticmethod
    def task_failed_template(task_name: str, error: str, store_name: str) -> Dict[str, Any]:
        """任务失败告警模板"""
        return {
            "title": "⚠️ 任务执行失败",
            "content": f"店铺「{store_name}」的任务「{task_name}」执行失败",
            "error": error,
            "action": "请检查任务状态和日志"
        }
    
    @staticmethod
    def task_completed_template(task_name: str, result: str, store_name: str) -> Dict[str, Any]:
        """任务完成告警模板"""
        return {
            "title": "✅ 任务执行成功",
            "content": f"店铺「{store_name}」的任务「{task_name}」已完成",
            "result": result
        }
    
    @staticmethod
    def resource_high_template(resource_type: str, usage: float, threshold: float) -> Dict[str, Any]:
        """资源过高告警模板"""
        return {
            "title": "⚡ 系统资源告警",
            "content": f"{resource_type} 使用率达到 {usage:.1f}%，超过阈值 {threshold:.1f}%",
            "action": "建议检查系统状态"
        }
    
    @staticmethod
    def daily_report_template(store_name: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """每日报告模板"""
        return {
            "title": f"📊 {store_name} 每日运营报告",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "stats": stats
        }


# 全局实例
_alert_manager: Optional[WebhookAlertManager] = None


def get_alert_manager(db: Session) -> WebhookAlertManager:
    """获取告警管理器"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = WebhookAlertManager(db)
    return _alert_manager


class IMIntegration:
    """IM 集成（通过 Webhook）"""
    
    def __init__(self, webhook_manager: WebhookAlertManager):
        self.webhook_manager = webhook_manager
    
    def setup_feishu_webhook(
        self,
        webhook_url: str,
        secret: Optional[str] = None
    ) -> bool:
        """设置飞书 Webhook"""
        return self.webhook_manager.add_webhook(
            webhook_id="feishu",
            url=webhook_url,
            events=[
                AlertEventTypes.TASK_FAILED,
                AlertEventTypes.TASK_COMPLETED,
                AlertEventTypes.SYSTEM_WARNING,
                AlertEventTypes.DAILY_REPORT
            ],
            name="飞书机器人",
            secret=secret
        )
    
    def setup_wecom_webhook(
        self,
        webhook_url: str
    ) -> bool:
        """设置企业微信 Webhook"""
        return self.webhook_manager.add_webhook(
            webhook_id="wecom",
            url=webhook_url,
            events=[
                AlertEventTypes.TASK_FAILED,
                AlertEventTypes.SYSTEM_ERROR
            ],
            name="企业微信机器人"
        )
    
    def setup_dingtalk_webhook(
        self,
        webhook_url: str,
        secret: Optional[str] = None
    ) -> bool:
        """设置钉钉 Webhook"""
        return self.webhook_manager.add_webhook(
            webhook_id="dingtalk",
            url=webhook_url,
            events=[
                AlertEventTypes.TASK_FAILED,
                AlertEventTypes.TASK_COMPLETED,
                AlertEventTypes.SYSTEM_WARNING
            ],
            name="钉钉机器人",
            secret=secret
        )
    
    async def send_daily_report(
        self,
        store_name: str,
        stats: Dict[str, Any]
    ):
        """发送每日报告"""
        template = AlertTemplates.daily_report_template(store_name, stats)
        await self.webhook_manager.send_alert(
            event_type="daily:report",
            data=template
        )
