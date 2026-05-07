from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from backend.database.models import Store


class OrderManager:
    """订单管理器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def fetch_orders(self, store: Store, days: int = 7) -> List[Dict[str, Any]]:
        """获取订单列表"""
        # 模拟获取订单数据
        return [
            {
                "order_id": f"ORD{store.id}{i:06d}",
                "store_id": store.id,
                "store_name": store.name,
                "platform": store.platform,
                "status": self._get_random_status(),
                "amount": round(10 + i * 5.5, 2),
                "item_count": 1 + i % 3,
                "buyer_name": f"买家{i}",
                "create_time": (datetime.now() - timedelta(days=i % 7)).isoformat(),
                "pay_time": (datetime.now() - timedelta(days=i % 7, hours=i)).isoformat()
            }
            for i in range(10)
        ]
    
    def _get_random_status(self) -> str:
        """获取随机订单状态"""
        import random
        statuses = ["待付款", "待发货", "已发货", "待收货", "已完成", "已取消"]
        return random.choice(statuses)
    
    async def get_order_stats(self, store: Store, days: int = 7) -> Dict[str, Any]:
        """获取订单统计"""
        orders = await self.fetch_orders(store, days)
        
        stats = {
            "total": len(orders),
            "total_amount": sum(o["amount"] for o in orders),
            "status_distribution": {},
            "daily_orders": {}
        }
        
        # 状态分布
        for order in orders:
            status = order["status"]
            stats["status_distribution"][status] = stats["status_distribution"].get(status, 0) + 1
        
        # 每日订单数
        for order in orders:
            date = order["create_time"][:10]
            stats["daily_orders"][date] = stats["daily_orders"].get(date, 0) + 1
        
        return stats
    
    async def get_unshipped_orders(self, store: Store) -> List[Dict[str, Any]]:
        """获取待发货订单"""
        orders = await self.fetch_orders(store)
        return [o for o in orders if o["status"] == "待发货"]
    
    async def get_order_details(self, store: Store, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单详情"""
        orders = await self.fetch_orders(store)
        return next((o for o in orders if o["order_id"] == order_id), None)
    
    async def update_order_status(self, store: Store, order_id: str, status: str) -> bool:
        """更新订单状态"""
        orders = await self.fetch_orders(store)
        order = next((o for o in orders if o["order_id"] == order_id), None)
        
        if order:
            order["status"] = status
            return True
        return False
    
    async def batch_update_status(self, store: Store, order_ids: List[str], status: str) -> Dict[str, bool]:
        """批量更新订单状态"""
        results = {}
        for order_id in order_ids:
            results[order_id] = await self.update_order_status(store, order_id, status)
        return results
    
    async def export_orders(self, store: Store, days: int = 30) -> str:
        """导出订单数据"""
        orders = await self.fetch_orders(store, days)
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 表头
        writer.writerow([
            "订单号", "店铺", "平台", "状态", "金额", "商品数量", 
            "买家", "创建时间", "付款时间"
        ])
        
        # 数据
        for order in orders:
            writer.writerow([
                order["order_id"],
                order["store_name"],
                order["platform"],
                order["status"],
                order["amount"],
                order["item_count"],
                order["buyer_name"],
                order["create_time"],
                order["pay_time"]
            ])
        
        return output.getvalue()
