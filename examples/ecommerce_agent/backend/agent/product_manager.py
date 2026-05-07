from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from backend.database.models import Store


class ProductManager:
    """商品管理器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def fetch_products(self, store: Store) -> List[Dict[str, Any]]:
        """获取商品列表"""
        categories = ["数码产品", "服装鞋帽", "家居用品", "食品饮料", "美妆护肤"]
        
        return [
            {
                "product_id": f"PRD{store.id}{i:06d}",
                "store_id": store.id,
                "store_name": store.name,
                "platform": store.platform,
                "title": f"商品{i + 1}: 精选商品标题",
                "category": categories[i % len(categories)],
                "price": round(29.9 + i * 15.5, 2),
                "original_price": round(49.9 + i * 20.0, 2),
                "stock": 100 + i * 10,
                "sales": 50 + i * 20,
                "status": self._get_random_status(),
                "create_time": (datetime.now() - timedelta(days=i % 30)).isoformat(),
                "update_time": (datetime.now() - timedelta(hours=i)).isoformat()
            }
            for i in range(15)
        ]
    
    def _get_random_status(self) -> str:
        """获取随机商品状态"""
        import random
        statuses = ["上架中", "已下架", "审核中", "违规下架"]
        return random.choice(statuses)
    
    async def get_product_stats(self, store: Store) -> Dict[str, Any]:
        """获取商品统计"""
        products = await self.fetch_products(store)
        
        stats = {
            "total": len(products),
            "online": sum(1 for p in products if p["status"] == "上架中"),
            "offline": sum(1 for p in products if p["status"] == "已下架"),
            "total_sales": sum(p["sales"] for p in products),
            "total_stock": sum(p["stock"] for p in products),
            "category_distribution": {},
            "avg_price": round(sum(p["price"] for p in products) / len(products), 2)
        }
        
        # 分类分布
        for product in products:
            category = product["category"]
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        return stats
    
    async def get_product_details(self, store: Store, product_id: str) -> Optional[Dict[str, Any]]:
        """获取商品详情"""
        products = await self.fetch_products(store)
        return next((p for p in products if p["product_id"] == product_id), None)
    
    async def update_product(self, store: Store, product_id: str, updates: Dict[str, Any]) -> bool:
        """更新商品信息"""
        products = await self.fetch_products(store)
        product = next((p for p in products if p["product_id"] == product_id), None)
        
        if product:
            product.update(updates)
            return True
        return False
    
    async def batch_update_status(self, store: Store, product_ids: List[str], status: str) -> Dict[str, bool]:
        """批量更新商品状态"""
        results = {}
        products = await self.fetch_products(store)
        
        for product_id in product_ids:
            product = next((p for p in products if p["product_id"] == product_id), None)
            if product:
                product["status"] = status
                results[product_id] = True
            else:
                results[product_id] = False
        
        return results
    
    async def get_low_stock_products(self, store: Store, threshold: int = 20) -> List[Dict[str, Any]]:
        """获取库存不足的商品"""
        products = await self.fetch_products(store)
        return [p for p in products if p["stock"] < threshold]
    
    async def get_top_selling_products(self, store: Store, limit: int = 10) -> List[Dict[str, Any]]:
        """获取畅销商品"""
        products = await self.fetch_products(store)
        return sorted(products, key=lambda p: p["sales"], reverse=True)[:limit]
    
    async def export_products(self, store: Store) -> str:
        """导出商品数据"""
        products = await self.fetch_products(store)
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 表头
        writer.writerow([
            "商品ID", "店铺", "平台", "标题", "分类", "价格", 
            "原价", "库存", "销量", "状态", "创建时间", "更新时间"
        ])
        
        # 数据
        for product in products:
            writer.writerow([
                product["product_id"],
                product["store_name"],
                product["platform"],
                product["title"],
                product["category"],
                product["price"],
                product["original_price"],
                product["stock"],
                product["sales"],
                product["status"],
                product["create_time"],
                product["update_time"]
            ])
        
        return output.getvalue()
    
    async def generate_product_report(self, store: Store) -> Dict[str, Any]:
        """生成商品报告"""
        stats = await self.get_product_stats(store)
        top_selling = await self.get_top_selling_products(store, 5)
        low_stock = await self.get_low_stock_products(store)
        
        return {
            "store_name": store.name,
            "platform": store.platform,
            "report_date": datetime.now().isoformat(),
            "stats": stats,
            "top_selling": top_selling,
            "low_stock": low_stock,
            "recommendations": self._generate_recommendations(stats, low_stock)
        }
    
    def _generate_recommendations(self, stats: Dict, low_stock: List) -> List[str]:
        """生成运营建议"""
        recommendations = []
        
        if stats["offline"] > stats["online"]:
            recommendations.append("建议检查下架商品，及时重新上架")
        
        if len(low_stock) > 3:
            recommendations.append(f"有 {len(low_stock)} 件商品库存不足，建议补货")
        
        avg_sales = stats["total_sales"] / stats["total"] if stats["total"] > 0 else 0
        if avg_sales < 10:
            recommendations.append("整体销量较低，建议优化商品标题和描述")
        
        return recommendations
