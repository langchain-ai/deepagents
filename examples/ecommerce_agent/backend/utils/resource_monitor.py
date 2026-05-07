import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.alert_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def get_current_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "per_cpu": psutil.cpu_percent(interval=0.1, percpu=True)
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "process": {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "memory_percent": self.process.memory_percent(),
                "threads": self.process.num_threads(),
                "open_files": len(self.process.open_files()),
                "connections": len(self.process.connections())
            }
        }
    
    def check_thresholds(self) -> Dict[str, Any]:
        """检查资源阈值"""
        usage = self.get_current_usage()
        alerts = []
        
        # CPU 检查
        if usage["cpu"]["percent"] > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu",
                "level": "warning",
                "message": f"CPU 使用率过高: {usage['cpu']['percent']:.1f}%",
                "value": usage["cpu"]["percent"]
            })
        
        # 内存检查
        if usage["memory"]["percent"] > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "memory",
                "level": "warning",
                "message": f"内存使用率过高: {usage['memory']['percent']:.1f}%",
                "value": usage["memory"]["percent"]
            })
        
        # 进程内存检查
        if usage["process"]["memory_percent"] > 50:
            alerts.append({
                "type": "process_memory",
                "level": "warning",
                "message": f"进程内存使用率过高: {usage['process']['memory_percent']:.1f}%",
                "value": usage["process"]["memory_percent"]
            })
        
        # 磁盘检查
        if usage["disk"]["percent"] > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "disk",
                "level": "critical",
                "message": f"磁盘使用率过高: {usage['disk']['percent']:.1f}%",
                "value": usage["disk"]["percent"]
            })
        
        # 保存警报
        self.alerts.extend(alerts)
        
        return {
            "has_alerts": len(alerts) > 0,
            "alerts": alerts,
            "usage": usage
        }
    
    def set_threshold(self, resource: str, threshold: float):
        """设置阈值"""
        if resource in self.alert_thresholds:
            self.alert_thresholds[resource] = threshold
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的警报"""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """清除警报"""
        self.alerts.clear()
    
    def should_trigger_gc(self) -> bool:
        """检查是否应该触发垃圾回收"""
        usage = self.get_current_usage()
        
        # 如果进程内存超过 2GB，考虑触发 GC
        if usage["process"]["memory_mb"] > 2048:
            return True
        
        # 如果内存使用率超过 90%
        if usage["memory"]["percent"] > 90:
            return True
        
        return False
    
    def trigger_gc(self):
        """触发垃圾回收"""
        import gc
        collected = gc.collect()
        return {"collected": collected}
    
    def get_browser_processes(self) -> List[Dict[str, Any]]:
        """获取浏览器进程信息"""
        browser_procs = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                if 'chrome' in proc.info['name'].lower() or 'playwright' in proc.info['name'].lower():
                    browser_procs.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_mb": proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return browser_procs
    
    def kill_browser_processes(self) -> int:
        """关闭所有浏览器进程（释放内存）"""
        killed_count = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return killed_count
    
    def restart_browser_if_needed(self) -> Dict[str, Any]:
        """如果内存泄漏则重启浏览器"""
        if self.should_trigger_gc():
            # 先触发 GC
            gc_result = self.trigger_gc()
            
            # 获取当前使用情况
            usage = self.get_current_usage()
            
            # 如果仍然过高，关闭浏览器
            if usage["process"]["memory_mb"] > 2048:
                killed = self.kill_browser_processes()
                return {
                    "action": "browser_killed",
                    "killed_processes": killed,
                    "gc_result": gc_result
                }
        
        return {"action": "no_action_needed"}


class MonitorScheduler:
    """监控调度器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.monitor = ResourceMonitor()
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.check_interval = 60  # 每分钟检查一次
    
    async def start(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        print("资源监控已启动")
    
    async def stop(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        print("资源监控已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检查资源
                result = self.monitor.check_thresholds()
                
                # 如果有警报
                if result["has_alerts"]:
                    print(f"资源警报: {result['alerts']}")
                    
                    # 如果是内存问题，尝试重启浏览器
                    if any(a["type"] in ["memory", "process_memory"] for a in result["alerts"]):
                        restart_result = self.monitor.restart_browser_if_needed()
                        print(f"浏览器重启: {restart_result}")
                
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"监控错误: {e}")
                await asyncio.sleep(self.check_interval)


# 全局实例
resource_monitor = ResourceMonitor()


class SystemHealthChecker:
    """系统健康检查"""
    
    def __init__(self, db: Session):
        self.db = db
        self.monitor = ResourceMonitor()
    
    async def check_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        usage = self.monitor.get_current_usage()
        
        # 基础健康检查
        health_score = 100
        
        # CPU
        if usage["cpu"]["percent"] > 90:
            health_score -= 30
        elif usage["cpu"]["percent"] > 70:
            health_score -= 15
        
        # 内存
        if usage["memory"]["percent"] > 90:
            health_score -= 30
        elif usage["memory"]["percent"] > 80:
            health_score -= 15
        
        # 磁盘
        if usage["disk"]["percent"] > 95:
            health_score -= 30
        elif usage["disk"]["percent"] > 85:
            health_score -= 15
        
        status = "healthy"
        if health_score < 50:
            status = "critical"
        elif health_score < 75:
            status = "warning"
        
        return {
            "status": status,
            "health_score": max(0, health_score),
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
            "recommendations": self._get_recommendations(usage)
        }
    
    def _get_recommendations(self, usage: Dict[str, Any]) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        if usage["cpu"]["percent"] > 80:
            recommendations.append("考虑减少并发任务数量")
        
        if usage["memory"]["percent"] > 85:
            recommendations.append("考虑关闭闲置的浏览器标签页")
            recommendations.append("可以考虑重启应用程序")
        
        if usage["disk"]["percent"] > 85:
            recommendations.append("清理旧的截图和录屏文件")
            recommendations.append("清理浏览器缓存")
        
        return recommendations
