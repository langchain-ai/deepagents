import random
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta


class AntiDetectManager:
    """防检测反风控管理器"""
    
    def __init__(self):
        self.last_click_time: Dict[int, float] = {}
        self.daily_product_count: Dict[int, int] = {}
        self.last_reset_date: str = ""
    
    def generate_fingerprint(self) -> Dict[str, Any]:
        """生成浏览器指纹"""
        return {
            "user_agent": self._get_random_user_agent(),
            "viewport": self._get_random_viewport(),
            "webgl_vendor": self._get_random_webgl_vendor(),
            "webgl_renderer": self._get_random_webgl_renderer(),
            "canvas_fingerprint": self._generate_canvas_fingerprint(),
            "audio_fingerprint": self._generate_audio_fingerprint(),
            "languages": ["zh-CN", "zh", "en"],
            "platform": "Win32",
            "device_memory": random.choice([4, 8, 16]),
            "hardware_concurrency": random.choice([4, 6, 8, 12]),
            "timezone": "Asia/Shanghai",
        }
    
    def _get_random_user_agent(self) -> str:
        """获取随机 User-Agent"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ]
        return random.choice(user_agents)
    
    def _get_random_viewport(self) -> Dict[str, int]:
        """获取随机视口大小"""
        viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1536, "height": 864},
            {"width": 1440, "height": 900},
            {"width": 2560, "height": 1440},
        ]
        return random.choice(viewports)
    
    def _get_random_webgl_vendor(self) -> str:
        """获取随机 WebGL 厂商"""
        vendors = [
            "Google Inc. (NVIDIA)",
            "Intel Inc.",
            "AMD",
            "NVIDIA Corporation",
        ]
        return random.choice(vendors)
    
    def _get_random_webgl_renderer(self) -> str:
        """获取随机 WebGL 渲染器"""
        renderers = [
            "ANGLE (NVIDIA, NVIDIA GeForce GTX 1050 Ti Direct3D11 vs_5_0 ps_5_0, D3D11)",
            "ANGLE (Intel, Intel(R) HD Graphics 530 Direct3D11 vs_5_0 ps_5_0, D3D11)",
            "ANGLE (AMD, Radeon RX 580 Series Direct3D11 vs_5_0 ps_5_0, D3D11)",
        ]
        return random.choice(renderers)
    
    def _generate_canvas_fingerprint(self) -> str:
        """生成 Canvas 指纹"""
        return ''.join(random.choices('0123456789abcdef', k=64))
    
    def _generate_audio_fingerprint(self) -> str:
        """生成音频指纹"""
        return str(random.uniform(1000, 10000))
    
    def random_delay(self, min_delay: float = 0.5, max_delay: float = 2.0):
        """随机延迟"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def human_like_click_delay(self, store_id: int):
        """人类行为点击延迟"""
        from backend.config import settings
        
        # 检查每日发布数量
        self._check_daily_limit(store_id)
        
        # 最小延迟
        if store_id in self.last_click_time:
            elapsed = time.time() - self.last_click_time[store_id]
            if elapsed < settings.MIN_CLICK_INTERVAL:
                time.sleep(settings.MIN_CLICK_INTERVAL - elapsed)
        
        # 添加随机延迟
        self.random_delay(0.3, 1.5)
        self.last_click_time[store_id] = time.time()
    
    def _check_daily_limit(self, store_id: int):
        """检查每日限制"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.last_reset_date != today:
            self.daily_product_count = {}
            self.last_reset_date = today
        
        if store_id not in self.daily_product_count:
            self.daily_product_count[store_id] = 0
        
        from backend.config import settings
        if self.daily_product_count[store_id] >= settings.MAX_DAILY_PRODUCTS:
            raise Exception(f"已达到每日发布限制 ({settings.MAX_DAILY_PRODUCTS})")
    
    def increment_product_count(self, store_id: int):
        """增加产品计数"""
        if store_id not in self.daily_product_count:
            self.daily_product_count[store_id] = 0
        self.daily_product_count[store_id] += 1
    
    def type_with_delay(self, text: str, min_char_delay: float = 0.05, max_char_delay: float = 0.15) -> list:
        """带延迟的打字，返回按键序列"""
        actions = []
        for char in text:
            # 随机输入几个字符后暂停
            if random.random() < 0.3:
                actions.append({
                    "type": "pause",
                    "duration": random.uniform(0.1, 0.3)
                })
            actions.append({
                "type": "type",
                "char": char,
                "delay": random.uniform(min_char_delay, max_char_delay)
            })
        return actions
    
    def random_scroll(self) -> Dict[str, Any]:
        """随机滚动"""
        return {
            "x": random.randint(0, 100),
            "y": random.randint(0, 200),
            "delay": random.uniform(0.1, 0.3)
        }
    
    def mouse_move_path(self, from_x: int, from_y: int, to_x: int, to_y: int) -> list:
        """生成鼠标移动路径"""
        path = []
        steps = random.randint(5, 15)
        
        for i in range(steps):
            progress = i / steps
            # 添加贝塞尔曲线效果
            x = from_x + (to_x - from_x) * progress + random.randint(-10, 10)
            y = from_y + (to_y - from_y) * progress + random.randint(-10, 10)
            path.append({"x": x, "y": y})
        
        path.append({"x": to_x, "y": to_y})
        return path
    
    def save_fingerprint(self, store_id: int, fingerprint: Dict[str, Any], profile_path: Path):
        """保存指纹"""
        fingerprint_file = profile_path / "fingerprint.json"
        with open(fingerprint_file, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, ensure_ascii=False, indent=2)
    
    def load_fingerprint(self, store_id: int, profile_path: Path) -> Optional[Dict[str, Any]]:
        """加载指纹"""
        fingerprint_file = profile_path / "fingerprint.json"
        if fingerprint_file.exists():
            with open(fingerprint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


anti_detect = AntiDetectManager()
