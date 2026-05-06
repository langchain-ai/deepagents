from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """应用配置"""
    APP_NAME: str = "Ecommerce Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_DIR: Path = DATA_DIR / "db"
    SCREENSHOT_DIR: Path = DATA_DIR / "screenshots"
    RECORDING_DIR: Path = DATA_DIR / "recordings"
    PROFILE_DIR: Path = DATA_DIR / "profiles"
    CONFIG_DIR: Path = BASE_DIR / "configs"
    
    # 数据库
    DATABASE_URL: str = f"sqlite:///{DB_DIR / 'ecommerce.db'}"
    
    # Chroma 向量库
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma"
    
    # 浏览器
    HEADLESS: bool = False
    BROWSER_TIMEOUT: int = 30000
    
    # 防检测
    ENABLE_ANTI_DETECT: bool = True
    MIN_CLICK_INTERVAL: float = 0.5
    MAX_DAILY_PRODUCTS: int = 50
    
    # 截图/录屏
    ENABLE_SCREENSHOT: bool = True
    ENABLE_RECORDING: bool = True
    SCREENSHOT_RETENTION_DAYS: int = 7
    RECORDING_RETENTION_DAYS: int = 7
    
    # Tab 回收
    TAB_IDLE_TIMEOUT_MINUTES: int = 30
    
    # 加密
    ENCRYPTION_KEY: str = "your-encryption-key-change-this-in-production"
    
    # 模型配置
    MODEL_PROVIDER: str = "openai"
    MODEL_NAME: str = "gpt-4o"
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        for dir_path in [
            self.DB_DIR,
            self.SCREENSHOT_DIR,
            self.RECORDING_DIR,
            self.PROFILE_DIR,
            self.CONFIG_DIR,
            self.CHROMA_PERSIST_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
