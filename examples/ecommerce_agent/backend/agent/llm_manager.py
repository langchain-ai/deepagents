from typing import Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """LLM 提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # 本地大模型
    LOCAL = "local"


class LLMManager:
    """大语言模型管理器"""
    
    def __init__(self):
        self.current_provider = LLMProvider.OPENAI
        self.providers = {
            LLMProvider.OPENAI: None,
            LLMProvider.ANTHROPIC: None,
            LLMProvider.OLLAMA: None,
            LLMProvider.LOCAL: None
        }
        self.config = {}
    
    def configure(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        local_endpoint: Optional[str] = None
    ):
        """配置 LLM 提供商"""
        provider_enum = LLMProvider(provider)
        
        if provider_enum == LLMProvider.OPENAI:
            self._configure_openai(api_key, base_url, model_name)
        elif provider_enum == LLMProvider.ANTHROPIC:
            self._configure_anthropic(api_key, model_name)
        elif provider_enum == LLMProvider.OLLAMA:
            self._configure_ollama(local_endpoint, model_name)
        elif provider_enum == LLMProvider.LOCAL:
            self._configure_local(local_endpoint, model_name)
        
        self.current_provider = provider_enum
        self.config[provider_enum] = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "local_endpoint": local_endpoint
        }
    
    def _configure_openai(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model_name: Optional[str]
    ):
        """配置 OpenAI"""
        from langchain_openai import ChatOpenAI
        
        self.providers[LLMProvider.OPENAI] = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name or "gpt-4o",
            temperature=0.7
        )
    
    def _configure_anthropic(
        self,
        api_key: Optional[str],
        model_name: Optional[str]
    ):
        """配置 Anthropic"""
        from langchain_anthropic import ChatAnthropic
        
        self.providers[LLMProvider.ANTHROPIC] = ChatAnthropic(
            api_key=api_key,
            model=model_name or "claude-3-sonnet-20240229",
            temperature=0.7
        )
    
    def _configure_ollama(
        self,
        endpoint: Optional[str],
        model_name: Optional[str]
    ):
        """配置 Ollama 本地模型"""
        from langchain_ollama import ChatOllama
        
        self.providers[LLMProvider.OLLAMA] = ChatOllama(
            base_url=endpoint or "http://localhost:11434",
            model=model_name or "qwen2.5",
            temperature=0.7
        )
    
    def _configure_local(
        self,
        endpoint: Optional[str],
        model_name: Optional[str]
    ):
        """配置其他本地模型"""
        # 预留其他本地模型的配置接口
        pass
    
    def get_llm(self):
        """获取当前 LLM 实例"""
        return self.providers.get(self.current_provider)
    
    def switch_provider(self, provider: str):
        """切换 LLM 提供商"""
        provider_enum = LLMProvider(provider)
        if provider_enum in self.providers and self.providers[provider_enum]:
            self.current_provider = provider_enum
            return True
        return False
    
    def get_current_provider(self) -> str:
        """获取当前提供商"""
        return self.current_provider.value
    
    def is_local(self) -> bool:
        """是否使用本地模型"""
        return self.current_provider in [LLMProvider.OLLAMA, LLMProvider.LOCAL]
    
    def get_available_providers(self) -> list:
        """获取可用的提供商列表"""
        available = []
        for provider in self.providers:
            if self.providers[provider]:
                available.append(provider.value)
        return available
    
    def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            llm = self.get_llm()
            if not llm:
                return {"success": False, "message": "LLM 未配置"}
            
            # 简单的测试调用
            response = llm.invoke("Hello")
            return {
                "success": True,
                "message": "连接成功",
                "response": str(response)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"连接失败: {str(e)}"
            }


# 全局实例
llm_manager = LLMManager()


class OllamaSetup:
    """Ollama 本地模型设置"""
    
    @staticmethod
    def check_ollama_running() -> bool:
        """检查 Ollama 是否运行"""
        import requests
        try:
            response = requests.get("http://localhost:11434", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def get_available_models() -> list:
        """获取可用的模型列表"""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
    
    @staticmethod
    def pull_model(model_name: str) -> bool:
        """拉取模型"""
        import subprocess
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            return True
        except:
            return False
    
    @staticmethod
    def recommended_models() -> list:
        """推荐模型"""
        return [
            {
                "name": "qwen2.5",
                "display_name": "Qwen 2.5",
                "description": "通识能力强，适合日常任务",
                "size": "~4GB"
            },
            {
                "name": "llama3.2",
                "display_name": "Llama 3.2",
                "description": "Meta 开源模型，表现优秀",
                "size": "~2GB"
            },
            {
                "name": "deepseek-r1",
                "display_name": "DeepSeek R1",
                "description": "专注推理和分析任务",
                "size": "~4GB"
            },
            {
                "name": "phi3",
                "display_name": "Phi-3",
                "description": "微软轻量级模型，速度快",
                "size": "~2GB"
            }
        ]
