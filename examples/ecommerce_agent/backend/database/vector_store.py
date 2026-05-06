import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from backend.config import settings


class VectorStore:
    """向量数据库管理"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        
        # 使用默认的嵌入函数
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # 创建或获取集合
        self.knowledge_collection = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.embedding_function
        )
        self.experience_collection = self.client.get_or_create_collection(
            name="experience",
            embedding_function=self.embedding_function
        )
    
    def add_knowledge(
        self,
        knowledge_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加知识"""
        self.knowledge_collection.add(
            ids=[knowledge_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def search_knowledge(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索知识"""
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)
    
    def add_experience(
        self,
        experience_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加经验"""
        self.experience_collection.add(
            ids=[experience_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def search_experience(
        self,
        query: str,
        platform: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索经验"""
        where = {"platform": platform} if platform else None
        results = self.experience_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return self._format_results(results)
    
    def delete_knowledge(self, knowledge_id: str):
        """删除知识"""
        self.knowledge_collection.delete(ids=[knowledge_id])
    
    def delete_experience(self, experience_id: str):
        """删除经验"""
        self.experience_collection.delete(ids=[experience_id])
    
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化搜索结果"""
        formatted = []
        if results and results.get("ids"):
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })
        return formatted


vector_store = VectorStore()
