import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import (
    Store, Task, ScheduledTask, DOMElement,
    KnowledgeItem, ExperienceItem
)


class ConfigExporter:
    """配置导出器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def export_store(self, store_id: int) -> Optional[Dict[str, Any]]:
        """导出单个店铺配置"""
        store = self.db.query(Store).get(store_id)
        if not store:
            return None
        
        # 导出基本信息（不含敏感密码）
        config = {
            "type": "store",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "data": {
                "name": store.name,
                "platform": store.platform,
                "username": store.username,
                "is_active": store.is_active,
                "fingerprint": store.fingerprint
            }
        }
        
        # 导出关联的定时任务
        scheduled_tasks = self.db.query(ScheduledTask).filter(
            ScheduledTask.store_id == store_id
        ).all()
        
        config["data"]["scheduled_tasks"] = [
            {
                "name": task.name,
                "task_type": task.task_type,
                "cron_expression": task.cron_expression,
                "is_active": task.is_active
            }
            for task in scheduled_tasks
        ]
        
        return config
    
    def export_all_stores(self) -> List[Dict[str, Any]]:
        """导出所有店铺"""
        stores = self.db.query(Store).all()
        return [self.export_store(store.id) for store in stores if store.id]
    
    def export_dom_elements(self, platform: Optional[str] = None) -> Dict[str, Any]:
        """导出 DOM 元素配置"""
        query = self.db.query(DOMElement)
        if platform:
            query = query.filter(DOMElement.platform == platform)
        
        elements = query.all()
        
        # 按平台和页面组织
        organized = {}
        for elem in elements:
            if elem.platform not in organized:
                organized[elem.platform] = {}
            
            if elem.page not in organized[elem.platform]:
                organized[elem.platform][elem.page] = []
            
            organized[elem.platform][elem.page].append({
                "name": elem.name,
                "selectors": elem.selectors,
                "description": elem.description,
                "version": elem.version
            })
        
        return {
            "type": "dom_elements",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "data": organized
        }
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """导出知识库"""
        items = self.db.query(KnowledgeItem).all()
        
        return {
            "type": "knowledge_base",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "data": [
                {
                    "type": item.type,
                    "title": item.title,
                    "content": item.content,
                    "metadata": item.extra_data
                }
                for item in items
            ]
        }
    
    def export_experience_base(self) -> Dict[str, Any]:
        """导出经验库"""
        items = self.db.query(ExperienceItem).all()
        
        return {
            "type": "experience_base",
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "data": [
                {
                    "platform": item.platform,
                    "scenario": item.scenario,
                    "content": item.content,
                    "is_success": item.is_success,
                    "metadata": item.extra_data
                }
                for item in items
            ]
        }
    
    def export_full_config(self) -> Dict[str, Any]:
        """导出完整配置"""
        return {
            "stores": self.export_all_stores(),
            "dom_elements": self.export_dom_elements(),
            "knowledge_base": self.export_knowledge_base(),
            "experience_base": self.export_experience_base(),
            "exported_at": datetime.now().isoformat()
        }
    
    def export_to_file(self, filepath: str, config_type: str = "full"):
        """导出配置到文件"""
        if config_type == "full":
            config = self.export_full_config()
        elif config_type == "stores":
            config = {"stores": self.export_all_stores()}
        elif config_type == "dom_elements":
            config = self.export_dom_elements()
        else:
            config = {}
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def export_to_zip(self, zip_path: str):
        """导出配置到压缩包"""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # 导出各个部分
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 店铺配置
            stores_data = json.dumps({"stores": self.export_all_stores()}, ensure_ascii=False, indent=2)
            zipf.writestr(f"stores_{timestamp}.json", stores_data)
            
            # DOM 元素配置
            dom_data = json.dumps(self.export_dom_elements(), ensure_ascii=False, indent=2)
            zipf.writestr(f"dom_elements_{timestamp}.json", dom_data)
            
            # 知识库
            kb_data = json.dumps(self.export_knowledge_base(), ensure_ascii=False, indent=2)
            zipf.writestr(f"knowledge_base_{timestamp}.json", kb_data)
            
            # 经验库
            exp_data = json.dumps(self.export_experience_base(), ensure_ascii=False, indent=2)
            zipf.writestr(f"experience_base_{timestamp}.json", exp_data)
        
        return zip_path


class ConfigImporter:
    """配置导入器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.import_results = []
    
    def import_store(self, config: Dict[str, Any], replace: bool = False) -> Dict[str, Any]:
        """导入店铺配置"""
        data = config.get("data", {})
        
        # 检查是否已存在同名店铺
        existing = self.db.query(Store).filter(
            Store.name == data["name"],
            Store.platform == data["platform"]
        ).first()
        
        if existing:
            if replace:
                # 替换现有店铺
                existing.username = data["username"]
                existing.is_active = data["is_active"]
                existing.fingerprint = data.get("fingerprint")
                self.db.commit()
                
                result = {
                    "status": "replaced",
                    "store_id": existing.id,
                    "name": data["name"]
                }
            else:
                result = {
                    "status": "skipped",
                    "message": "店铺已存在",
                    "name": data["name"]
                }
        else:
            # 创建新店铺
            store = Store(
                name=data["name"],
                platform=data["platform"],
                username=data.get("username", ""),
                is_active=data.get("is_active", True),
                fingerprint=data.get("fingerprint")
            )
            self.db.add(store)
            self.db.commit()
            
            # 导入定时任务
            for task_config in data.get("scheduled_tasks", []):
                scheduled_task = ScheduledTask(
                    store_id=store.id,
                    task_type=task_config["task_type"],
                    name=task_config["name"],
                    cron_expression=task_config["cron_expression"],
                    is_active=task_config.get("is_active", True)
                )
                self.db.add(scheduled_task)
            
            self.db.commit()
            
            result = {
                "status": "created",
                "store_id": store.id,
                "name": data["name"]
            }
        
        self.import_results.append(result)
        return result
    
    def import_dom_elements(self, config: Dict[str, Any], platform: Optional[str] = None) -> int:
        """导入 DOM 元素配置"""
        data = config.get("data", {})
        imported_count = 0
        
        for plat, pages in data.items():
            if platform and plat != platform:
                continue
            
            for page, elements in pages.items():
                for elem_data in elements:
                    # 检查是否已存在
                    existing = self.db.query(DOMElement).filter(
                        DOMElement.platform == plat,
                        DOMElement.page == page,
                        DOMElement.name == elem_data["name"]
                    ).first()
                    
                    if existing:
                        # 更新现有元素
                        existing.selectors = elem_data["selectors"]
                        existing.description = elem_data.get("description", "")
                        existing.version += 1
                    else:
                        # 创建新元素
                        elem = DOMElement(
                            platform=plat,
                            page=page,
                            name=elem_data["name"],
                            selectors=elem_data["selectors"],
                            description=elem_data.get("description", ""),
                            version=elem_data.get("version", 1)
                        )
                        self.db.add(elem)
                    
                    imported_count += 1
        
        self.db.commit()
        return imported_count
    
    def import_knowledge_base(self, config: Dict[str, Any]) -> int:
        """导入知识库"""
        data = config.get("data", [])
        imported_count = 0
        
        for item_data in data:
            item = KnowledgeItem(
                type=item_data["type"],
                title=item_data["title"],
                content=item_data["content"],
                metadata=item_data.get("metadata")
            )
            self.db.add(item)
            imported_count += 1
        
        self.db.commit()
        return imported_count
    
    def import_experience_base(self, config: Dict[str, Any]) -> int:
        """导入经验库"""
        data = config.get("data", [])
        imported_count = 0
        
        for item_data in data:
            item = ExperienceItem(
                platform=item_data.get("platform"),
                scenario=item_data.get("scenario"),
                content=item_data["content"],
                is_success=item_data.get("is_success", True),
                metadata=item_data.get("metadata")
            )
            self.db.add(item)
            imported_count += 1
        
        self.db.commit()
        return imported_count
    
    def import_from_file(self, filepath: str) -> Dict[str, Any]:
        """从文件导入配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        results = {}
        
        # 导入各个部分
        if "stores" in config:
            for store_config in config["stores"]:
                self.import_store(store_config)
            results["stores"] = len(config.get("stores", []))
        
        if "dom_elements" in config:
            results["dom_elements"] = self.import_dom_elements(config["dom_elements"])
        
        if "knowledge_base" in config:
            results["knowledge_base"] = self.import_knowledge_base(config["knowledge_base"])
        
        if "experience_base" in config:
            results["experience_base"] = self.import_experience_base(config["experience_base"])
        
        return results
    
    def import_from_zip(self, zip_path: str) -> Dict[str, Any]:
        """从压缩包导入配置"""
        results = {}
        
        with zipfile.ZipFile(zip_path, "r") as zipf:
            for filename in zipf.namelist():
                if filename.endswith(".json"):
                    content = zipf.read(filename)
                    config = json.loads(content)
                    
                    if "stores" in filename:
                        data = json.loads(content)
                        for store_config in data.get("stores", []):
                            self.import_store(store_config)
                        results["stores"] = len(data.get("stores", []))
                    
                    elif "dom_elements" in filename:
                        results["dom_elements"] = self.import_dom_elements(config)
                    
                    elif "knowledge_base" in filename:
                        results["knowledge_base"] = self.import_knowledge_base(config)
                    
                    elif "experience_base" in filename:
                        results["experience_base"] = self.import_experience_base(config)
        
        return results
    
    def get_import_results(self) -> List[Dict[str, Any]]:
        """获取导入结果"""
        return self.import_results
