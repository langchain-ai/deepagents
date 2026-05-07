import base64
import os
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database.models import Store


class AccountEncryption:
    """账号加密管理器"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.key = self._get_or_create_key(encryption_key)
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self, encryption_key: Optional[str]) -> bytes:
        """获取或创建加密密钥"""
        if encryption_key:
            return self._derive_key(encryption_key)
        else:
            # 使用默认密钥
            return self._derive_key(settings.ENCRYPTION_KEY)
    
    def _derive_key(self, password: str) -> bytes:
        """从密码派生密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"ecommerce_agent_salt",  # 生产环境应该随机生成
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, plaintext: str) -> str:
        """加密文本"""
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """解密文本"""
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()
    
    def encrypt_store_credentials(self, db: Session, store_id: int):
        """加密店铺凭证"""
        store = db.query(Store).get(store_id)
        if not store:
            return False
        
        # 加密密码
        if store.password:
            store.password = self.encrypt(store.password)
        
        db.commit()
        return True
    
    def decrypt_store_credentials(self, store: Store) -> dict:
        """解密店铺凭证"""
        return {
            "username": store.username,
            "password": self.decrypt(store.password) if store.password else ""
        }


class CredentialManager:
    """凭证管理器"""
    
    def __init__(self, db: Session):
        self.db = db
        self.encryption = AccountEncryption()
    
    def save_credentials(
        self,
        store_id: int,
        username: str,
        password: str
    ):
        """保存凭证"""
        store = self.db.query(Store).get(store_id)
        if not store:
            raise ValueError(f"Store {store_id} not found")
        
        # 加密保存
        store.username = username
        store.password = self.encryption.encrypt(password)
        self.db.commit()
    
    def get_credentials(self, store_id: int) -> Optional[dict]:
        """获取凭证"""
        store = self.db.query(Store).get(store_id)
        if not store:
            return None
        
        if not store.password:
            return None
        
        return self.encryption.decrypt_store_credentials(store)
    
    def delete_credentials(self, store_id: int):
        """删除凭证"""
        store = self.db.query(Store).get(store_id)
        if store:
            store.username = ""
            store.password = ""
            self.db.commit()
    
    def validate_credentials(self, store_id: int, password: str) -> bool:
        """验证凭证"""
        stored = self.get_credentials(store_id)
        if not stored:
            return False
        
        # 使用 constant-time 比较
        import hmac
        return hmac.compare_digest(stored["password"], password)
    
    def mask_credentials(self, store: Store) -> dict:
        """脱敏凭证"""
        return {
            "id": store.id,
            "name": store.name,
            "platform": store.platform,
            "username": store.username,
            "password": "********" if store.password else "",
            "has_credentials": bool(store.password)
        }
    
    def get_all_masked_credentials(self) -> list:
        """获取所有脱敏凭证"""
        stores = self.db.query(Store).all()
        return [self.mask_credentials(store) for store in stores]


credential_manager_factory = lambda db: CredentialManager(db)
