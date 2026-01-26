"""Encryption utilities for sensitive data like tokens."""

import base64
import hashlib
import os

from cryptography.fernet import Fernet


def _get_encryption_key() -> bytes:
    """Get or derive the encryption key from environment variable.

    Uses TOKEN_ENCRYPTION_KEY env var if set (must be 32 url-safe base64 bytes),
    otherwise derives a key from LANGSMITH_API_KEY using SHA256.

    Returns:
        32-byte Fernet-compatible key
    """
    # Check for explicit encryption key first
    explicit_key = os.environ.get("TOKEN_ENCRYPTION_KEY")
    if explicit_key:
        return explicit_key.encode()

    # Fall back to deriving from LANGSMITH_API_KEY
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not langsmith_key:
        # Use a default seed if nothing else available (not recommended for production)
        langsmith_key = "deepagents-default-key"

    # Derive a 32-byte key using SHA256 and encode as url-safe base64
    derived = hashlib.sha256(langsmith_key.encode()).digest()
    return base64.urlsafe_b64encode(derived)


def encrypt_token(token: str) -> str:
    """Encrypt a token for safe storage.

    Args:
        token: The plaintext token to encrypt

    Returns:
        Base64-encoded encrypted token
    """
    if not token:
        return ""

    key = _get_encryption_key()
    f = Fernet(key)
    encrypted = f.encrypt(token.encode())
    return encrypted.decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an encrypted token.

    Args:
        encrypted_token: The base64-encoded encrypted token

    Returns:
        The plaintext token, or empty string if decryption fails
    """
    if not encrypted_token:
        return ""

    try:
        key = _get_encryption_key()
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_token.encode())
        return decrypted.decode()
    except Exception as e:
        print(f"[Encryption] Failed to decrypt token: {e}")
        return ""
