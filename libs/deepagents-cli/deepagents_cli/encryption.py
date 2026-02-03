"""Encryption utilities for sensitive data like tokens."""

import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class EncryptionKeyMissingError(ValueError):
    """Raised when TOKEN_ENCRYPTION_KEY environment variable is not set."""


def _get_encryption_key() -> bytes:
    """Get or derive the encryption key from environment variable.

    Uses TOKEN_ENCRYPTION_KEY env var if set (must be 32 url-safe base64 bytes),
    otherwise derives a key from LANGSMITH_API_KEY using SHA256.

    Returns:
        32-byte Fernet-compatible key

    Raises:
        EncryptionKeyMissingError: If TOKEN_ENCRYPTION_KEY is not set
    """
    explicit_key = os.environ.get("TOKEN_ENCRYPTION_KEY")
    if not explicit_key:
        raise EncryptionKeyMissingError

    return explicit_key.encode()


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
    except InvalidToken:
        logger.warning("Failed to decrypt token: invalid token")
        return ""
    except EncryptionKeyMissingError:
        logger.warning("Failed to decrypt token: encryption key not set")
        return ""
