from cryptography.fernet import Fernet
from .config import settings

class SecurityVault:
    def __init__(self):
        assert settings.MASTER_KEY, "❌ EXECUTION_MASTER_KEY required"
        self._cipher = Fernet(settings.MASTER_KEY.encode())

    def encrypt(self, raw: str) -> str:
        return self._cipher.encrypt(raw.encode()).decode()

    def decrypt(self, enc: str) -> str:
        return self._cipher.decrypt(enc.encode()).decode()

vault = SecurityVault()