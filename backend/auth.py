import datetime
import secrets
import threading
from typing import Dict, Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from backend.config import settings
from backend.schemas import KeyInfo

security = HTTPBearer(auto_error=False)


class KeyManager:
    """Manages active API keys and usage tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: Dict[str, KeyInfo] = {}
        # Pre-seed with configured keys from settings
        for key in settings.valid_api_keys:
            self._keys[key] = KeyInfo(
                key=key,
                name="Default Pre-configured Key",
                created_at=datetime.datetime.utcnow().isoformat(),
            )

    def is_valid_key(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def generate_key(self, name: str) -> KeyInfo:
        random_suffix = secrets.token_hex(16)
        new_key = f"sk-antigravity-{random_suffix}"
        info = KeyInfo(
            key=new_key,
            name=name,
            created_at=datetime.datetime.utcnow().isoformat(),
        )
        with self._lock:
            self._keys[new_key] = info
        return info

    def record_usage(self, key: str, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            if key in self._keys:
                info = self._keys[key]
                info.total_requests += 1
                info.total_prompt_tokens += prompt_tokens
                info.total_completion_tokens += completion_tokens

    def list_keys(self) -> Dict[str, KeyInfo]:
        with self._lock:
            return dict(self._keys)


key_manager = KeyManager()


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    """FastAPI dependency to validate OpenAI Bearer tokens (sk-...)."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "missing_api_key",
                }
            },
        )

    token = credentials.credentials.strip()
    if not key_manager.is_valid_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": f"Incorrect API key provided: {token[:6]}...{token[-4:] if len(token) > 10 else ''}. You can find your API key in the gateway configuration.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    return token


async def verify_admin_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    """FastAPI dependency to validate Admin management tokens."""
    if credentials is None or credentials.credentials != settings.GATEWAY_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Invalid or missing admin credentials."},
        )
    return credentials.credentials
