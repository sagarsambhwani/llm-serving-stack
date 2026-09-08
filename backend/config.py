import os
from typing import List, Set
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Upstream vLLM configuration
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_API_KEY: str = "dev-secret"
    VLLM_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Gateway Server configuration
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 9000
    
    # Gateway API Keys (comma-separated list of valid client keys)
    # Format: sk-antigravity-...
    GATEWAY_API_KEYS: str = "sk-antigravity-dev-key,sk-test-client-key"
    GATEWAY_ADMIN_KEY: str = "admin-secret-key"

    # Optional multi-model routing table (JSON format)
    # e.g. {"model_a": "http://localhost:8000/v1", "model_b": "http://localhost:8001/v1"}
    MODEL_ROUTES_JSON: str = "{}"

    # Timeouts (seconds)
    REQUEST_TIMEOUT: float = 120.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def valid_api_keys(self) -> Set[str]:
        return {k.strip() for k in self.GATEWAY_API_KEYS.split(",") if k.strip()}

    def get_upstream_url(self, model: str) -> str:
        """Dynamically resolve upstream base URL for a requested model."""
        import json
        try:
            routes = json.loads(self.MODEL_ROUTES_JSON)
            if model in routes:
                return routes[model].rstrip("/")
        except Exception:
            pass
        return self.VLLM_BASE_URL.rstrip("/")


settings = Settings()
