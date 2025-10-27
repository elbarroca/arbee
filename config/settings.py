"""
POLYSEER Configuration Settings
Centralized configuration using pydantic-settings
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API KEYS
    # =======================================
    OPENAI_API_KEY: str = ""
    VALYU_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""  # Optional, not currently used

    # DATABASE & MEMORY BACKEND
    # =======================================
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_ANON_KEY: str = ""  # Alias for SUPABASE_KEY
    SUPABASE_SERVICE_KEY: str = ""

    # Alternative: Direct PostgreSQL connection
    POSTGRES_URL: str = ""  # postgresql://user:pass@host:port/dbname

    # Alternative: Redis for memory store
    REDIS_URL: str = ""  # redis://host:port/db

    # Memory Backend Configuration
    MEMORY_BACKEND: str = "postgresql"  # "postgresql", "redis", or "memory"
    ENABLE_MEMORY_PERSISTENCE: bool = True  # False = use in-memory store only

    # Weaviate Vector Database (optional, for enhanced memory search)
    WEAVIATE_URL: str = ""
    WEAVIATE_API_KEY: str = ""
    WEAVIATE_MARKETS_CLASS: str = "MarketAnalysisMemory"

    # TRADING API KEYS
    KALSHI_API_KEY_ID: str = ""
    KALSHI_API_KEY: str = ""  # For Bearer token authentication
    KALSHI_PRIVATE_KEY_PATH: str = "./keys/kalshi_private_key.pem"

    POLYMARKET_PRIVATE_KEY: str = ""
    POLYMARKET_FUNDER_ADDRESS: str = ""
    POLYMARKET_API_KEY: str = ""

    # ========================================
    # API URLS
    # ========================================
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB_URL: str = "https://clob.polymarket.com"
    KALSHI_API_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

    # ========================================
    # RISK MANAGEMENT
    # ========================================
    DEFAULT_BANKROLL: float = 10000.0
    MAX_KELLY_FRACTION: float = 0.05  # 5% max position size
    MIN_EDGE_THRESHOLD: float = 0.02  # 2% minimum edge
    RISK_FREE_RATE: float = 0.02  # 2% annual
    CONFIDENCE_LEVEL: float = 0.95  # 95% for VaR

    # ========================================
    # ENVIRONMENT
    # ========================================
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # ========================================
    # API SERVER (if running FastAPI)
    # ========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ========================================
    # LANGSMITH (Optional observability)
    # ========================================
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = ""

    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()
