"""
POLYSEER configuration & tunables (simplified)

- Settings: essential environment-driven configuration
- Constants: core magic numbers for Bayesian calc, memory, agents
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional, Dict, Any


# =========================
# Environment-driven settings
# =========================
class Settings(BaseSettings):
    """Essential application settings loaded from environment variables."""

    # --- API KEYS ---
    OPENAI_API_KEY: str = ""
    VALYU_API_KEY: str = ""
    POLYROUTER_API_KEY: str = ""

    # --- DATABASE ---
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    MEMORY_BACKEND: str = "postgresql"  # "postgresql" | "redis" | "memory"
    ENABLE_MEMORY_PERSISTENCE: bool = True

    # --- TRADING PROVIDERS ---
    KALSHI_API_KEY_ID: str = ""
    KALSHI_API_KEY: str = ""     # Bearer token
    KALSHI_PRIVATE_KEY_PATH: str = "./keys/kalshi_private_key.pem"

    POLYMARKET_CLOB_API_KEY: str = ""
    POLYMARKET_CLOB_SECRET: str = ""
    TRADING_WALLET_PRIVATE_KEY: str = ""  # Store securely, never commit
    TRADING_WALLET_ADDRESS: str = ""

    # --- API URLs ---
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB_URL: str = "https://clob.polymarket.com"
    KALSHI_API_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

    # --- CORE SETTINGS ---
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DRY_RUN_MODE: bool = True  # Paper trading by default

    # --- RISK MANAGEMENT ---
    INITIAL_BANKROLL: float = 10000.0
    MAX_KELLY_FRACTION: float = 0.05
    MIN_EDGE_THRESHOLD: float = 0.02
    MAX_POSITION_SIZE_PCT: float = 0.05  # 5% max per position
    MAX_TOTAL_EXPOSURE_PCT: float = 0.25  # 25% max total exposure

    # --- EXECUTION ---
    AUTO_EXECUTE_THRESHOLD: float = 100.0  # Auto-execute if bet < $100
    MIN_CONFIDENCE_FOR_AUTO: int = 75  # Min confidence for auto-execution

    # --- CACHE SETTINGS ---
    CACHE_TTL_SHORT: int = 60    # 1 minute for markets/prices
    CACHE_TTL_LONG: int = 300    # 5 minutes for events
    CACHE_MAX_SIZE: int = 5000   # Max cache entries

    # --- RATE LIMITING ---
    POLYMARKET_RATE_LIMIT: float = 10.0  # requests per second
    KALSHI_RATE_LIMIT: float = 5.0  # requests per second
    VALYU_RATE_LIMIT: float = 2.0  # requests per second

    # --- OPTIONAL FEATURES ---
    ENABLE_COPY_TRADING: bool = False
    ENABLE_EDGE_DETECTION: bool = True
    ENABLE_INSIDER_TRACKING: bool = False

    # --- WEBHOOKS ---
    ALCHEMY_API_KEY: str = ""
    ALCHEMY_WEBHOOK_URL: str = ""
    WEBHOOK_URL: str = ""  # Default webhook URL

    # --- ON-CHAIN ANALYTICS ---
    BITQUERY_API_KEY: str = ""
    BITQUERY_API_URL: str = "https://graphql.bitquery.io"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # allow unknown keys in .env for forward compatibility

    def validate_memory_config(self) -> Dict[str, Any]:
        """
        Validate memory configuration and return validation results.
        
        Returns:
            Dictionary with validation results including:
            - valid: Whether configuration is valid
            - errors: List of error messages
            - warnings: List of warning messages
            - recommendations: List of recommendations
        """
        errors = []
        warnings = []
        recommendations = []
        
        # Check memory persistence setting
        if not self.ENABLE_MEMORY_PERSISTENCE:
            warnings.append("Memory persistence is disabled - agents will not use memory")
            recommendations.append("Set ENABLE_MEMORY_PERSISTENCE=true to enable memory features")
            return {
                "valid": True,  # Still valid, just disabled
                "errors": errors,
                "warnings": warnings,
                "recommendations": recommendations,
            }
        
        # Check backend configuration
        if self.MEMORY_BACKEND not in ["postgresql", "redis", "memory"]:
            errors.append(f"Invalid MEMORY_BACKEND: {self.MEMORY_BACKEND}. Must be 'postgresql', 'redis', or 'memory'")
        
        # Check PostgreSQL/Supabase credentials if using PostgreSQL backend
        if self.MEMORY_BACKEND == "postgresql":
            if not self.SUPABASE_URL and not getattr(self, "POSTGRES_URL", ""):
                errors.append("PostgreSQL backend requires SUPABASE_URL or POSTGRES_URL")
            
            if not self.SUPABASE_KEY and not self.SUPABASE_SERVICE_KEY and not getattr(self, "POSTGRES_URL", ""):
                errors.append("PostgreSQL backend requires SUPABASE_KEY or SUPABASE_SERVICE_KEY or POSTGRES_URL")
            
            # Check URL format if Supabase URL provided
            if self.SUPABASE_URL:
                import re
                if not re.match(r"https://[a-zA-Z0-9-]+\.supabase\.co", self.SUPABASE_URL):
                    warnings.append(f"SUPABASE_URL format may be invalid: {self.SUPABASE_URL}")
        
        # Check Redis configuration if using Redis backend
        if self.MEMORY_BACKEND == "redis":
            if not getattr(self, "REDIS_URL", ""):
                warnings.append("Redis backend requires REDIS_URL environment variable")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }


# =========================
# Global settings instance
# =========================

# Global settings instance
settings = Settings()
