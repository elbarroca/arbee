"""
POLYSEER configuration & tunables (merged)

- Settings: environment-driven configuration (Pydantic)
- Constants: centralized magic numbers for priors, Bayesian calc, memory, agents, etc.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import List, Optional, Dict, Any


# =========================
# Environment-driven settings
# =========================
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API KEYS ---
    OPENAI_API_KEY: str = ""
    VALYU_API_KEY: str = ""

    # --- DATABASE & MEMORY BACKEND ---
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_ANON_KEY: str = ""   # alias for SUPABASE_KEY if you prefer
    SUPABASE_SERVICE_KEY: str = ""

    MEMORY_BACKEND: str = "postgresql"  # "postgresql" | "redis" | "memory"
    ENABLE_MEMORY_PERSISTENCE: bool = True

    # --- TRADING / PROVIDER KEYS ---
    KALSHI_API_KEY_ID: str = ""
    KALSHI_API_KEY: str = ""     # Bearer token
    KALSHI_PRIVATE_KEY_PATH: str = "./keys/kalshi_private_key.pem"

    # --- API URLs ---
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    POLYMARKET_CLOB_URL: str = "https://clob.polymarket.com"
    # Our kalshi client defaults internally to https://trading-api.kalshi.com.
    KALSHI_API_URL: str = "https://api.elections.kalshi.com/trade-api/v2"

    # --- RISK MGMT ---
    DEFAULT_BANKROLL: float = 10000.0
    MAX_KELLY_FRACTION: float = 0.05
    MIN_EDGE_THRESHOLD: float = 0.02
    RISK_FREE_RATE: float = 0.02
    CONFIDENCE_LEVEL: float = 0.95

    # --- ENV / LOGGING ---
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- API SERVER (optional, if you run FastAPI) ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- LANGSMITH (optional) ---
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = ""

    # --- EDGE DETECTION & INSIDER TRACKING ---
    ENABLE_INSIDER_TRACKING: bool = False
    ENABLE_EDGE_DETECTION: bool = True
    INSIDER_WALLET_ADDRESSES: List[str] = Field(default_factory=list)
    POLYSIGHTS_API_KEY: str = ""
    STAND_TRADE_API_KEY: str = ""
    
    # --- COPY TRADING ---
    ENABLE_COPY_TRADING: bool = False
    COPY_TRADING_MAX_SLIPPAGE_BPS: int = 50
    COPY_TRADING_MAX_SIZE_PER_WALLET: float = 1000.0
    COPY_TRADING_COOLDOWN_SECONDS: int = 60
    COPY_TRADING_MIN_EV_THRESHOLD: float = 0.02
    COPY_TRADING_ADVERSE_FILL_LIMIT: int = 3
    COPY_TRADING_ADVERSE_FILL_WINDOW_MINUTES: int = 10
    COPY_TRADING_MIN_TRADE_SIZE_USD: float = 10.0
    COPY_TRADING_MIN_LIQUIDITY_USD: float = 1000.0
    
    # --- WEBHOOK PROVIDERS ---
    ALCHEMY_API_KEY: str = ""
    ALCHEMY_WEBHOOK_URL: str = ""
    QUICKNODE_API_KEY: str = ""
    QUICKNODE_WEBHOOK_URL: str = ""
    MORALIS_API_KEY: str = ""
    MORALIS_WEBHOOK_URL: str = ""
    WEBHOOK_URL: str = ""  # Default webhook URL if provider-specific not set

    # --- ON-CHAIN ANALYTICS ---
    BITQUERY_API_KEY: str = ""
    BITQUERY_API_URL: str = "https://graphql.bitquery.io"
    DUNE_API_KEY: str = ""

    # --- TRADING WALLET ---
    TRADING_WALLET_PRIVATE_KEY: str = ""  # Store securely, never commit
    TRADING_WALLET_ADDRESS: str = ""
    POLYMARKET_CLOB_API_KEY: str = ""
    POLYMARKET_CLOB_SECRET: str = ""
    DRY_RUN_MODE: bool = True  # Paper trading by default
    
    # --- COPY TRADER CRITERIA ---
    COPY_TRADER_MIN_PNL_30D: float = 0.0
    COPY_TRADER_MIN_SHARPE: float = 0.7
    COPY_TRADER_MIN_TRADES: int = 200
    COPY_TRADER_MAX_WALLET_AGE_DAYS: Optional[int] = None
    COPY_TRADER_MIN_WIN_RATE: float = 0.5
    COPY_TRADING_MAX_TRADERS_TO_CHECK: int = 5  # Max traders to check per market
    ARBITRAGE_MAX_OPPORTUNITIES_TO_LOG: int = 5  # Max arbitrage opportunities to log

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


# Global settings instance
settings = Settings()


# =========================
# System constants / tunables
# =========================

# --- PRIOR ESTIMATION ---
PRIOR_MIN_BOUND = 0.01
PRIOR_MAX_BOUND = 0.99
PRIOR_BEST_RANGE_MIN = 0.20
PRIOR_BEST_RANGE_MAX = 0.80
PRIOR_ESTIMATION_MIN = 0.10
PRIOR_ESTIMATION_MAX = 0.90
MAX_POSITIVE_ADJUSTMENT = 0.10
MAX_NEGATIVE_ADJUSTMENT = -0.10

# --- BAYESIAN CALCULATION ---
PROB_CLAMP_MIN = 0.0001
PROB_CLAMP_MAX = 0.9999
POSTERIOR_CLAMP_MIN = 0.01
POSTERIOR_CLAMP_MAX = 0.99
CALCULATION_TOLERANCE = 0.01

# --- EVIDENCE WEIGHTING ---
MIN_RECENCY_WEIGHT = 0.60
DEFAULT_VERIFIABILITY_SCORE = 0.50
DEFAULT_INDEPENDENCE_SCORE = 0.80
DEFAULT_RECENCY_SCORE = 0.70
CORRELATION_SHRINKAGE_METHOD = "sqrt"  # 1/sqrt(cluster_size)

# --- SENSITIVITY ANALYSIS ---
SENSITIVITY_LLR_MULTIPLIERS = [
    ("baseline", 1.0),
    ("+25% LLR", 1.25),
    ("-25% LLR", 0.75),
]
SENSITIVITY_WEAKEST_REMOVAL_PCT = 0.20
SENSITIVITY_VERY_ROBUST_THRESHOLD = 0.05
SENSITIVITY_ROBUST_THRESHOLD = 0.10
SENSITIVITY_MODERATE_THRESHOLD = 0.20
# >20% = highly sensitive

# --- RESEARCH CONSTRAINTS ---
MIN_EVIDENCE_ITEMS = 5
MIN_EVIDENCE_FOR_HIGH_CONFIDENCE = 15
MIN_EVIDENCE_FOR_MODERATE_CONFIDENCE = 10
MAX_SEARCH_ATTEMPTS = 10
MIN_SEARCH_RESULTS_PER_QUERY = 3
MIN_SUBCLAIMS = 4
MIN_SEARCH_SEEDS_PER_DIRECTION = 3
MAX_SUBCLAIM_IMBALANCE = 2

# --- MULTI-PERSPECTIVE ANALYSIS ---
ASSERTIVE_BOUND_OFFSET = 0.10
SKEPTICAL_BOUND_OFFSET = 0.10

# --- LLR CALIBRATION RANGES ---
LLR_RANGES = {
    "A": {"min": 1.0, "max": 3.0, "description": "Definitive evidence (polls, official data)"},
    "B": {"min": 0.3, "max": 1.0, "description": "Strong evidence (expert analysis, quality reporting)"},
    "C": {"min": 0.1, "max": 0.5, "description": "Moderate evidence (news, credible sources)"},
    "D": {"min": 0.01, "max": 0.2, "description": "Weak evidence (opinions, speculative)"},
}
EXTREME_LLR_THRESHOLD = 5.0

# --- INTERPRETATION THRESHOLDS ---
SMALL_CHANGE_THRESHOLD = 0.05
MODERATE_CHANGE_THRESHOLD = 0.10
LOW_CONFIDENCE_EVIDENCE_COUNT = 5
MODERATE_CONFIDENCE_EVIDENCE_COUNT = 10
HIGH_CONFIDENCE_EVIDENCE_COUNT = 15

# --- PLAN QUALITY WEIGHTS ---
PLAN_QUALITY_PRIOR_WEIGHT = 0.2
PLAN_QUALITY_JUSTIFICATION_WEIGHT = 0.2
PLAN_QUALITY_SUBCLAIMS_WEIGHT = 0.2
PLAN_QUALITY_BALANCE_WEIGHT = 0.2
PLAN_QUALITY_SEEDS_WEIGHT = 0.2

# --- PRIOR JUSTIFICATION ---
MIN_PRIOR_JUSTIFICATION_LENGTH = 20

# --- MEMORY SYSTEM CONSTANTS ---
MEMORY_STORE_TYPE_DEFAULT = "postgresql"   # "redis" | "postgresql" | "memory"
MEMORY_CHECKPOINTER_TYPE_DEFAULT = "memory"  # "memory" | "sqlite" | "postgresql"
MAX_WORKING_MEMORY_MESSAGES = 50
MAX_EPISODE_MEMORY_ITEMS = 100
EPISODE_RETENTION_DAYS = 90
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"
SIMILARITY_THRESHOLD_DEFAULT = 0.7

SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT = 5
SEARCH_SIMILAR_MARKETS_LIMIT_MAX = 20
SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT = 10
GET_BASE_RATES_LIMIT_DEFAULT = 5

WEAVIATE_TIMEOUT_SECONDS = 5.0
WEAVIATE_HYBRID_SEARCH_ALPHA = 0.35
WEAVIATE_MARKETS_CLASS_DEFAULT = "MarketAnalysisMemory"

# Namespaces for LangGraph Store
NAMESPACE_KNOWLEDGE_BASE = ("knowledge_base",)
NAMESPACE_EPISODE_MEMORY = ("episode_memory",)
NAMESPACE_STRATEGIES = ("strategies",)

# --- AUTONOMOUS AGENT CONSTANTS ---
AGENT_MAX_ITERATIONS_DEFAULT = 20
AGENT_ITERATION_EXTENSION_DEFAULT = 5
AGENT_MAX_ITERATION_CAP_DEFAULT = 50
AGENT_ITERATION_WARNING_THRESHOLD = 0.8
AGENT_LLM_TIMEOUT_SECONDS = 60.0
AGENT_TIMEOUT_SECONDS = 600.0
AGENT_RECURSION_LIMIT_MULTIPLIER = 5
AGENT_RECURSION_LIMIT_MIN = 60
MEMORY_CONTEXT_QUERY_HISTORY_SIZE = 5
MEMORY_CONTEXT_BLOCKED_URL_DISPLAY_LIMIT = 3
MEMORY_CONTEXT_URL_MAX_LENGTH = 80
LOOP_DETECTION_MIN_ITERATIONS = 5
LOOP_DETECTION_TOOL_WINDOW = 10
LOOP_DETECTION_SAME_TOOL_THRESHOLD = 5
LOOP_DETECTION_TOOL_DIVERSITY_THRESHOLD = 6
LOOP_DETECTION_MAX_UNIQUE_TOOLS = 2
LOOP_DETECTION_QUERY_THRESHOLD = 5
LOOP_DETECTION_QUERY_DIVERSITY_THRESHOLD = 2
LOOP_DETECTION_VALIDATION_CALL_THRESHOLD = 4
PROGRESS_STALL_CHECK_START_ITERATION = 5
PROGRESS_STALL_NO_CHANGE_THRESHOLD = 3
MESSAGE_HISTORY_RECENT_TOOL_LOOKUP_WINDOW = 5
LOG_MESSAGE_PREVIEW_LENGTH = 240
LOG_TOOL_ARGS_PREVIEW_LENGTH = 140
LOG_MEMORY_CONTEXT_PREVIEW_LENGTH = 80

# --- AUTO-MEMORY FEATURE FLAGS ---
AUTO_QUERY_MEMORY_ENABLED = True
AUTO_QUERY_SIMILAR_MARKETS_LIMIT = 3
AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT = 5
AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT = 3
MEMORY_QUERY_TIMEOUT_SECONDS = 5.0

ENABLE_MEMORY_TRACKING_DEFAULT = True
ENABLE_QUERY_DEDUPLICATION_DEFAULT = True
ENABLE_URL_BLOCKING_DEFAULT = True
ENABLE_CIRCUIT_BREAKERS_DEFAULT = True
ENABLE_AUTO_MEMORY_QUERY_DEFAULT = True

# --- TOKEN MAPPING CACHE ---
TOKEN_MAPPING_CACHE_TTL_SECONDS = 300  # 5 minutes
MARKET_VALIDATION_CACHE_TTL_SECONDS = 300  # 5 minutes
ORDERBOOK_SNAPSHOT_CACHE_TTL_SECONDS = 60  # 1 minute

# --- TRADER DISCOVERY SCORING ---
TRADER_SCORE_EARLY_BETTING_WEIGHT = 0.30  # Trades within 24h of market creation
TRADER_SCORE_VOLUME_CONSISTENCY_WEIGHT = 0.20  # Low coefficient of variation
TRADER_SCORE_WIN_RATE_WEIGHT = 0.20  # Resolved market success rate
TRADER_SCORE_EDGE_DETECTION_WEIGHT = 0.20  # Price movement after bet
TRADER_SCORE_ACTIVITY_LEVEL_WEIGHT = 0.10  # Trade frequency

TRADER_MIN_TRADES_30D = 50  # Minimum trades to be considered
TRADER_MIN_RESOLVED_MARKETS = 10  # For win rate calculation
TRADER_WIN_RATE_THRESHOLD = 0.55  # 55% win rate for full points
TRADER_IDEAL_WEEKLY_TRADES_MIN = 5
TRADER_IDEAL_WEEKLY_TRADES_MAX = 20
TRADER_EARLY_BET_WINDOW_HOURS = 24  # Bet within X hours of market creation
TRADER_AUTO_ADD_SCORE_THRESHOLD = 70  # Auto-add if score > 70
TRADER_AUTO_PAUSE_SHARPE_THRESHOLD = 0.3  # Pause if 7-day Sharpe < 0.3
TRADER_SCORE_REFRESH_INTERVAL_HOURS = 24  # Daily refresh

# --- PAPER TRADING ---
PAPER_TRADING_LOG_TABLE = "paper_trading_logs"
PAPER_TRADING_MIN_DAYS = 7  # Minimum days of paper trading before live
PAPER_TRADING_MIN_POSITIVE_ROI = 0.02  # 2% ROI required to go live

# --- ENV VAR KEYS (reference) ---
ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_KEY = "SUPABASE_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

__all__ = [
    "settings",
    # Constants (export the ones you commonly use outside)
    "PRIOR_MIN_BOUND", "PRIOR_MAX_BOUND", "PRIOR_BEST_RANGE_MIN", "PRIOR_BEST_RANGE_MAX",
    "PRIOR_ESTIMATION_MIN", "PRIOR_ESTIMATION_MAX",
    "PROB_CLAMP_MIN", "PROB_CLAMP_MAX", "POSTERIOR_CLAMP_MIN", "POSTERIOR_CLAMP_MAX",
    "SENSITIVITY_LLR_MULTIPLIERS", "LLR_RANGES",
    "MIN_EVIDENCE_ITEMS", "MAX_SEARCH_ATTEMPTS",
    "MIN_SUBCLAIMS", "MIN_SEARCH_SEEDS_PER_DIRECTION",
    "NAMESPACE_KNOWLEDGE_BASE", "NAMESPACE_EPISODE_MEMORY", "NAMESPACE_STRATEGIES",
    "AGENT_MAX_ITERATIONS_DEFAULT", "AGENT_TIMEOUT_SECONDS",
]
