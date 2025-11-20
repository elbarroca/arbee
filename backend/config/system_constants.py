"""
POLYSEER System Constants - Configurable Parameters
All magic numbers and hardcoded values centralized here for easy tuning
"""

# ============================================================================
# PRIOR ESTIMATION
# ============================================================================

# Prior probability bounds
PRIOR_MIN_BOUND = 0.01  # Minimum allowed prior (1%)
PRIOR_MAX_BOUND = 0.99  # Maximum allowed prior (99%)

# Ideal prior range (represents healthy uncertainty)
PRIOR_BEST_RANGE_MIN = 0.20  # Lower bound of ideal range (20%)
PRIOR_BEST_RANGE_MAX = 0.80  # Upper bound of ideal range (80%)

# Prior estimation clamping (for data-driven estimation)
PRIOR_ESTIMATION_MIN = 0.10  # Don't estimate below 10%
PRIOR_ESTIMATION_MAX = 0.90  # Don't estimate above 90%

# Contextual adjustment limits for prior estimation
MAX_POSITIVE_ADJUSTMENT = 0.10  # Max upward adjustment from base rate
MAX_NEGATIVE_ADJUSTMENT = -0.10  # Max downward adjustment from base rate


# ============================================================================
# BAYESIAN CALCULATION
# ============================================================================

# Probability clamping (prevents division by zero)
PROB_CLAMP_MIN = 0.0001  # Minimum probability for log-odds calculation
PROB_CLAMP_MAX = 0.9999  # Maximum probability for log-odds calculation

# Posterior probability clamping (final output bounds)
POSTERIOR_CLAMP_MIN = 0.01  # Minimum posterior (1%)
POSTERIOR_CLAMP_MAX = 0.99  # Maximum posterior (99%)

# Calculation validation
CALCULATION_TOLERANCE = 0.01  # Tolerance for validation checks (1%)


# ============================================================================
# EVIDENCE WEIGHTING
# ============================================================================

# Recency score adjustment (Phase 1 fix - prevents over-discounting old evidence)
MIN_RECENCY_WEIGHT = 0.60  # Minimum recency multiplier (even old evidence gets 60%)

# Default quality scores (when not specified)
DEFAULT_VERIFIABILITY_SCORE = 0.50  # Default if unknown
DEFAULT_INDEPENDENCE_SCORE = 0.80  # Default if unknown
DEFAULT_RECENCY_SCORE = 0.70  # Default if unknown

# Correlation shrinkage
CORRELATION_SHRINKAGE_METHOD = "sqrt"  # Method: 1/sqrt(cluster_size)


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

# LLR sensitivity scenarios
SENSITIVITY_LLR_MULTIPLIERS = [
    ("baseline", 1.0),
    ("+25% LLR", 1.25),
    ("-25% LLR", 0.75)
]

# Evidence removal scenario
SENSITIVITY_WEAKEST_REMOVAL_PCT = 0.20  # Remove weakest 20% of evidence

# Robustness thresholds (for interpreting sensitivity results)
SENSITIVITY_VERY_ROBUST_THRESHOLD = 0.05  # <5% range = very robust
SENSITIVITY_ROBUST_THRESHOLD = 0.10  # <10% range = robust
SENSITIVITY_MODERATE_THRESHOLD = 0.20  # <20% range = moderately sensitive
# >20% range = highly sensitive


# ============================================================================
# RESEARCH CONSTRAINTS
# ============================================================================

# Evidence quantity thresholds
MIN_EVIDENCE_ITEMS = 5  # Minimum evidence before completion
MIN_EVIDENCE_FOR_HIGH_CONFIDENCE = 15  # Threshold for "high confidence"
MIN_EVIDENCE_FOR_MODERATE_CONFIDENCE = 10  # Threshold for "moderate confidence"

# Search behavior
MAX_SEARCH_ATTEMPTS = 10  # Maximum web searches per researcher
MIN_SEARCH_RESULTS_PER_QUERY = 3  # Minimum results to consider query successful

# Plan validation
MIN_SUBCLAIMS = 4  # Minimum subclaims in plan
MIN_SEARCH_SEEDS_PER_DIRECTION = 3  # Minimum search seeds per direction (pro/con/general)

# Subclaim balance tolerance
MAX_SUBCLAIM_IMBALANCE = 2  # Max difference between PRO and CON counts


# ============================================================================
# MULTI-PERSPECTIVE ANALYSIS (Phase 4)
# ============================================================================

# Assertive/Skeptical view bounds (for multi-perspective analysis)
ASSERTIVE_BOUND_OFFSET = 0.10  # Assertive view = p_bayesian + 10%
SKEPTICAL_BOUND_OFFSET = 0.10  # Skeptical view = p_bayesian - 10%


# ============================================================================
# LLR (LOG-LIKELIHOOD RATIO) CALIBRATION RANGES
# ============================================================================

# These ranges guide LLR estimation by evidence strength
# Format: (min_llr, max_llr) for each grade

LLR_RANGES = {
    "A": {"min": 1.0, "max": 3.0, "description": "Definitive evidence (polls, official data)"},
    "B": {"min": 0.3, "max": 1.0, "description": "Strong evidence (expert analysis, quality reporting)"},
    "C": {"min": 0.1, "max": 0.5, "description": "Moderate evidence (news articles, credible sources)"},
    "D": {"min": 0.01, "max": 0.2, "description": "Weak evidence (opinions, speculative)"}
}

# Extreme LLR warning threshold
EXTREME_LLR_THRESHOLD = 5.0  # Warn if |LLR| exceeds this value


# ============================================================================
# INTERPRETATION THRESHOLDS
# ============================================================================

# Change magnitude interpretation
SMALL_CHANGE_THRESHOLD = 0.05  # <5% change = balanced evidence
MODERATE_CHANGE_THRESHOLD = 0.10  # <10% change = moderate shift
# ≥10% change = strong shift

# Confidence assessment
LOW_CONFIDENCE_EVIDENCE_COUNT = 5  # <5 items = low confidence
MODERATE_CONFIDENCE_EVIDENCE_COUNT = 10  # ≥10 items = moderate
HIGH_CONFIDENCE_EVIDENCE_COUNT = 15  # ≥15 items = high


# ============================================================================
# QUALITY SCORE WEIGHTS
# ============================================================================

# Plan quality scoring weights
PLAN_QUALITY_PRIOR_WEIGHT = 0.2  # Weight for having reasonable prior
PLAN_QUALITY_JUSTIFICATION_WEIGHT = 0.2  # Weight for having justification
PLAN_QUALITY_SUBCLAIMS_WEIGHT = 0.2  # Weight for having sufficient subclaims
PLAN_QUALITY_BALANCE_WEIGHT = 0.2  # Weight for balanced subclaims
PLAN_QUALITY_SEEDS_WEIGHT = 0.2  # Weight for sufficient search seeds


# ============================================================================
# PRIOR JUSTIFICATION REQUIREMENTS
# ============================================================================

MIN_PRIOR_JUSTIFICATION_LENGTH = 20  # Minimum characters in justification


# ============================================================================
# MEMORY SYSTEM CONSTANTS
# ============================================================================

# Store Configuration
MEMORY_STORE_TYPE_DEFAULT = "postgresql"  # "redis", "postgresql", or "memory"
MEMORY_CHECKPOINTER_TYPE_DEFAULT = "memory"  # "memory", "sqlite", or "postgresql"

# Memory Limits
MAX_WORKING_MEMORY_MESSAGES = 50
MAX_EPISODE_MEMORY_ITEMS = 100
EPISODE_RETENTION_DAYS = 90

# Vector Search Configuration
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"
SIMILARITY_THRESHOLD_DEFAULT = 0.7

# Memory Search Tool Limits
SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT = 5
SEARCH_SIMILAR_MARKETS_LIMIT_MAX = 20
SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT = 10
GET_BASE_RATES_LIMIT_DEFAULT = 5

# Weaviate Configuration
WEAVIATE_TIMEOUT_SECONDS = 5.0
WEAVIATE_HYBRID_SEARCH_ALPHA = 0.35
WEAVIATE_MARKETS_CLASS_DEFAULT = "MarketAnalysisMemory"

# Memory Namespaces (tuples for LangGraph Store API compatibility)
NAMESPACE_KNOWLEDGE_BASE = ("knowledge_base",)
NAMESPACE_EPISODE_MEMORY = ("episode_memory",)
NAMESPACE_STRATEGIES = ("strategies",)


# ============================================================================
# AUTONOMOUS AGENT CONSTANTS
# ============================================================================

# Iteration Control
AGENT_MAX_ITERATIONS_DEFAULT = 20
AGENT_ITERATION_EXTENSION_DEFAULT = 5
AGENT_MAX_ITERATION_CAP_DEFAULT = 50
AGENT_ITERATION_WARNING_THRESHOLD = 0.8  # Warn at 80% of max iterations

# Timeout Configuration
AGENT_LLM_TIMEOUT_SECONDS = 60.0  # Per LLM API call
AGENT_TIMEOUT_SECONDS = 600.0  # Total agent execution (10 minutes)

# Recursion Limit
AGENT_RECURSION_LIMIT_MULTIPLIER = 5  # max_iterations * 5
AGENT_RECURSION_LIMIT_MIN = 60

# Memory Context Configuration
MEMORY_CONTEXT_QUERY_HISTORY_SIZE = 5  # Show last N queries
MEMORY_CONTEXT_BLOCKED_URL_DISPLAY_LIMIT = 3  # Show max N blocked URLs
MEMORY_CONTEXT_URL_MAX_LENGTH = 80  # Truncate URLs longer than this

# Loop Detection Thresholds
LOOP_DETECTION_MIN_ITERATIONS = 5  # Start checking after this many iterations
LOOP_DETECTION_TOOL_WINDOW = 10  # Look at last N messages for tool calls
LOOP_DETECTION_SAME_TOOL_THRESHOLD = 5  # Force stop if same tool called N times
LOOP_DETECTION_TOOL_DIVERSITY_THRESHOLD = 6  # Check diversity in last N calls
LOOP_DETECTION_MAX_UNIQUE_TOOLS = 2  # Max unique tools before flagging loop
LOOP_DETECTION_QUERY_THRESHOLD = 5  # Min queries to check for loops
LOOP_DETECTION_QUERY_DIVERSITY_THRESHOLD = 2  # Max unique queries before flagging
LOOP_DETECTION_VALIDATION_CALL_THRESHOLD = 4  # Max validation calls in window

# Progress Stall Detection
PROGRESS_STALL_CHECK_START_ITERATION = 5  # Start checking after this iteration
PROGRESS_STALL_NO_CHANGE_THRESHOLD = 3  # Force stop after N iterations with no progress

# Message History
MESSAGE_HISTORY_RECENT_TOOL_LOOKUP_WINDOW = 5  # Look back N messages for tool calls

# Logging Limits
LOG_MESSAGE_PREVIEW_LENGTH = 240
LOG_TOOL_ARGS_PREVIEW_LENGTH = 140
LOG_MEMORY_CONTEXT_PREVIEW_LENGTH = 80


# ============================================================================
# AUTO-MEMORY QUERY CONSTANTS
# ============================================================================

# Enable auto-querying memory at agent start
AUTO_QUERY_MEMORY_ENABLED = True  # Query memory at agent start by default
AUTO_QUERY_SIMILAR_MARKETS_LIMIT = 3  # Fetch top N similar markets
AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT = 5  # Fetch top N historical evidence items
AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT = 3  # Fetch top N successful strategies

# Memory query timeout
MEMORY_QUERY_TIMEOUT_SECONDS = 5.0  # Fail fast if memory query hangs


# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Memory Features
ENABLE_MEMORY_TRACKING_DEFAULT = True
ENABLE_QUERY_DEDUPLICATION_DEFAULT = True
ENABLE_URL_BLOCKING_DEFAULT = True
ENABLE_CIRCUIT_BREAKERS_DEFAULT = True
ENABLE_AUTO_MEMORY_QUERY_DEFAULT = True


# ============================================================================
# ENVIRONMENT VARIABLE KEYS
# ============================================================================

# Memory Backend
ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_KEY = "SUPABASE_KEY"
ENV_POSTGRES_URL = "POSTGRES_URL"
ENV_REDIS_URL = "REDIS_URL"

# Weaviate
ENV_WEAVIATE_URL = "WEAVIATE_URL"
ENV_WEAVIATE_ENDPOINT = "WEAVIATE_ENDPOINT"
ENV_WEAVIATE_API_KEY = "WEAVIATE_API_KEY"
ENV_WCS_API_KEY = "WCS_API_KEY"
ENV_WEAVIATE_MARKETS_CLASS = "WEAVIATE_MARKETS_CLASS"

# OpenAI
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
