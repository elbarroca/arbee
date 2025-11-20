# ========================================================================
# CORE PREDICTION MARKET CLIENTS
# ========================================================================

# Primary platform clients
from .polymarket import PolymarketClient

# Research and analytics clients
# try:
#     from .kalshi import KalshiClient
# except ImportError:
#     KalshiClient = None

try:
    from .valyu import ValyuResearchClient
except ImportError:
    ValyuResearchClient = None

# try:
#     from .trader_analytics import TraderAnalytics
# except ImportError:
#     TraderAnalytics = None

# Wallet tracking client
from .wallet_tracker import WalletTracker

# ========================================================================
# WALLET INTELLIGENCE FUNCTIONS
# ========================================================================

# Wallet discovery and performance orchestration
from .functions.wallets import (
    PolymarketWalletCollector,
    enrich_unenriched_events,
)

# ========================================================================
# TRADER ANALYSIS FUNCTIONS
# ========================================================================

# Trader analysis and copy-trading intelligence
# try:
#     from .functions.traders import (
#         PolymarketTraderAnalyzer,
#         filter_best_traders,
#         get_best_traders_positions,
#         get_copy_trade_suggestions,
#         analyze_smart_money_consensus,
#         execute_full_trader_analysis,
#     )
#     trader_functions_available = True
# except ImportError:
#     PolymarketTraderAnalyzer = None
#     filter_best_traders = None
#     get_best_traders_positions = None
#     get_copy_trade_suggestions = None
#     analyze_smart_money_consensus = None
#     execute_full_trader_analysis = None
#     trader_functions_available = False
trader_functions_available = False

# ========================================================================
# DATA UTILITIES
# ========================================================================

from .functions.data import PolymarketDataCollector

# ========================================================================
# PUBLIC API
# ========================================================================

# Build __all__ dynamically to handle optional imports
__all__ = [
    # Core clients (always available)
    "PolymarketClient",
    "WalletTracker",

    # Data collection
    "PolymarketDataCollector",

    # Wallet intelligence
    "PolymarketWalletCollector",
    "enrich_unenriched_events",
]

# Add trader analysis functions if available
# if trader_functions_available:
#     __all__.extend([
#         "PolymarketTraderAnalyzer",
#         "filter_best_traders",
#         "get_best_traders_positions",
#         "get_copy_trade_suggestions",
#         "analyze_smart_money_consensus",
#         "execute_full_trader_analysis",
#     ])

# Add optional clients if available
# if TraderAnalytics is not None:
#     __all__.append("TraderAnalytics")

# if KalshiClient is not None:
#     __all__.append("KalshiClient")

if ValyuResearchClient is not None:
    __all__.append("ValyuResearchClient")
