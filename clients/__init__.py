from .polymarket import PolymarketClient

# Optional Kalshi import (may not exist yet)
try:
    from .kalshi import KalshiClient
except ImportError:
    KalshiClient = None

from .valyu import ValyuResearchClient
from .web3.wallet_tracker import WalletTrackerClient
from .trade.insider_detector import InsiderDetectorClient
from .web3.alchemy import AlchemyWebhooksClient

# Optional imports for webhook providers
try:
    from .web3.quicknode import QuickNodeWebhooksClient
except ImportError:
    QuickNodeWebhooksClient = None

try:
    from .web3.moralis import MoralisStreamsClient
except ImportError:
    MoralisStreamsClient = None

from .trade.trader_analytics import TraderAnalyticsClient
from .trade.trade_executor import TradeExecutor

__all__ = [
    "PolymarketClient",
    "KalshiClient",
    "ValyuResearchClient",
    "WalletTrackerClient",
    "InsiderDetectorClient",
    "AlchemyWebhooksClient",
    "QuickNodeWebhooksClient",
    "MoralisStreamsClient",
    "TraderAnalyticsClient",
    "TradeExecutor",
]
