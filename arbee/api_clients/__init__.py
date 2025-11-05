from arbee.api_clients.polymarket import PolymarketClient
from arbee.api_clients.kalshi import KalshiClient

try:
    from arbee.api_clients.valyu import ValyuResearchClient
    __all__ = ["PolymarketClient", "KalshiClient", "ValyuResearchClient"]
except ImportError:
    # ValyuResearchClient requires optional langchain_valyu dependency
    __all__ = ["PolymarketClient", "KalshiClient"]
