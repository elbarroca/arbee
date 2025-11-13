"""
FastAPI Dependencies
Shared dependencies for dependency injection
"""
import logging
from typing import Annotated

from fastapi import Depends
from clients.web3.wallet_tracker import WalletTrackerClient
from agents.copy_trading_agent import CopyTradingAgent
from betting.copy_trading import TradeSignalProcessor
from database.client import SupabaseClient
from clients.trade.trade_executor import TradeExecutor
from clients.polymarket import PolymarketClient
from config.settings import settings

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
_wallet_tracker: WalletTrackerClient | None = None
_copy_agent: CopyTradingAgent | None = None
_trade_processor: TradeSignalProcessor | None = None
_db_client: SupabaseClient | None = None
_trade_executor: TradeExecutor | None = None
_polymarket_client: PolymarketClient | None = None


def initialize_dependencies():
    """Initialize all dependencies (called on startup)"""
    global _wallet_tracker, _copy_agent, _trade_processor, _db_client, _trade_executor, _polymarket_client
    
    logger.info("Initializing API dependencies...")
    
    # Initialize wallet tracker
    _wallet_tracker = WalletTrackerClient()
    
    # Initialize copy trading agent
    _copy_agent = CopyTradingAgent(
        min_pnl_30d=settings.COPY_TRADER_MIN_PNL_30D,
        min_sharpe=settings.COPY_TRADER_MIN_SHARPE,
        min_trades=settings.COPY_TRADER_MIN_TRADES,
        min_win_rate=settings.COPY_TRADER_MIN_WIN_RATE
    )
    
    # Initialize trade signal processor
    _trade_processor = TradeSignalProcessor(
        copy_agent=_copy_agent,
        wallet_tracker=_wallet_tracker
    )
    
    # Initialize database client
    _db_client = SupabaseClient()
    
    # Initialize Polymarket client
    _polymarket_client = PolymarketClient()
    
    # Initialize trade executor
    _trade_executor = TradeExecutor(
        polymarket_client=_polymarket_client
    )
    
    trader_count = len(_copy_agent.copy_list) if hasattr(_copy_agent, 'copy_list') else 0
    logger.info(f"API dependencies initialized. Active traders: {trader_count}")


def cleanup_dependencies():
    """Cleanup dependencies (called on shutdown)"""
    global _wallet_tracker, _copy_agent, _trade_processor, _db_client, _trade_executor, _polymarket_client
    logger.info("Cleaning up API dependencies...")
    # Add any cleanup logic here if needed


def get_wallet_tracker() -> WalletTrackerClient:
    """Get wallet tracker instance"""
    if _wallet_tracker is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _wallet_tracker


def get_copy_agent() -> CopyTradingAgent:
    """Get copy trading agent instance"""
    if _copy_agent is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _copy_agent


def get_trade_processor() -> TradeSignalProcessor:
    """Get trade signal processor instance"""
    if _trade_processor is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _trade_processor


def get_db_client() -> SupabaseClient:
    """Get database client instance"""
    if _db_client is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _db_client


def get_trade_executor() -> TradeExecutor:
    """Get trade executor instance"""
    if _trade_executor is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _trade_executor


def get_polymarket_client() -> PolymarketClient:
    """Get Polymarket client instance"""
    if _polymarket_client is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _polymarket_client


# Dependency injection annotations
WalletTrackerDep = Annotated[WalletTrackerClient, Depends(get_wallet_tracker)]
CopyAgentDep = Annotated[CopyTradingAgent, Depends(get_copy_agent)]
TradeProcessorDep = Annotated[TradeSignalProcessor, Depends(get_trade_processor)]
DBClientDep = Annotated[SupabaseClient, Depends(get_db_client)]
TradeExecutorDep = Annotated[TradeExecutor, Depends(get_trade_executor)]
PolymarketClientDep = Annotated[PolymarketClient, Depends(get_polymarket_client)]

