"""
Copy Trade Signal Tool
Detects copy trading opportunities from wallet activity.
"""
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import logging
from datetime import datetime

from clients.web3.wallet_tracker import WalletTrackerClient
from agents.copy_trading_agent import CopyTradingAgent
from betting.copy_trading import TradeSignalProcessor
from utils.rich_logging import (
    log_tool_start,
    log_tool_success,
    log_tool_error,
)

logger = logging.getLogger(__name__)


@tool
async def copy_trade_signal_tool(
    market_slug: str,
    wallet_address: Optional[str] = None,
    p_bayesian: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Detect copy trading opportunities from wallet activity.
    
    Checks if any copy-list wallets have trades on the specified market
    and validates them for potential copying.
    
    Args:
        market_slug: Market to check for copy opportunities
        wallet_address: Optional specific wallet to check
        p_bayesian: Optional Bayesian probability for EV calculation
        
    Returns:
        Dict with copy_opportunities (list), signals_count, and trader_count
    """
    try:
        log_tool_start("copy_trade_signal_tool", {"market_slug": market_slug, "wallet_address": wallet_address})
        logger.info(f"📋 Detecting copy trade signals for {market_slug}")
        
        # Initialize components
        copy_agent = CopyTradingAgent()
        wallet_tracker = WalletTrackerClient()
        trade_processor = TradeSignalProcessor(
            copy_agent=copy_agent,
            wallet_tracker=wallet_tracker
        )
        
        # Get active traders
        active_traders = copy_agent.get_active_traders()
        
        if not active_traders:
            return {
                "copy_opportunities": [],
                "signals_count": 0,
                "trader_count": 0,
                "market_slug": market_slug,
            }
        
        # Check traders for activity on this market
        copy_opportunities = []
        
        traders_to_check = [t for t in active_traders if not wallet_address or t.wallet_address.lower() == wallet_address.lower()]
        
        for trader in traders_to_check[:10]:  # Limit to top 10
            try:
                trades = await wallet_tracker.get_wallet_trades(
                    trader.wallet_address,
                    lookback_hours=24,
                    market_slug=market_slug
                )
                
                if trades:
                    # Process most recent trade
                    latest_trade = trades[0]
                    
                    webhook_event = {
                        "type": "trade_signal",
                        "wallet_address": trader.wallet_address,
                        "transaction_hash": latest_trade.get("transaction_hash", ""),
                        "market_slug": market_slug,
                        "raw_event": latest_trade
                    }
                    
                    signal = await trade_processor.process_webhook_event(webhook_event)
                    
                    if signal:
                        # Calculate EV if p_bayesian provided
                        ev_metrics = None
                        if p_bayesian is not None:
                            ev_metrics = trade_processor.calculate_ev_and_kelly(signal, p_bayesian)
                        
                        copy_opportunities.append({
                            "wallet_address": trader.wallet_address,
                            "signal": signal.model_dump() if hasattr(signal, 'model_dump') else signal,
                            "ev_metrics": ev_metrics,
                            "trader_pnl_30d": trader.pnl_30d,
                            "trader_sharpe": trader.sharpe_equivalent,
                        })
            except Exception as e:
                logger.warning(f"Error checking trader {trader.wallet_address[:8]}...: {e}")
                continue
        
        result = {
            "copy_opportunities": copy_opportunities,
            "signals_count": len(copy_opportunities),
            "trader_count": len(active_traders),
            "market_slug": market_slug,
        }
        
        log_tool_success("copy_trade_signal_tool", {"signals_count": len(copy_opportunities), "trader_count": len(active_traders)})
        return result
        
    except Exception as e:
        log_tool_error("copy_trade_signal_tool", e, f"Market: {market_slug}")
        logger.error(f"❌ Copy trade signal detection failed: {e}")
        return {
            "copy_opportunities": [],
            "signals_count": 0,
            "trader_count": 0,
            "error": str(e),
        }

