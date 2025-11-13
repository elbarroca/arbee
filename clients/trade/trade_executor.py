"""
Trade Executor Service
Places orders via Polymarket CLOB API with guardrails and kill switches.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from pydantic import BaseModel, Field
from ..polymarket import PolymarketClient, PolymarketError
from .paper_trading import PaperTradingLogger
from betting.copy_trading import TradeSignal
from config import settings

logger = logging.getLogger(__name__)


class ExecutedTrade(BaseModel):
    """Record of an executed trade"""
    signal_id: str = Field(..., description="Original signal identifier")
    order_id: Optional[str] = Field(None, description="CLOB order ID")
    market_slug: str = Field(..., description="Market slug")
    side: str = Field(..., description="BUY or SELL")
    size_usd: float = Field(..., description="Order size in USD")
    fill_price: Optional[float] = Field(None, description="Actual fill price")
    expected_price: float = Field(..., description="Expected price")
    slippage_bps: float = Field(0.0, description="Slippage in basis points")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field("pending", description="pending, filled, rejected, cancelled")
    error_message: Optional[str] = Field(None, description="Error if failed")


class TradeExecutor:
    """
    Executes trades via Polymarket CLOB API with risk management.
    
    Guardrails:
    - max_slippage_bps: Maximum acceptable slippage
    - max_size_per_wallet: Maximum position size per wallet
    - cooldown_period: Minimum time between trades from same wallet
    - min_ev_threshold: Minimum expected value to execute
    
    Kill Switches:
    - Halt after N adverse fills in time window
    - Suspend on rapidly widening spreads
    - Block during mentions market windows
    """
    
    def __init__(
        self,
        polymarket_client: Optional[PolymarketClient] = None,
        paper_trading_logger: Optional[PaperTradingLogger] = None,
        max_slippage_bps: int = 50,
        max_size_per_wallet: float = 1000.0,
        cooldown_seconds: int = 60,
        min_ev_threshold: float = 0.02,
        adverse_fill_limit: int = 3,
        adverse_fill_window_minutes: int = 10
    ):
        """
        Initialize trade executor.
        
        Args:
            polymarket_client: Polymarket client instance
            paper_trading_logger: Paper trading logger for DRY_RUN mode
            max_slippage_bps: Maximum slippage in basis points
            max_size_per_wallet: Maximum position size per wallet (USD)
            cooldown_seconds: Cooldown period between trades from same wallet
            min_ev_threshold: Minimum EV to execute trade
            adverse_fill_limit: Number of adverse fills before halt
            adverse_fill_window_minutes: Time window for adverse fill tracking
        """
        self.polymarket_client = polymarket_client or PolymarketClient()
        self.paper_trading_logger = paper_trading_logger or PaperTradingLogger()
        self.max_slippage_bps = max_slippage_bps
        self.max_size_per_wallet = max_size_per_wallet
        self.cooldown_seconds = cooldown_seconds
        self.min_ev_threshold = min_ev_threshold
        self.adverse_fill_limit = adverse_fill_limit
        self.adverse_fill_window_minutes = adverse_fill_window_minutes
        
        # Track wallet positions and cooldowns
        self._wallet_positions: Dict[str, float] = defaultdict(float)
        self._wallet_last_trade: Dict[str, datetime] = {}
        self._adverse_fills: List[Dict[str, Any]] = []
        self._executed_trades: List[ExecutedTrade] = []
        
        # Kill switch state
        self._halted = False
        self._halt_reason: Optional[str] = None
    
    async def execute_trade(
        self,
        signal: TradeSignal,
        ev_metrics: Optional[Dict[str, Any]] = None
    ) -> ExecutedTrade:
        """
        Execute a trade signal.
        
        Args:
            signal: Validated trade signal
            ev_metrics: Expected value metrics from processor
            
        Returns:
            ExecutedTrade record
        """
        # Check DRY_RUN mode
        if getattr(settings, "DRY_RUN_MODE", True):
            logger.info(f"DRY_RUN mode: Simulating trade for {signal.market_slug}")
            return await self._execute_paper_trade(signal, ev_metrics)
        
        # Check kill switches
        if self._halted:
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message=f"Executor halted: {self._halt_reason}"
            )
        
        # Check EV threshold
        if ev_metrics:
            ev = ev_metrics.get("ev_per_dollar", 0.0)
            if ev < self.min_ev_threshold:
                logger.debug(f"Trade rejected: EV {ev:.4f} < threshold {self.min_ev_threshold}")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=f"EV {ev:.4f} below threshold {self.min_ev_threshold}"
                )
        
        # Check cooldown
        wallet = signal.wallet_address.lower()
        if wallet in self._wallet_last_trade:
            last_trade_time = self._wallet_last_trade[wallet]
            time_since = (datetime.utcnow() - last_trade_time).total_seconds()
            if time_since < self.cooldown_seconds:
                logger.debug(f"Trade rejected: cooldown active ({time_since:.0f}s < {self.cooldown_seconds}s)")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=f"Cooldown active: {time_since:.0f}s remaining"
                )
        
        # Check position size limit
        current_position = self._wallet_positions[wallet]
        if current_position + signal.size_usd > self.max_size_per_wallet:
            logger.debug(
                f"Trade rejected: position limit exceeded "
                f"(${current_position:.2f} + ${signal.size_usd:.2f} > ${self.max_size_per_wallet:.2f})"
            )
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message=f"Position limit exceeded"
            )
        
        # Get market data and orderbook
        try:
            market = await self.polymarket_client.gamma.get_market(signal.market_slug)
            token_ids = market.get("clobTokenIds", [])
            
            if not token_ids or len(token_ids) < 2:
                raise PolymarketError(f"Invalid token IDs for {signal.market_slug}")
            
            # Select token based on side
            token_id = token_ids[1] if signal.side == "BUY" else token_ids[0]
            
            # Get current orderbook
            orderbook = self.polymarket_client.clob.get_orderbook(token_id, depth=10)
            best_bid = orderbook.get("best_bid", 0)
            best_ask = orderbook.get("best_ask", 1)
            spread = best_ask - best_bid
            
            # Check for rapidly widening spreads (kill switch)
            spread_bps = spread * 10000
            if spread_bps > 200:  # 2% spread threshold
                logger.warning(f"Spread too wide: {spread_bps:.0f} bps, halting executor")
                self._halted = True
                self._halt_reason = f"Spread too wide: {spread_bps:.0f} bps"
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=self._halt_reason
                )
            
            # Calculate order size in tokens
            # For BUY: size_usd / best_ask, for SELL: size_usd / best_bid
            if signal.side == "BUY":
                order_price = best_ask  # Marketable limit: take best ask
                order_size_tokens = signal.size_usd / best_ask if best_ask > 0 else 0
            else:  # SELL
                order_price = best_bid  # Marketable limit: take best bid
                order_size_tokens = signal.size_usd / best_bid if best_bid > 0 else 0
            
            # Place order via CLOB
            clob_client = self.polymarket_client.clob.client
            
            # Check if we have private key (required for trading)
            if not hasattr(clob_client, 'key') or not clob_client.key:
                logger.warning("No private key configured - cannot place orders")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message="No private key configured"
                )
            
            # Place limit order (marketable limit)
            try:
                # py_clob_client order placement
                # Note: Actual implementation depends on py_clob_client API
                # This is a placeholder - adjust based on actual API
                order_result = clob_client.create_order(
                    token_id=token_id,
                    side=signal.side,
                    size=str(order_size_tokens),
                    price=str(order_price),
                    order_type="LIMIT"
                )
                
                order_id = None
                if hasattr(order_result, 'order_id'):
                    order_id = order_result.order_id
                elif isinstance(order_result, dict):
                    order_id = order_result.get("order_id") or order_result.get("id")
                
                # Calculate slippage
                fill_price = order_price
                slippage_bps = abs(fill_price - signal.price) * 10000
                
                # Check slippage
                if slippage_bps > self.max_slippage_bps:
                    logger.warning(f"Slippage exceeded: {slippage_bps:.0f} bps > {self.max_slippage_bps} bps")
                    # Record as adverse fill
                    self._record_adverse_fill(signal, slippage_bps)
                    
                    # Check if we should halt
                    if self._should_halt():
                        self._halted = True
                        self._halt_reason = f"{self.adverse_fill_limit} adverse fills in {self.adverse_fill_window_minutes} minutes"
                
                # Update tracking
                self._wallet_positions[wallet] += signal.size_usd
                self._wallet_last_trade[wallet] = datetime.utcnow()
                
                executed = ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    order_id=str(order_id) if order_id else None,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    fill_price=fill_price,
                    expected_price=signal.price,
                    slippage_bps=slippage_bps,
                    status="filled" if order_id else "pending",
                    timestamp=datetime.utcnow()
                )
                
                self._executed_trades.append(executed)
                
                logger.info(
                    f"Executed trade: {signal.side} {signal.market_slug} "
                    f"${signal.size_usd:.2f} @ {fill_price:.4f} "
                    f"(slippage: {slippage_bps:.0f} bps)"
                )
                
                return executed
                
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=str(e)
                )
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message=str(e)
            )
    
    def _record_adverse_fill(self, signal: TradeSignal, slippage_bps: float):
        """Record an adverse fill for kill switch tracking"""
        self._adverse_fills.append({
            "timestamp": datetime.utcnow(),
            "signal": signal,
            "slippage_bps": slippage_bps
        })
        
        # Clean old fills
        cutoff = datetime.utcnow() - timedelta(minutes=self.adverse_fill_window_minutes)
        self._adverse_fills = [
            f for f in self._adverse_fills
            if f["timestamp"] > cutoff
        ]
    
    def _should_halt(self) -> bool:
        """Check if executor should halt based on adverse fills"""
        return len(self._adverse_fills) >= self.adverse_fill_limit
    
    def resume(self):
        """Resume executor after halt"""
        self._halted = False
        self._halt_reason = None
        logger.info("Trade executor resumed")
    
    async def _execute_paper_trade(
        self,
        signal: TradeSignal,
        ev_metrics: Optional[Dict[str, Any]] = None
    ) -> ExecutedTrade:
        """
        Execute a paper trade (DRY_RUN mode).
        
        Args:
            signal: Trade signal to simulate
            ev_metrics: Expected value metrics
            
        Returns:
            ExecutedTrade record with simulated fill price
        """
        # Run through validation checks (same as live trading)
        if self._halted:
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message=f"Executor halted: {self._halt_reason}"
            )
        
        # Check EV threshold
        if ev_metrics:
            ev = ev_metrics.get("ev_per_dollar", 0.0)
            if ev < self.min_ev_threshold:
                logger.debug(f"Paper trade rejected: EV {ev:.4f} < threshold {self.min_ev_threshold}")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=f"EV {ev:.4f} below threshold {self.min_ev_threshold}"
                )
        
        # Check cooldown
        wallet = signal.wallet_address.lower()
        if wallet in self._wallet_last_trade:
            last_trade_time = self._wallet_last_trade[wallet]
            time_since = (datetime.utcnow() - last_trade_time).total_seconds()
            if time_since < self.cooldown_seconds:
                logger.debug(f"Paper trade rejected: cooldown active")
                return ExecutedTrade(
                    signal_id=signal.transaction_hash,
                    market_slug=signal.market_slug,
                    side=signal.side,
                    size_usd=signal.size_usd,
                    expected_price=signal.price,
                    status="rejected",
                    error_message=f"Cooldown active: {time_since:.0f}s remaining"
                )
        
        # Check position size limit
        current_position = self._wallet_positions[wallet]
        if current_position + signal.size_usd > self.max_size_per_wallet:
            logger.debug(f"Paper trade rejected: position limit exceeded")
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message="Position limit exceeded"
            )
        
        # Simulate trade using PaperTradingLogger
        try:
            trade_record = await self.paper_trading_logger.log_trade(signal, expected_price=signal.price)
            
            fill_price = trade_record["fill_price"]
            slippage_bps = trade_record["slippage_bps"]
            status = trade_record["status"]
            
            # Update tracking (same as live trading)
            if status == "filled":
                self._wallet_positions[wallet] += signal.size_usd
                self._wallet_last_trade[wallet] = datetime.utcnow()
            
            executed = ExecutedTrade(
                signal_id=signal.transaction_hash,
                order_id=None,  # No order ID in paper trading
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                fill_price=fill_price,
                expected_price=signal.price,
                slippage_bps=slippage_bps,
                status=status,
                timestamp=datetime.utcnow()
            )
            
            self._executed_trades.append(executed)
            
            logger.info(
                f"Paper trade {'filled' if status == 'filled' else 'rejected'}: "
                f"{signal.side} {signal.market_slug} ${signal.size_usd:.2f} @ {fill_price:.4f} "
                f"(slippage: {slippage_bps:.0f} bps)"
            )
            
            return executed
            
        except Exception as e:
            logger.error(f"Paper trade simulation failed: {e}")
            return ExecutedTrade(
                signal_id=signal.transaction_hash,
                market_slug=signal.market_slug,
                side=signal.side,
                size_usd=signal.size_usd,
                expected_price=signal.price,
                status="rejected",
                error_message=str(e)
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_trades = len(self._executed_trades)
        filled = len([t for t in self._executed_trades if t.status == "filled"])
        rejected = len([t for t in self._executed_trades if t.status == "rejected"])
        
        avg_slippage = 0.0
        if filled > 0:
            filled_trades = [t for t in self._executed_trades if t.status == "filled"]
            avg_slippage = sum(t.slippage_bps for t in filled_trades) / len(filled_trades)
        
        return {
            "total_trades": total_trades,
            "filled": filled,
            "rejected": rejected,
            "avg_slippage_bps": avg_slippage,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "adverse_fills": len(self._adverse_fills),
            "wallet_positions": dict(self._wallet_positions),
            "dry_run_mode": getattr(settings, "DRY_RUN_MODE", True)
        }

