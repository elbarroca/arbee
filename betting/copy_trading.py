"""
Copy Trading Core Logic
Validates and processes webhook trade events for copy trading.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from clients.polymarket import PolymarketClient
from clients.web3.wallet_tracker import WalletTrackerClient
from agents.copy_trading_agent import CopyTradingAgent
from utils.token_resolver import get_token_resolver
from config import settings

logger = logging.getLogger(__name__)


class TradeSignal(BaseModel):
    """Validated trade signal ready for execution"""
    wallet_address: str = Field(..., description="Source wallet address")
    market_slug: str = Field(..., description="Market slug")
    token_address: Optional[str] = Field(None, description="CTF token address")
    side: str = Field(..., description="Trade side: BUY or SELL")
    size_usd: float = Field(..., description="Trade size in USD")
    price: float = Field(..., description="Trade price (0-1)")
    transaction_hash: str = Field(..., description="Source transaction hash")
    block_number: Optional[int] = Field(None, description="Block number")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_provider: str = Field(..., description="Webhook provider (alchemy, quicknode, moralis)")
    raw_event: Dict[str, Any] = Field(default_factory=dict)


class TradeSignalProcessor:
    """
    Processes webhook trade events and validates them for copy trading.
    
    Validation checks:
    - Market exists and is active
    - Sufficient liquidity
    - Trade size > minimum threshold
    - Within slippage tolerance
    - Trader is in copy list
    """
    
    def __init__(
        self,
        copy_agent: CopyTradingAgent,
        wallet_tracker: Optional[WalletTrackerClient] = None,
        polymarket_client: Optional[PolymarketClient] = None,
        min_trade_size_usd: float = 10.0,
        min_liquidity_usd: float = 1000.0,
        max_slippage_bps: int = 50
    ):
        """
        Initialize trade signal processor.
        
        Args:
            copy_agent: Copy trading agent with trader list
            wallet_tracker: Wallet tracker client (optional)
            polymarket_client: Polymarket client for market validation
            min_trade_size_usd: Minimum trade size to copy
            min_liquidity_usd: Minimum market liquidity required
            max_slippage_bps: Maximum acceptable slippage in basis points
        """
        self.copy_agent = copy_agent
        self.wallet_tracker = wallet_tracker or WalletTrackerClient()
        self.polymarket_client = polymarket_client or PolymarketClient()
        self.min_trade_size_usd = min_trade_size_usd
        self.min_liquidity_usd = min_liquidity_usd
        self.max_slippage_bps = max_slippage_bps
        
        # Track recent signals to avoid duplicates
        self._recent_signals: Dict[str, datetime] = {}
        self._signal_cooldown_seconds = 30
    
    async def process_webhook_event(
        self,
        webhook_event: Dict[str, Any]
    ) -> Optional[TradeSignal]:
        """
        Process a webhook event and return validated trade signal.
        
        Args:
            webhook_event: Raw webhook payload
            
        Returns:
            Validated TradeSignal or None if invalid/rejected
        """
        # Parse webhook event
        parsed = self.wallet_tracker.detect_trade_signal(webhook_event)
        if not parsed:
            logger.debug("Webhook event did not contain valid trade signal")
            return None
        
        wallet_address = parsed.get("wallet_address")
        if not wallet_address:
            logger.debug("Trade signal missing wallet address")
            return None
        
        # Check if trader is in copy list
        if not self.copy_agent.is_trader_active(wallet_address):
            logger.debug(f"Wallet {wallet_address[:8]}... not in active copy list")
            return None
        
        # Check for duplicate signals (same tx hash)
        tx_hash = parsed.get("transaction_hash", "")
        if tx_hash:
            if tx_hash in self._recent_signals:
                last_seen = self._recent_signals[tx_hash]
                if (datetime.utcnow() - last_seen).total_seconds() < self._signal_cooldown_seconds:
                    logger.debug(f"Duplicate signal detected: {tx_hash[:16]}...")
                    return None
        
        # Extract market information from token address or transaction
        market_slug = await self._extract_market_slug(parsed)
        if not market_slug:
            logger.debug("Could not extract market slug from trade signal")
            return None
        
        # Validate market
        market_valid = await self._validate_market(market_slug)
        if not market_valid:
            logger.debug(f"Market validation failed: {market_slug}")
            return None
        
        # Extract trade details
        trade_size = parsed.get("amount") or parsed.get("value", 0)
        if isinstance(trade_size, str):
            try:
                trade_size = float(trade_size)
            except ValueError:
                trade_size = 0
        
        # Convert to USD if needed (simplified - would need price oracle)
        size_usd = trade_size  # Assume already in USD for now
        
        if size_usd < self.min_trade_size_usd:
            logger.debug(f"Trade size too small: ${size_usd:.2f} < ${self.min_trade_size_usd:.2f}")
            return None
        
        # Determine trade side (BUY/SELL) - simplified logic
        side = self._determine_trade_side(parsed)
        
        # Get current market price
        current_price = await self._get_market_price(market_slug)
        if current_price is None:
            logger.debug(f"Could not get market price for {market_slug}")
            return None
        
        # Check slippage
        trade_price = parsed.get("price") or current_price
        slippage_bps = abs(trade_price - current_price) * 10000
        
        if slippage_bps > self.max_slippage_bps:
            logger.debug(f"Slippage too high: {slippage_bps} bps > {self.max_slippage_bps} bps")
            return None
        
        # Create validated trade signal
        signal = TradeSignal(
            wallet_address=wallet_address,
            market_slug=market_slug,
            token_address=parsed.get("token_address"),
            side=side,
            size_usd=size_usd,
            price=current_price,  # Use current price, not trade price
            transaction_hash=tx_hash,
            block_number=parsed.get("block_number"),
            timestamp=datetime.utcnow(),
            source_provider=parsed.get("provider", "unknown"),
            raw_event=parsed
        )
        
        # Record signal to prevent duplicates
        if tx_hash:
            self._recent_signals[tx_hash] = datetime.utcnow()
        
        # Clean old signals (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self._recent_signals = {
            k: v for k, v in self._recent_signals.items()
            if v > cutoff
        }
        
        logger.info(
            f"Validated trade signal: {side} {market_slug} "
            f"${size_usd:.2f} @ {current_price:.4f} "
            f"(wallet: {wallet_address[:8]}...)"
        )
        
        return signal
    
    async def _extract_market_slug(self, parsed_event: Dict[str, Any]) -> Optional[str]:
        """
        Extract market slug from parsed event using token resolution.

        Resolution strategy:
        1. Check if market slug is directly in event
        2. Resolve token address to market using Gamma API
        3. Cache result to reduce API calls
        """
        # Try to get from raw event (if already resolved)
        raw_event = parsed_event.get("raw_event", {})

        # Check for market slug in metadata
        if "market_slug" in raw_event:
            return raw_event["market_slug"]

        # Check for market identifier in event data
        event_data = parsed_event.get("event_data", {})
        if "market" in event_data:
            return event_data["market"]

        # Try to resolve via token address
        token_address = parsed_event.get("token_address")
        if not token_address:
            # Try to extract from raw event
            token_address = raw_event.get("token_address") or raw_event.get("rawContract", {}).get("address")

        if token_address:
            logger.debug(f"Resolving token {token_address[:16]}... to market slug")

            # Use token resolver to find market
            resolver = get_token_resolver()
            market_data = await resolver.resolve_token_to_market(token_address)

            if market_data:
                market_slug = market_data.get("slug") or market_data.get("id")
                logger.info(f"Resolved token to market: {market_slug}")

                # Store matched token info for later use (trade side detection)
                if "matched_token" in market_data:
                    parsed_event["matched_token"] = market_data["matched_token"]

                return market_slug
            else:
                logger.warning(f"Token {token_address} not found in any market")
                return None

        # Could not extract market slug
        logger.debug("No token address or market identifier found in event")
        return None
    
    async def _validate_market(self, market_slug: str) -> bool:
        """Validate that market exists, is active, and has sufficient liquidity"""
        try:
            market = await self.polymarket_client.gamma.get_market(market_slug)
            
            if not market.get("active", False):
                logger.debug(f"Market {market_slug} is not active")
                return False
            
            # Check liquidity via orderbook
            token_ids = market.get("clobTokenIds", [])
            if not token_ids:
                logger.debug(f"Market {market_slug} has no token IDs")
                return False
            
            try:
                orderbook = self.polymarket_client.clob.get_orderbook(token_ids[0], depth=10)
                liquidity = orderbook.get("total_liquidity", 0)
                
                if liquidity < self.min_liquidity_usd:
                    logger.debug(f"Market {market_slug} liquidity too low: ${liquidity:.2f}")
                    return False
                
                return True
            except Exception as e:
                logger.warning(f"Failed to check orderbook for {market_slug}: {e}")
                return False
        except Exception as e:
            logger.warning(f"Market validation failed for {market_slug}: {e}")
            return False
    
    async def _get_market_price(self, market_slug: str) -> Optional[float]:
        """Get current market price"""
        try:
            market = await self.polymarket_client.gamma.get_market(market_slug)
            token_ids = market.get("clobTokenIds", [])
            
            if not token_ids or len(token_ids) < 2:
                return None
            
            # Get YES token price
            yes_token_id = token_ids[1]
            orderbook = self.polymarket_client.clob.get_orderbook(yes_token_id, depth=5)
            return orderbook.get("mid_price")
        except Exception as e:
            logger.warning(f"Failed to get market price for {market_slug}: {e}")
            return None
    
    def _determine_trade_side(self, parsed_event: Dict[str, Any]) -> str:
        """
        Determine trade side (BUY/SELL) from event using CTF Transfer analysis.

        Logic:
        - If matched_token info is available, use the side directly
        - Otherwise analyze Transfer event:
          - wallet sending tokens to CTF Exchange = SELL
          - wallet receiving tokens from CTF Exchange = BUY
        - Fall back to event name analysis
        """
        # Use matched token side if available (most accurate)
        if "matched_token" in parsed_event:
            side = parsed_event["matched_token"].get("side", "YES")
            # For now, assume all YES outcomes are BUYs (simplified)
            return "BUY" if side == "YES" else "SELL"

        # Analyze Transfer event from raw event
        raw_event = parsed_event.get("raw_event", {})

        # Get wallet address and transaction parties
        wallet_address = parsed_event.get("wallet_address", "").lower()
        from_address = raw_event.get("from", "").lower()
        to_address = raw_event.get("to", "").lower()

        # CTF Exchange contracts (normalized)
        ctf_exchanges = [
            "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",  # Main CTF
            "0xc5d563a36ae78145c45a50134d48a1215220f80a",  # Neg Risk CTF
        ]

        # Check if this is a token transfer
        if from_address and to_address:
            # SELL: Wallet sending tokens to CTF Exchange
            if from_address == wallet_address and any(to_address == exchange for exchange in ctf_exchanges):
                logger.debug(f"Detected SELL (wallet -> CTF Exchange)")
                return "SELL"

            # BUY: Wallet receiving tokens from CTF Exchange
            if to_address == wallet_address and any(from_address == exchange for exchange in ctf_exchanges):
                logger.debug(f"Detected BUY (CTF Exchange -> wallet)")
                return "BUY"

        # Fallback: Check event name
        event_name = parsed_event.get("event_name", "")
        if "Fill" in event_name or "Buy" in event_name:
            return "BUY"
        if "Sell" in event_name:
            return "SELL"

        # Final fallback: Default to BUY (safer assumption for copy trading)
        logger.debug("Could not determine trade side, defaulting to BUY")
        return "BUY"
    
    def calculate_ev_and_kelly(
        self,
        signal: TradeSignal,
        p_bayesian: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate expected value and Kelly fraction for trade signal.
        
        Args:
            signal: Trade signal
            p_bayesian: Bayesian probability (if available from analysis)
            
        Returns:
            Dict with ev, kelly_fraction, and other metrics
        """
        market_price = signal.price
        p_market = market_price
        
        # Use p_bayesian if available, otherwise use market price
        p_true = p_bayesian if p_bayesian is not None else p_market
        
        # Calculate edge
        if signal.side == "BUY":
            edge = p_true - p_market
        else:  # SELL
            edge = p_market - (1 - p_true)
        
        # Expected value per dollar
        ev_per_dollar = edge
        
        # Kelly fraction (conservative, capped)
        if signal.side == "BUY":
            kelly = edge / (1 - p_market) if p_market < 1.0 else 0.0
        else:
            kelly = edge / p_market if p_market > 0.0 else 0.0
        
        # Cap Kelly at 5%
        max_kelly = getattr(settings, "MAX_KELLY_FRACTION", 0.05)
        kelly = min(kelly, max_kelly)
        kelly = max(0.0, kelly)  # No negative Kelly
        
        return {
            "edge": edge,
            "ev_per_dollar": ev_per_dollar,
            "kelly_fraction": kelly,
            "p_market": p_market,
            "p_true": p_true,
            "suggested_stake_usd": signal.size_usd * kelly if kelly > 0 else 0.0
        }

