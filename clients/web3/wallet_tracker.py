"""
Wallet Tracker API Client
Tracks known insider wallets and detects activity patterns.
Integrates with Alchemy webhooks for real-time tracking.
"""
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from config.settings import settings
from .alchemy import AlchemyWebhooksClient

logger = logging.getLogger(__name__)


class WalletTrackerClient:
    """
    Client for tracking wallet activity on prediction markets.
    
    Integrates with Alchemy webhooks for real-time wallet tracking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "alchemy",  # "alchemy" or "auto"
        webhook_url: Optional[str] = None
    ):
        """
        Initialize wallet tracker client.
        
        Args:
            api_key: API key (provider-specific, from settings if not provided)
            provider: Webhook provider to use ("alchemy" or "auto")
            webhook_url: Webhook endpoint URL for receiving events
        """
        self.provider = provider
        self.webhook_url = webhook_url or getattr(settings, "WEBHOOK_URL", "")
        self.insider_wallets = getattr(settings, "INSIDER_WALLET_ADDRESSES", [])
        
        # Initialize providers
        self.alchemy_client = None
        
        # Enable providers based on configuration
        if provider in ["alchemy", "auto"]:
            try:
                self.alchemy_client = AlchemyWebhooksClient(webhook_url=self.webhook_url)
                if self.alchemy_client.api_key:
                    self.enabled = True
            except Exception as e:
                logger.warning(f"Alchemy client initialization failed: {e}")
        
        # Fallback to enabled flag from settings
        if not hasattr(self, 'enabled'):
            self.enabled = getattr(settings, "ENABLE_INSIDER_TRACKING", False)
        
        if not self.enabled:
            logger.info(
                "Wallet tracker initialized in mock mode (no API keys provided)"
            )
        
        # Track subscribed wallets
        self._subscribed_wallets: Dict[str, Dict[str, Any]] = {}

    async def get_wallet_trades(
        self,
        wallet_address: str,
        lookback_hours: int = 24,
        market_slug: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a specific wallet.
        
        Args:
            wallet_address: Wallet address to query
            lookback_hours: Hours to look back for trades
            market_slug: Optional market slug to filter trades
            
        Returns:
            List of trade dicts with keys: market_slug, token_address, 
            trade_size, price, timestamp, transaction_hash
        """
        if not self.enabled:
            return []
        
        trades = []
        
        # Use Alchemy if available
        if self.alchemy_client and self.alchemy_client.api_key:
            # Calculate block range (approximate: 2 seconds per block on Polygon)
            blocks_per_hour = 1800  # ~2 sec/block
            to_block = None  # Latest
            from_block = None  # Will be calculated if needed
            
            transactions = await self.alchemy_client.get_address_transactions(
                wallet_address,
                limit=100
            )
            
            # Filter by time window
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            for tx in transactions:
                tx_time = tx.get("metadata", {}).get("blockTimestamp")
                if tx_time:
                    try:
                        tx_dt = datetime.fromisoformat(tx_time.replace('Z', '+00:00'))
                        if tx_dt >= cutoff_time:
                            trades.append({
                                "wallet_address": wallet_address,
                                "transaction_hash": tx.get("hash", ""),
                                "block_number": tx.get("blockNum", ""),
                                "timestamp": tx_time,
                                "value": tx.get("value", 0),
                                "raw_tx": tx
                            })
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Failed to parse transaction timestamp: {e}")
        
        return trades
    
    async def get_wallet_positions(
        self, wallet_address: str, market_slug: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get positions for a specific wallet (legacy method, uses get_wallet_trades).
        
        Args:
            wallet_address: Wallet address to query
            market_slug: Optional market slug to filter positions
            
        Returns:
            List of position dicts with keys: market_slug, position_size, 
            entry_price, current_price, pnl, timestamp
        """
        trades = await self.get_wallet_trades(wallet_address, lookback_hours=24, market_slug=market_slug)
        
        # Convert trades to positions format (simplified)
        positions = []
        for trade in trades:
            positions.append({
                "market_slug": market_slug or "unknown",
                "position_size": trade.get("value", 0),
                "entry_price": None,  # Would need market data to calculate
                "current_price": None,
                "pnl": None,
                "timestamp": trade.get("timestamp")
            })
        
        return positions

    async def subscribe_to_wallet(
        self,
        wallet_address: str,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Subscribe to a wallet address for real-time tracking via webhooks.
        
        Args:
            wallet_address: Wallet address to subscribe to
            provider: Provider to use ("alchemy" or None for auto)
            
        Returns:
            Dict with subscription details (webhook_id, provider, etc.)
        """
        if not self.enabled:
            logger.warning("Wallet tracking not enabled - cannot subscribe")
            return {"success": False, "error": "Tracking not enabled"}
        
        provider = provider or self.provider
        
        # Try Alchemy provider
        if provider == "auto" or provider == "alchemy":
            if self.alchemy_client and self.alchemy_client.api_key:
                try:
                    result = await self.alchemy_client.create_address_activity_webhook(wallet_address)
                    self._subscribed_wallets[wallet_address] = {
                        "provider": "alchemy",
                        "webhook_id": result.get("id") or result.get("webhook_id"),
                        "subscribed_at": datetime.utcnow().isoformat()
                    }
                    return {"success": True, "provider": "alchemy", **result}
                except Exception as e:
                    logger.warning(f"Alchemy subscription failed: {e}")
                    if provider != "auto":
                        raise
        
        return {"success": False, "error": "No available providers"}
    
    def detect_trade_signal(self, webhook_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse and validate a webhook event to detect trade signals.
        
        Args:
            webhook_event: Raw webhook payload from provider
            
        Returns:
            Parsed trade signal dict or None if not a valid trade signal
        """
        if not webhook_event:
            return None
        
        # Try to parse with Alchemy parser
        parsed = None
        
        if self.alchemy_client:
            try:
                parsed = self.alchemy_client.parse_webhook_event(webhook_event)
                if parsed:
                    parsed["provider"] = "alchemy"
            except Exception:
                pass
        
        # Validate trade signal
        if parsed and parsed.get("event_type") == "trade_signal":
            # Additional validation
            if parsed.get("wallet_address") and parsed.get("transaction_hash"):
                return parsed
        
        return None
    
    async def detect_insider_activity(
        self, market_slug: str, lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Detect insider activity for a specific market.
        
        Args:
            market_slug: Market to analyze
            lookback_hours: Hours to look back for activity
            
        Returns:
            Dict with:
            - flagged_wallets: List of wallet addresses with suspicious activity
            - activity_patterns: List of detected patterns
            - confidence: Confidence score (0-1)
            - evidence: List of evidence strings
        """
        flagged_wallets = []
        activity_patterns = []
        evidence = []
        confidence = 0.0

        if not self.enabled or not self.insider_wallets:
            return {
                "flagged_wallets": [],
                "activity_patterns": [],
                "confidence": 0.0,
                "evidence": ["Wallet tracking not enabled or no insider wallets configured"],
            }

        # Check each insider wallet for activity
        for wallet in self.insider_wallets:
            trades = await self.get_wallet_trades(wallet, lookback_hours, market_slug)

            if trades:
                # Analyze trade timing and size
                for trade in trades:
                    trade_size = trade.get("value", 0)
                    trade_time = trade.get("timestamp")

                    # Flag large trades
                    if trade_size > 10000:  # $10k+ trade
                        flagged_wallets.append(wallet)
                        activity_patterns.append("large_trade")
                        evidence.append(
                            f"Wallet {wallet[:8]}... has large trade: ${trade_size:,.0f}"
                        )
                        confidence = max(confidence, 0.6)

                    # Flag recent trades
                    if trade_time:
                        try:
                            trade_dt = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                            time_diff = (datetime.utcnow() - trade_dt.replace(tzinfo=None)).total_seconds() / 3600
                            if time_diff < 6:  # Trade within last 6 hours
                                flagged_wallets.append(wallet)
                                activity_patterns.append("recent_trade")
                                evidence.append(
                                    f"Wallet {wallet[:8]}... traded {time_diff:.1f}h ago"
                                )
                                confidence = max(confidence, 0.5)
                        except Exception:
                            pass

        # Remove duplicates
        flagged_wallets = list(set(flagged_wallets))

        return {
            "flagged_wallets": flagged_wallets,
            "activity_patterns": list(set(activity_patterns)),
            "confidence": min(confidence, 0.9),
            "evidence": evidence,
            "market_slug": market_slug,
            "lookback_hours": lookback_hours,
        }

    async def get_coordinated_activity(
        self, market_slug: str, threshold_wallets: int = 3
    ) -> Dict[str, Any]:
        """
        Detect coordinated buying/selling across multiple wallets.
        
        Args:
            market_slug: Market to analyze
            threshold_wallets: Minimum number of wallets for coordination
            
        Returns:
            Dict with coordination detection results
        """
        if not self.enabled:
            return {
                "is_coordinated": False,
                "wallet_count": 0,
                "confidence": 0.0,
                "evidence": [],
            }

        # Get positions for all insider wallets
        all_positions = []
        for wallet in self.insider_wallets:
            positions = await self.get_wallet_positions(wallet, market_slug)
            all_positions.extend(
                [{"wallet": wallet, **pos} for pos in positions]
            )

        # Check if multiple wallets took positions around the same time
        if len(all_positions) >= threshold_wallets:
            # Group by time windows
            time_windows = {}
            for pos in all_positions:
                entry_time = pos.get("timestamp")
                if entry_time:
                    # Round to nearest hour
                    window = entry_time.replace(minute=0, second=0, microsecond=0)
                    if window not in time_windows:
                        time_windows[window] = []
                    time_windows[window].append(pos)

            # Check for coordination (multiple wallets in same time window)
            coordinated_windows = {
                w: positions
                for w, positions in time_windows.items()
                if len(positions) >= threshold_wallets
            }

            if coordinated_windows:
                return {
                    "is_coordinated": True,
                    "wallet_count": len(all_positions),
                    "coordinated_windows": len(coordinated_windows),
                    "confidence": 0.7,
                    "evidence": [
                        f"Found {len(coordinated_windows)} time windows with coordinated activity"
                    ],
                }

        return {
            "is_coordinated": False,
            "wallet_count": len(all_positions),
            "confidence": 0.0,
            "evidence": ["No coordinated activity detected"],
        }

