"""
Bitquery Client for On-Chain Trader Analytics
Queries Polygon blockchain for CTF token transfers and calculates trader metrics.
"""
import logging
import asyncio
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from config import settings

logger = logging.getLogger(__name__)


class BitqueryError(Exception):
    """Base exception for Bitquery API errors"""
    pass


class BitqueryRateLimitError(BitqueryError):
    """Rate limit exceeded"""
    pass


class BitqueryClient:
    """
    Client for Bitquery GraphQL API to analyze Polygon CTF token transfers.
    
    Features:
    - Query ERC1155 transfers for CTF tokens
    - Calculate trader P&L using FIFO accounting
    - Detect early bets (trades within 24h of market creation)
    - Rate limiting (0.5s delay, 2 req/sec max)
    """
    
    # CTF Exchange contract addresses (Polygon)
    CTF_EXCHANGE_MAIN = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    CTF_EXCHANGE_NEG_RISK = "0xc5d563a36ae78145c45a50134d48a1215220f80a"
    
    # ERC1155 Transfer event signature
    TRANSFER_SINGLE_EVENT = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
    TRANSFER_BATCH_EVENT = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.5):
        """
        Initialize Bitquery client.
        
        Args:
            api_key: Bitquery API key (defaults to settings.BITQUERY_API_KEY)
            rate_limit_delay: Delay between requests in seconds (default 0.5s)
        """
        self.api_key = api_key or settings.BITQUERY_API_KEY
        self.base_url = settings.BITQUERY_API_URL or "https://graphql.bitquery.io"
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: Optional[datetime] = None
        
        if not self.api_key:
            logger.warning("Bitquery API key not configured - queries will fail")
    
    async def _make_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GraphQL request to Bitquery API with rate limiting.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Response JSON data
            
        Raises:
            BitqueryRateLimitError: If rate limit exceeded
            BitqueryError: For other API errors
        """
        if not self.api_key:
            raise BitqueryError("Bitquery API key not configured")
        
        # Rate limiting: ensure minimum delay between requests
        if self._last_request_time:
            elapsed = (datetime.utcnow() - self._last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                self._last_request_time = datetime.utcnow()
                
                # Check rate limit
                if response.status_code == 429:
                    raise BitqueryRateLimitError("Rate limit exceeded")
                
                response.raise_for_status()
                data = response.json()
                
                # Check for GraphQL errors
                if "errors" in data:
                    error_msg = "; ".join([e.get("message", "Unknown error") for e in data["errors"]])
                    raise BitqueryError(f"GraphQL errors: {error_msg}")
                
                return data.get("data", {})
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in Bitquery request: {e}")
            raise BitqueryError(f"HTTP error: {e}")
    
    async def get_trader_transfers(
        self,
        wallet_address: str,
        days: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch all CTF token transfers for a trader wallet.
        
        Args:
            wallet_address: Wallet address to query
            days: Number of days to look back
            limit: Maximum number of transfers to return
            
        Returns:
            List of transfer dicts with transaction details
        """
        # Normalize wallet address
        wallet_address = wallet_address.lower()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # GraphQL query for ERC1155 transfers
        query = """
        query GetCTFTransfers($wallet: String!, $from: ISO8601DateTime!, $to: ISO8601DateTime!, $limit: Int!) {
          ethereum(network: polygon) {
            transfers(
              currency: {is: "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"}
              date: {since: $from, till: $to}
              amount: {gt: 0}
              any: [{receiver: {is: $wallet}}, {sender: {is: $wallet}}]
            ) {
              transaction {
                hash
                timestamp {
                  time
                }
                block {
                  height
                }
              }
              currency {
                address
                symbol
              }
              amount
              receiver {
                address
              }
              sender {
                address
              }
              tokenId
            }
          }
        }
        """
        
        variables = {
            "wallet": wallet_address,
            "from": start_time.isoformat(),
            "to": end_time.isoformat(),
            "limit": limit
        }
        
        try:
            data = await self._make_request(query, variables)
            transfers = data.get("ethereum", {}).get("transfers", [])
            
            # Normalize transfer data
            normalized = []
            for transfer in transfers:
                tx = transfer.get("transaction", {})
                normalized.append({
                    "transaction_hash": tx.get("hash", ""),
                    "timestamp": datetime.fromisoformat(tx.get("timestamp", {}).get("time", "").replace("Z", "+00:00")),
                    "block_number": tx.get("block", {}).get("height", 0),
                    "token_address": transfer.get("currency", {}).get("address", ""),
                    "token_id": transfer.get("tokenId", ""),
                    "amount": float(transfer.get("amount", 0)),
                    "from_address": transfer.get("sender", {}).get("address", "").lower(),
                    "to_address": transfer.get("receiver", {}).get("address", "").lower(),
                    "wallet_address": wallet_address,
                    "direction": "BUY" if transfer.get("receiver", {}).get("address", "").lower() == wallet_address else "SELL"
                })
            
            logger.info(f"Retrieved {len(normalized)} transfers for wallet {wallet_address[:8]}...")
            return normalized[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching transfers for {wallet_address[:8]}...: {e}")
            return []
    
    def calculate_trader_pnl(self, transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trader P&L using FIFO accounting.
        
        Args:
            transfers: List of transfer dicts from get_trader_transfers()
            
        Returns:
            Dict with P&L metrics:
            - total_pnl: Total realized P&L
            - realized_pnl: Realized P&L from closed positions
            - unrealized_pnl: Unrealized P&L from open positions
            - trade_count: Number of trades
            - win_rate: Percentage of profitable trades
            - avg_trade_size: Average trade size in USD
        """
        if not transfers:
            return {
                "total_pnl": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_trade_size": 0.0
            }
        
        # Group transfers by token_id (market)
        positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for transfer in transfers:
            token_id = transfer.get("token_id", "")
            if not token_id:
                continue
            
            positions[token_id].append(transfer)
        
        # Calculate P&L per position using FIFO
        total_realized_pnl = 0.0
        open_positions: Dict[str, List[Dict[str, Any]]] = {}
        trade_count = 0
        
        for token_id, token_transfers in positions.items():
            # Sort by timestamp
            token_transfers.sort(key=lambda x: x.get("timestamp", datetime.min))
            
            # FIFO queue for this token
            fifo_queue: List[Dict[str, Any]] = []
            
            for transfer in token_transfers:
                direction = transfer.get("direction", "")
                amount = transfer.get("amount", 0.0)
                
                if direction == "BUY":
                    # Add to FIFO queue
                    fifo_queue.append({
                        "amount": amount,
                        "price": transfer.get("price", 0.0),  # Would need price from orderbook
                        "timestamp": transfer.get("timestamp")
                    })
                    trade_count += 1
                    
                elif direction == "SELL":
                    # Match against FIFO queue
                    remaining_sell = amount
                    
                    while remaining_sell > 0 and fifo_queue:
                        buy_order = fifo_queue[0]
                        buy_amount = buy_order["amount"]
                        buy_price = buy_order.get("price", 0.0)
                        sell_price = transfer.get("price", 0.0)
                        
                        if buy_amount <= remaining_sell:
                            # Fully consumed buy order
                            realized_pnl = (sell_price - buy_price) * buy_amount
                            total_realized_pnl += realized_pnl
                            remaining_sell -= buy_amount
                            fifo_queue.pop(0)
                        else:
                            # Partially consumed buy order
                            realized_pnl = (sell_price - buy_price) * remaining_sell
                            total_realized_pnl += realized_pnl
                            buy_order["amount"] -= remaining_sell
                            remaining_sell = 0
                    
                    trade_count += 1
            
            # Store remaining open positions
            if fifo_queue:
                open_positions[token_id] = fifo_queue
        
        # Calculate unrealized P&L (would need current market prices)
        unrealized_pnl = 0.0  # Placeholder - requires mark-to-market
        
        # Calculate win rate (simplified - would need resolved market data)
        win_rate = 0.0  # Placeholder - requires market resolution data
        
        # Calculate average trade size
        total_volume = sum(abs(t.get("amount", 0.0)) for t in transfers)
        avg_trade_size = total_volume / trade_count if trade_count > 0 else 0.0
        
        return {
            "total_pnl": total_realized_pnl + unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "avg_trade_size": avg_trade_size,
            "open_positions": len(open_positions)
        }
    
    async def detect_early_bets(
        self,
        transfers: List[Dict[str, Any]],
        market_creation_times: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """
        Identify trades placed within 24h of market creation.
        
        Args:
            transfers: List of transfer dicts
            market_creation_times: Dict mapping token_id to market creation datetime
            
        Returns:
            Dict with early bet metrics:
            - early_bet_count: Number of early bets
            - total_bet_count: Total number of bets
            - early_bet_pct: Percentage of early bets
            - early_bet_transfers: List of early bet transfers
        """
        early_bet_window_hours = getattr(settings, "TRADER_EARLY_BET_WINDOW_HOURS", 24)
        early_bet_transfers = []
        
        for transfer in transfers:
            token_id = transfer.get("token_id", "")
            if not token_id:
                continue
            
            market_created_at = market_creation_times.get(token_id)
            if not market_created_at:
                continue
            
            transfer_time = transfer.get("timestamp")
            if not isinstance(transfer_time, datetime):
                continue
            
            # Check if transfer is within early bet window
            time_diff = (transfer_time - market_created_at).total_seconds() / 3600
            if 0 <= time_diff <= early_bet_window_hours:
                early_bet_transfers.append(transfer)
        
        total_bets = len([t for t in transfers if t.get("direction") == "BUY"])
        early_bet_count = len([t for t in early_bet_transfers if t.get("direction") == "BUY"])
        early_bet_pct = (early_bet_count / total_bets * 100) if total_bets > 0 else 0.0
        
        return {
            "early_bet_count": early_bet_count,
            "total_bet_count": total_bets,
            "early_bet_pct": early_bet_pct,
            "early_bet_transfers": early_bet_transfers
        }
    
    async def get_trader_metrics(self, wallet_address: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive trader metrics.
        
        Args:
            wallet_address: Wallet address to analyze
            days: Number of days to analyze
            
        Returns:
            Dict with trader metrics:
            - wallet_address
            - trade_count
            - total_volume
            - pnl_metrics (from calculate_trader_pnl)
            - early_bet_metrics (from detect_early_bets)
            - activity_level: Trades per week
        """
        # Fetch transfers
        transfers = await self.get_trader_transfers(wallet_address, days=days)
        
        if not transfers:
            return {
                "wallet_address": wallet_address,
                "trade_count": 0,
                "total_volume": 0.0,
                "pnl_metrics": {},
                "early_bet_metrics": {},
                "activity_level": 0.0
            }
        
        # Calculate P&L
        pnl_metrics = self.calculate_trader_pnl(transfers)
        
        # Calculate total volume
        total_volume = sum(abs(t.get("amount", 0.0)) for t in transfers)
        
        # Calculate activity level (trades per week)
        if transfers:
            first_trade = min(t.get("timestamp", datetime.utcnow()) for t in transfers)
            last_trade = max(t.get("timestamp", datetime.utcnow()) for t in transfers)
            days_span = max((last_trade - first_trade).days, 1)
            trades_per_week = (len(transfers) / days_span) * 7
        else:
            trades_per_week = 0.0
        
        # Early bet detection (requires market creation times - placeholder)
        early_bet_metrics = {
            "early_bet_count": 0,
            "total_bet_count": len([t for t in transfers if t.get("direction") == "BUY"]),
            "early_bet_pct": 0.0,
            "early_bet_transfers": []
        }
        
        return {
            "wallet_address": wallet_address,
            "trade_count": len(transfers),
            "total_volume": total_volume,
            "pnl_metrics": pnl_metrics,
            "early_bet_metrics": early_bet_metrics,
            "activity_level": trades_per_week,
            "transfers": transfers  # Include raw transfers for further analysis
        }

