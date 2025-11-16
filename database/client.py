"""
Async database client for Supabase with prediction market data persistence.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from supabase import create_client, Client
from .schema import Event, Market, ScanSession


class MarketDatabase:
    """Async database client for storing prediction market data.
    
    Provides high-level async interface for persisting events, markets, and scan sessions
    to Supabase with batch support and thread-safe operations.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize database client with Supabase credentials.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            
        Raises:
            AssertionError: If URL or key is empty
        """
        assert supabase_url, "supabase_url must not be empty"
        assert supabase_key, "supabase_key must not be empty"
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._lock = asyncio.Lock()

    @staticmethod
    def _serialize(obj: Any) -> Optional[str]:
        """Serialize object to JSON if not None.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON string or None
        """
        return json.dumps(obj) if obj else None

    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert Event schema to database record.
        
        Args:
            event: Event object
            
        Returns:
            Dictionary ready for database insertion
        """
        assert event.id, "Event must have id"
        assert event.platform, "Event must have platform"
        
        return {
            "id": event.id,
            "platform": event.platform,
            "title": event.title,
            "description": event.description,
            "category": event.category,
            "status": event.status,
            "start_date": event.start_date,
            "end_date": event.end_date,
            "tags": self._serialize(event.tags),
            "market_count": event.market_count,
            "total_liquidity": event.total_liquidity,
            "created_at": event.created_at,
            "updated_at": event.updated_at,
            "raw_data": self._serialize(event.raw_data),
        }

    def _market_to_dict(self, market: Market) -> Dict[str, Any]:
        """Convert Market schema to database record.
        
        Args:
            market: Market object
            
        Returns:
            Dictionary ready for database insertion
        """
        assert market.id, "Market must have id"
        assert market.platform, "Market must have platform"
        
        return {
            "id": market.id,
            "platform": market.platform,
            "event_id": market.event_id,
            "event_title": market.event_title,
            "title": market.title,
            "description": market.description,
            "category": market.category,
            "tags": self._serialize(market.tags),
            "status": market.status,
            "p_yes": market.p_yes,
            "p_no": market.p_no,
            "bid": market.bid,
            "ask": market.ask,
            "liquidity": market.liquidity,
            "volume_24h": market.volume_24h,
            "total_volume": market.total_volume,
            "num_outcomes": market.num_outcomes,
            "created_at": market.created_at,
            "updated_at": market.updated_at,
            "close_date": market.close_date,
            "raw_data": self._serialize(market.raw_data),
        }

    async def save_event(self, event: Event) -> bool:
        """Save a single event to the database.
        
        Args:
            event: Event object to persist
            
        Returns:
            True if upsert succeeded, False otherwise
        """
        async with self._lock:
            data = self._event_to_dict(event)
            result = self.supabase.table("events").upsert(data).execute()
            return len(result.data) > 0

    async def save_market(self, market: Market) -> bool:
        """Save a single market to the database.
        
        Args:
            market: Market object to persist
            
        Returns:
            True if upsert succeeded, False otherwise
        """
        async with self._lock:
            data = self._market_to_dict(market)
            result = self.supabase.table("markets").upsert(data).execute()
            return len(result.data) > 0

    async def save_events_batch(self, events: List[Event]) -> int:
        """Save multiple events in a batch.
        
        Args:
            events: List of Event objects to persist
            
        Returns:
            Count of successfully saved events
        """
        assert events, "events list must not be empty"
        
        saved_count = 0
        for event in events:
            if await self.save_event(event):
                saved_count += 1
        return saved_count

    async def save_markets_batch(self, markets: List[Market]) -> int:
        """Save multiple markets in a batch.
        
        Args:
            markets: List of Market objects to persist
            
        Returns:
            Count of successfully saved markets
        """
        assert markets, "markets list must not be empty"
        
        saved_count = 0
        for market in markets:
            if await self.save_market(market):
                saved_count += 1
        return saved_count

    async def create_scan_session(self, platforms: List[str]) -> Optional[str]:
        """Create a new scan session record.
        
        Args:
            platforms: List of platforms scanned
            
        Returns:
            Session ID if created, None otherwise
        """
        assert platforms, "platforms list must not be empty"
        
        data = {
            "started_at": datetime.utcnow().isoformat(),
            "platforms_scanned": json.dumps(platforms),
            "status": "running"
        }
        result = self.supabase.table("scan_sessions").insert(data).execute()
        return result.data[0]["id"] if result.data else None

    async def update_scan_session(self, session_id: str, **updates) -> None:
        """Update a scan session with completion stats.
        
        Args:
            session_id: ID of scan session to update
            **updates: Fields to update
        """
        assert session_id, "session_id must not be empty"
        assert updates, "updates must contain at least one field"
        
        updates["db_updated_at"] = datetime.utcnow().isoformat()
        self.supabase.table("scan_sessions").update(updates).eq("id", session_id).execute()

    async def get_recent_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recently added markets for verification.
        
        Args:
            limit: Maximum number of markets to retrieve (default: 100)
            
        Returns:
            List of market records ordered by creation date (newest first)
        """
        assert limit > 0, "limit must be positive"
        
        result = (self.supabase.table("markets")
                 .select("*")
                 .order("created_at", desc=True)
                 .limit(limit)
                 .execute())
        return result.data if result.data else []

    async def get_market_count(self) -> int:
        """Get total count of markets in database.

        Returns:
            Total number of market records
        """
        result = self.supabase.table("markets").select("id", count="exact").execute()
        return result.count or 0

    # ─────────────────────────────────────────────────────────────────────────
    # BULK OPERATIONS FOR WALLET TRACKING (High-Performance)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_existing_wallets_batch(
        self, wallet_addresses: List[str], staleness_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """Get existing wallet info to determine which need processing.

        Args:
            wallet_addresses: List of proxy_wallet addresses to check
            staleness_hours: Consider wallet stale if not updated in this many hours

        Returns:
            Dict mapping wallet address to {exists: bool, is_stale: bool, last_sync_at: str}
        """
        if not wallet_addresses:
            return {}

        # Query in batches of 500 (Supabase limit)
        result_map = {}
        batch_size = 500

        for i in range(0, len(wallet_addresses), batch_size):
            batch = wallet_addresses[i:i + batch_size]

            result = (
                self.supabase.table("wallets")
                .select("proxy_wallet, last_sync_at, updated_at")
                .in_("proxy_wallet", batch)
                .execute()
            )

            # Build lookup for this batch
            existing = {row["proxy_wallet"]: row for row in (result.data or [])}

            cutoff = datetime.now(timezone.utc).timestamp() - (staleness_hours * 3600)

            for addr in batch:
                if addr in existing:
                    row = existing[addr]
                    last_sync = row.get("last_sync_at") or row.get("updated_at")
                    is_stale = True

                    if last_sync:
                        try:
                            sync_ts = datetime.fromisoformat(last_sync.replace("Z", "+00:00")).timestamp()
                            is_stale = sync_ts < cutoff
                        except (ValueError, TypeError):
                            is_stale = True

                    result_map[addr] = {
                        "exists": True,
                        "is_stale": is_stale,
                        "last_sync_at": last_sync
                    }
                else:
                    result_map[addr] = {
                        "exists": False,
                        "is_stale": True,  # New wallets are always "stale"
                        "last_sync_at": None
                    }

        return result_map

    async def upsert_wallets_bulk(self, wallets: List[Dict[str, Any]]) -> int:
        """Bulk upsert wallet records.

        Args:
            wallets: List of wallet dictionaries (proxy_wallet as key)

        Returns:
            Count of successfully upserted wallets
        """
        if not wallets:
            return 0

        # Supabase bulk upsert limit is ~500-1000 records
        batch_size = 500
        total_saved = 0

        for i in range(0, len(wallets), batch_size):
            batch = wallets[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallets")
                    .upsert(batch, on_conflict="proxy_wallet")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk wallet upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_trades_bulk(self, trades: List[Dict[str, Any]]) -> int:
        """Bulk upsert trade records.

        Args:
            trades: List of trade dictionaries

        Returns:
            Count of successfully upserted trades
        """
        if not trades:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(trades), batch_size):
            batch = trades[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("trades")
                    .upsert(batch, on_conflict="id")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk trades upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_closed_positions_bulk(self, positions: List[Dict[str, Any]]) -> int:
        """Bulk upsert closed position records.

        Args:
            positions: List of position dictionaries

        Returns:
            Count of successfully upserted positions
        """
        if not positions:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_closed_positions")
                    .upsert(batch, on_conflict="id")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk positions upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_wallet_stats_bulk(self, stats: List[Dict[str, Any]]) -> int:
        """Bulk upsert wallet statistics.

        Args:
            stats: List of wallet stats dictionaries

        Returns:
            Count of successfully upserted stats
        """
        if not stats:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(stats), batch_size):
            batch = stats[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_stats")
                    .upsert(batch, on_conflict="proxy_wallet")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk stats upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_wallet_scores_bulk(self, scores: List[Dict[str, Any]]) -> int:
        """Bulk upsert wallet scores.

        Args:
            scores: List of wallet score dictionaries

        Returns:
            Count of successfully upserted scores
        """
        if not scores:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(scores), batch_size):
            batch = scores[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_scores")
                    .upsert(batch, on_conflict="proxy_wallet")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk scores upsert error (batch {i}): {e}")

        return total_saved

    async def get_wallet_count(self) -> int:
        """Get total count of wallets in database.

        Returns:
            Total number of wallet records
        """
        result = self.supabase.table("wallets").select("proxy_wallet", count="exact").execute()
        return result.count or 0

    async def get_trades_count(self) -> int:
        """Get total count of trades in database.

        Returns:
            Total number of trade records
        """
        result = self.supabase.table("trades").select("id", count="exact").execute()
        return result.count or 0

    async def upsert_wallet_tag_stats_bulk(self, tag_stats: List[Dict[str, Any]]) -> int:
        """Bulk upsert wallet tag statistics.

        Args:
            tag_stats: List of tag stats dictionaries (id=proxy_wallet_tag)

        Returns:
            Count of successfully upserted tag stats
        """
        if not tag_stats:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(tag_stats), batch_size):
            batch = tag_stats[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_tag_stats")
                    .upsert(batch, on_conflict="id")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk tag stats upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_wallet_market_stats_bulk(self, market_stats: List[Dict[str, Any]]) -> int:
        """Bulk upsert wallet market statistics (concentration analysis).

        Args:
            market_stats: List of market stats dictionaries (id=proxy_wallet_condition_id)

        Returns:
            Count of successfully upserted market stats
        """
        if not market_stats:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(market_stats), batch_size):
            batch = market_stats[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_market_stats")
                    .upsert(batch, on_conflict="id")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk market stats upsert error (batch {i}): {e}")

        return total_saved

    async def upsert_open_positions_bulk(self, positions: List[Dict[str, Any]]) -> int:
        """Bulk upsert open position records.

        Args:
            positions: List of open position dictionaries

        Returns:
            Count of successfully upserted positions
        """
        if not positions:
            return 0

        batch_size = 500
        total_saved = 0

        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            try:
                result = (
                    self.supabase.table("wallet_open_positions")
                    .upsert(batch, on_conflict="id")
                    .execute()
                )
                total_saved += len(result.data) if result.data else 0
            except Exception as e:
                print(f"    ⚠️ Bulk open positions upsert error (batch {i}): {e}")

        return total_saved