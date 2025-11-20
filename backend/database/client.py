"""
Async database client for Supabase with prediction market data persistence.
Assertive, clean, and optimized for batch operations.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from supabase import create_client, Client
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../frontend'))
from models.database_schema import Event, Market

class MarketDatabase:
    """Async database client for storing prediction market data."""

    def __init__(self, supabase_url: str, supabase_key: str):
        assert supabase_url, "supabase_url must not be empty"
        assert supabase_key, "supabase_key must not be empty"
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._lock = asyncio.Lock()

    # CORE UTILITIES

    @staticmethod
    def _serialize(obj: Any) -> Optional[str]:
        return json.dumps(obj) if obj else None

    async def _batch_upsert(self, table: str, data: List[Dict[str, Any]], unique_key: str) -> int:
        """
        Generic assertive batch upsert engine. 
        Fails loudly if DB operations fail.
        """
        if not data:
            return 0

        BATCH_SIZE = 500
        total_saved = 0

        # Process in chunks strictly
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            
            # Assertive execution: No try/catch. Raises APIError on failure.
            result = (
                self.supabase.table(table)
                .upsert(batch, on_conflict=unique_key)
                .execute()
            )
            
            # Assert we actually got data back
            assert result.data is not None, f"Database returned no data for {table} upsert"
            total_saved += len(result.data)

        return total_saved

    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
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
            "enriched": event.enriched,
            "retrieve_data": getattr(event, 'retrieve_data', False),
            "raw_data": self._serialize(event.raw_data),
        }

    def _market_to_dict(self, market: Market) -> Dict[str, Any]:
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

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE ENTITY OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    async def save_event(self, event: Event) -> bool:
        async with self._lock:
            data = self._event_to_dict(event)
            result = self.supabase.table("events").upsert(data).execute()
            return bool(result.data)

    async def save_market(self, market: Market) -> bool:
        async with self._lock:
            data = self._market_to_dict(market)
            result = self.supabase.table("markets").upsert(data).execute()
            return bool(result.data)

    async def update_event_enriched_status(self, event_id: str, enriched: bool) -> bool:
        assert event_id, "event_id required"
        async with self._lock:
            result = self.supabase.table("events").update({
                "enriched": enriched,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", event_id).execute()
            return bool(result.data)

    async def update_wallet_enriched_status(self, proxy_wallet: str, enriched: bool) -> bool:
        assert proxy_wallet, "proxy_wallet required"
        async with self._lock:
            result = self.supabase.table("wallets").update({
                "enriched": enriched,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("proxy_wallet", proxy_wallet).execute()
            return bool(result.data)

    async def create_scan_session(self, platforms: List[str]) -> str:
        assert platforms, "platforms list must not be empty"
        
        data = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platforms_scanned": json.dumps(platforms),
            "status": "running"
        }
        result = self.supabase.table("scan_sessions").insert(data).execute()
        assert result.data, "Failed to create scan session"
        return result.data[0]["id"]

    async def update_scan_session(self, session_id: str, **updates) -> None:
        assert session_id, "session_id required"
        assert updates, "updates required"
        
        updates["db_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.supabase.table("scan_sessions").update(updates).eq("id", session_id).execute()

    # BULK OPERATIONS (Refactored & Assertive)

    async def upsert_wallets_bulk(self, wallets: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("wallets", wallets, "proxy_wallet")

    async def upsert_trades_bulk(self, trades: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("trades", trades, "id")

    async def upsert_closed_positions_bulk(self, positions: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("wallet_closed_positions", positions, "id")

    async def upsert_open_positions_bulk(self, positions: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("elite_open_positions", positions, "id")

    async def upsert_elite_trader_performance_bulk(self, records: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("elite_trader_performance", records, "id")

    async def upsert_wallet_tag_stats_bulk(self, stats: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("wallet_tag_stats", stats, "id")

    async def upsert_wallet_market_stats_bulk(self, stats: List[Dict[str, Any]]) -> int:
        return await self._batch_upsert("wallet_market_stats", stats, "id")

    async def save_events_batch(self, events: List[Event]) -> int:
        data = [self._event_to_dict(e) for e in events]
        return await self._batch_upsert("events", data, "id")

    async def save_markets_batch(self, markets: List[Market]) -> int:
        data = [self._market_to_dict(m) for m in markets]
        return await self._batch_upsert("markets", data, "id")

    # ─────────────────────────────────────────────────────────────────────────
    # READ OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    async def get_wallets_needing_enrichment(self, limit: int = 100) -> List[Dict[str, Any]]:
        result = (self.supabase.table("wallets")
                 .select("*")
                 .eq("enriched", False)
                 .order("created_at", desc=False)
                 .limit(limit)
                 .execute())
        return result.data or []

    async def get_existing_wallets_batch(
        self, wallet_addresses: List[str], staleness_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently checks presence and staleness of wallets.
        """
        if not wallet_addresses:
            return {}

        batch_size = 500
        result_map = {}
        now_ts = datetime.now(timezone.utc).timestamp()
        cutoff = now_ts - (staleness_hours * 3600)

        for i in range(0, len(wallet_addresses), batch_size):
            batch = wallet_addresses[i:i + batch_size]
            
            result = (
                self.supabase.table("wallets")
                .select("proxy_wallet, last_sync_at, updated_at")
                .in_("proxy_wallet", batch)
                .execute()
            )
            
            # Assertive read check
            assert result.data is not None, "Read operation returned None"

            existing = {row["proxy_wallet"]: row for row in result.data}

            for addr in batch:
                if addr in existing:
                    row = existing[addr]
                    # Determine staleness
                    last_sync = row.get("last_sync_at") or row.get("updated_at")
                    is_stale = True
                    
                    if last_sync:
                        try:
                            # Handle timestamp parsing strictly
                            ts = datetime.fromisoformat(last_sync.replace("Z", "+00:00")).timestamp()
                            is_stale = ts < cutoff
                        except ValueError:
                            pass # Defaults to true if unparseable

                    result_map[addr] = {
                        "exists": True,
                        "is_stale": is_stale,
                        "last_sync_at": last_sync
                    }
                else:
                    result_map[addr] = {
                        "exists": False,
                        "is_stale": True,
                        "last_sync_at": None
                    }

        return result_map

    async def get_market_count(self) -> int:
        result = self.supabase.table("markets").select("id", count="exact").head().execute()
        return result.count if result.count is not None else 0

    async def get_wallet_count(self) -> int:
        result = self.supabase.table("wallets").select("proxy_wallet", count="exact").head().execute()
        return result.count if result.count is not None else 0