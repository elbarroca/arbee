import asyncio
import logging
import json

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from config.settings import Settings
from clients.polymarket import PolymarketGamma, PolymarketDataAPI, _parse_iso_datetime

logger = logging.getLogger(__name__)

class PolymarketDataCollector:
    """
    Elite Polymarket Data Engine.
    Protocol: Hygiene -> Intelligence -> Deployment -> Resolution.
    """

    def __init__(self):
        from database.client import MarketDatabase
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.max_concurrent = 20

    async def collect_new_events_and_markets(self, limit: Optional[int] = None) -> Dict[str, Any]:
        logger.info("🚀 ELITE PROTOCOL INITIATED")

        # 1. HYGIENE (Archive outdated intelligence first)
        logger.info("🧹 Phase 1: Database Hygiene")
        mkts_moved, evts_moved = await asyncio.gather(
            self._archive_expired_items("markets", "close_date"),
            self._archive_expired_items("events", "end_date")
        )

        # 2. INTELLIGENCE (Acquire fresh data)
        logger.info("📡 Phase 2: Intelligence Acquisition")
        raw_intel = await self._fetch_open_events(limit)
        new_events = await self._filter_new_events(raw_intel)
        
        # 3. DEPLOYMENT (Write to DB)
        logger.info(f"💎 Phase 3: Deployment ({len(new_events)} new events)")
        e_saved = await self._deploy_events(new_events)
        m_saved = await self._deploy_markets(new_events) # Assumes new events imply new markets

        # 4. RESOLUTION (Populate outcomes)
        logger.info("🏁 Phase 4: Outcome Resolution")
        outcomes_filled = await self._populate_missing_outcomes(limit=100)

        return {
            "archived": {"events": evts_moved, "markets": mkts_moved},
            "deployed": {"events": e_saved, "markets": m_saved},
            "resolved": outcomes_filled
        }

    # --- CORE LOGIC ---

    async def _archive_expired_items(self, table: str, date_field: str) -> int:
        """Moves items to _closed if Date < Now OR Status == 'closed'."""
        now = datetime.now(timezone.utc).isoformat()
        # Assertive Filter: Expired OR Explicitly Closed
        or_filter = f"{date_field}.lt.{now},status.eq.closed"
        
        items = self.db.supabase.table(table).select("*").or_(or_filter).execute().data
        if not items: return 0

        async def transition(item: Dict):
            # 1. Prepare Payload
            closed_data = {**item, "status": "closed", "closed_at": now}
            
            # 2. Resolve Outcome (Markets Only)
            if table == "markets":
                outcome = await self._resolve_outcome(item)
                if outcome: closed_data["outcome"] = outcome

            # 3. Atomic Upsert -> Delete
            self.db.supabase.table(f"{table}_closed").upsert(closed_data).execute()
            self.db.supabase.table(table).delete().eq("id", item["id"]).execute()

        # Execute in parallel batches
        await self._batch_process(items, transition, batch_size=10)
        return len(items)

    async def _deploy_events(self, events: List[Dict]) -> int:
        if not events: return 0
        payloads = [self._normalize_event(e) for e in events]
        # Remove invalid IDs
        valid = [p for p in payloads if p.get("id")]
        
        async def save(batch):
            self.db.supabase.table("events").upsert(batch).execute()
            
        await self._batch_process(valid, save, batch_size=100, is_bulk=True)
        return len(valid)

    async def _deploy_markets(self, events: List[Dict]) -> int:
        if not events: return 0
        
        # Flatten markets from events
        all_markets = []
        for e in events:
            markets = e.get("markets", [])
            for m in markets:
                # Optimization: Enrich only if high value (skipping fetch for speed here, can rely on background)
                all_markets.append(self._normalize_market(m, e))

        async def save(batch):
            self.db.supabase.table("markets").upsert(batch).execute()

        await self._batch_process(all_markets, save, batch_size=50, is_bulk=True)
        return len(all_markets)

    async def _populate_missing_outcomes(self, limit: int = 50) -> int:
        missing = self.db.supabase.table("markets_closed").select("id").is_("outcome", "null").limit(limit).execute().data
        if not missing: return 0

        async def resolve(row):
            outcome = await self._fetch_outcome_api(row["id"])
            if outcome:
                self.db.supabase.table("markets_closed").update({"outcome": outcome}).eq("id", row["id"]).execute()
                return 1
            return 0

        results = await self._batch_process(missing, resolve, batch_size=10)
        return sum(results)

    # --- DATA FETCHING & UTILS ---

    async def _fetch_open_events(self, limit: Optional[int]) -> List[Dict]:
        return await self.gamma.get_all_events(active=True, page_size=500, max_events=limit)

    async def _filter_new_events(self, events: List[Dict]) -> List[Dict]:
        existing = self.db.supabase.table("events").select("id").execute()
        ids = {row['id'] for row in existing.data}
        return [e for e in events if str(e.get('id')) not in ids]

    async def _resolve_outcome(self, item: Dict) -> Optional[str]:
        """Extracts outcome from item or fetches from API."""
        raw = item.get("raw_data", {})
        return raw.get("outcome") or raw.get("winner") or await self._fetch_outcome_api(item["id"])

    async def _fetch_outcome_api(self, market_id: str) -> Optional[str]:
        # Search API for closed market data
        try:
            res = await self.gamma.get_markets(closed="true", condition_id=market_id, limit=1)
            if res:
                m = res[0]
                if m.get("outcome"): return str(m["outcome"])
                # Fallback: check prices
                prices = m.get("outcomePrices", [])
                outcomes = json.loads(m.get("outcomes")) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
                for i, p in enumerate(prices):
                    if float(p) >= 0.99 and i < len(outcomes): return str(outcomes[i])
        except Exception:
            pass
        return None

    async def _batch_process(self, items: List, func, batch_size: int, is_bulk: bool = False) -> List:
        """High-performance batch executor."""
        results = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i+batch_size]
            if is_bulk:
                # Pass the whole chunk to the function
                await func(chunk)
            else:
                # Run individual items in parallel
                res = await asyncio.gather(*[func(item) for item in chunk])
                results.extend(res)
        return results

    # --- NORMALIZATION (PURE) ---

    def _serialize(self, obj: Any) -> Any:
        """Recursively cleans data for JSON."""
        if isinstance(obj, dict): return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._serialize(i) for i in obj]
        if isinstance(obj, datetime): return obj.isoformat()
        return obj

    def _normalize_event(self, e: Dict) -> Dict:
        ts = lambda x: _parse_iso_datetime(x).isoformat() if x else None
        tags = e.get("tags", [])
        return {
            "id": str(e.get("id")),
            "title": e.get("title"),
            "description": e.get("description"),
            "category": e.get("category"),
            "status": "active" if e.get("active") else "closed",
            "start_date": ts(e.get("startDate")),
            "end_date": ts(e.get("endDate")),
            "market_count": len(e.get("markets", [])),
            "total_liquidity": float(e.get("liquidity") or 0),
            "tags": [t.get("label") if isinstance(t, dict) else t for t in tags],
            "raw_data": self._serialize(e),
            "platform": "polymarket"
        }

    def _normalize_market(self, m: Dict, e: Dict) -> Dict:
        ts = lambda x: _parse_iso_datetime(x).isoformat() if x else None
        prices = m.get("outcomePrices", []) or ["0", "0"]
        return {
            "id": str(m.get("id")),
            "event_id": str(e.get("id")),
            "title": m.get("question"),
            "description": m.get("description"),
            "category": m.get("category") or e.get("category"),
            "status": "active" if m.get("active") else "closed",
            "close_date": ts(m.get("endDate")),
            "p_yes": float(prices[1]) if len(prices) > 1 else 0,
            "p_no": float(prices[0]) if len(prices) > 0 else 0,
            "liquidity": float(m.get("liquidity") or 0),
            "total_volume": float(m.get("volume") or 0),
            "raw_data": self._serialize(m),
            "platform": "polymarket"
        }