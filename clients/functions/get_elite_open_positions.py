"""
Open Positions Collector (Production Scale).
Enriches positions with Gamma Event Metadata.
Strictly enforces Postgres types (No empty strings for dates).
FILTERS: Ignores Tier 'D' traders (Only S, A, B, C).
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketDataAPI, PolymarketGamma

# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WalletOpenPositionsCollector:
    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.api = PolymarketDataAPI()
        self.gamma = PolymarketGamma()
        
        # Tuning
        self.API_CONCURRENCY = 5
        self.API_DELAY = 1.0
        self.DB_PAGE_SIZE = 1000
        self.DB_WRITE_BATCH = 200
        
        self._event_cache: Dict[str, Dict] = {}

    async def run(self, force_refresh: bool = False) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        await self._cleanup_dead_positions()

        stats = {"processed": 0, "saved": 0, "errors": 0}
        offset = 0

        logger.info("🚀 Starting Enriched Position Collection (Tiers S, A, B, C only)...")

        while True:
            traders_page = await self._fetch_trader_batch(limit=self.DB_PAGE_SIZE, offset=offset)
            if not traders_page:
                break 

            logger.info(f"   📂 Processing DB Page: {offset} - {offset + len(traders_page)}")
            
            batch_stats = await self._process_trader_page(traders_page)
            
            stats["processed"] += len(traders_page)
            stats["saved"] += batch_stats["saved"]
            stats["errors"] += batch_stats["errors"]
            
            offset += self.DB_PAGE_SIZE

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        stats["duration"] = duration
        stats["rate_per_sec"] = round(stats["processed"] / duration, 2) if duration > 0 else 0
        
        logger.info(f"✅ Collection Complete: {stats}")
        return stats

    async def _process_trader_page(self, traders: List[Dict]) -> Dict[str, int]:
        all_raw_positions = []
        errors = 0

        chunk_size = self.API_CONCURRENCY
        for i in range(0, len(traders), chunk_size):
            batch = traders[i : i + chunk_size]
            tasks = [self._worker_fetch_positions(t) for t in batch]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                if res is not None:
                    all_valid = [p for p in res if self._is_valid_pre_check(p)]
                    for p in all_valid:
                        owner = next((t for t in batch if t['proxy_wallet'] == p.get('proxyWallet')), {})
                        p['_trader_meta'] = owner
                    all_raw_positions.extend(all_valid)
                else:
                    errors += 1
            
            await asyncio.sleep(self.API_DELAY)

        # Enrich
        unique_slugs = set()
        for p in all_raw_positions:
            slug = p.get('eventSlug')
            if slug and slug not in self._event_cache:
                unique_slugs.add(slug)
        
        if unique_slugs:
            logger.info(f"   🌍 Enriching {len(unique_slugs)} new events...")
            await self._fetch_event_metadata(list(unique_slugs))

        # Transform & Write
        final_write_queue = []
        for p in all_raw_positions:
            if self._is_valid_post_check(p):
                final_write_queue.append(self._transform(p))

        saved_count = await self._bulk_write_to_db(final_write_queue)
        
        return {"saved": saved_count, "errors": errors}

    async def _fetch_event_metadata(self, slugs: List[str]):
        semaphore = asyncio.Semaphore(10) 
        
        async def fetch_one(slug):
            async with semaphore:
                try:
                    event = await self.gamma.get_event(slug)
                    if event:
                        cat = event.get('category')
                        if not cat:
                             tags = event.get('tags', [])
                             if tags and isinstance(tags[0], dict):
                                 cat = tags[0].get('label')
                             elif tags:
                                 cat = str(tags[0])

                        self._event_cache[slug] = {
                            "endDate": event.get("endDate"),
                            "category": cat,
                            "id": event.get("id")
                        }
                except Exception:
                    pass

        tasks = [fetch_one(s) for s in slugs]
        await asyncio.gather(*tasks)

    async def _worker_fetch_positions(self, trader: Dict) -> Optional[List[Dict]]:
        wallet = trader['proxy_wallet']
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = await self.api.get_positions(wallet, limit=500)
                return res if res else []
            except Exception as e:
                if "429" in str(e):
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                return None
        return None

    # --- VALIDATION ---

    def _is_valid_pre_check(self, p: Dict) -> bool:
        if not p.get("conditionId"): return False
        if p.get("redeemable") is True: return False
        cur_price = float(p.get("curPrice") or 0)
        if cur_price <= 0.02 or cur_price >= 0.98: return False
        return True

    def _is_valid_post_check(self, p: Dict) -> bool:
        slug = p.get("eventSlug")
        meta = self._event_cache.get(slug, {})
        
        # Date Check
        end_date = self._parse_iso_date(meta.get("endDate") or p.get("endDate"))
        
        if end_date:
            if end_date < datetime.now(timezone.utc):
                return False # Expired
        
        return True

    def _transform(self, p: Dict) -> Dict:
        wallet = p['_trader_meta']['proxy_wallet']
        meta = p['_trader_meta']
        
        slug = p.get("eventSlug")
        event_meta = self._event_cache.get(slug, {})
        
        # Date Parsing
        raw_date = event_meta.get("endDate") or p.get("endDate")
        final_end_date = None
        if raw_date:
             parsed = self._parse_iso_date(raw_date)
             if parsed:
                 final_end_date = parsed.isoformat()

        final_event_id = event_meta.get("id") or p.get("eventId")
        category = event_meta.get("category")

        size = float(p.get("size", 0))
        cur_price = float(p.get("curPrice", 0))
        entry_price = float(p.get("avgPrice", 0))
        
        pid = hashlib.sha256(f"{wallet}{p['conditionId']}{p['outcomeIndex']}".encode()).hexdigest()[:32]
        now = datetime.now(timezone.utc).isoformat()

        return {
            "id": pid,
            "proxy_wallet": wallet,
            "event_id": final_event_id,
            "condition_id": p["conditionId"],
            "asset": p.get("asset"),
            "outcome": p.get("outcome"),
            "outcome_index": int(p.get("outcomeIndex", 0)),
            
            "size": size,
            "avg_entry_price": entry_price,
            "current_price": cur_price,
            "unrealized_pnl": size * (cur_price - entry_price),
            "position_value": size * cur_price,
            "cash_pnl": float(p.get("cashPnl", 0)),
            "initial_value": float(p.get("initialValue", 0)),
            
            "title": p.get("title"),
            "slug": p.get("slug"),
            "event_slug": slug,
            
            "event_end_date": final_end_date,
            "event_category": category,
            
            "redeemable": bool(p.get("redeemable")),
            "mergeable": bool(p.get("mergeable")),
            "negative_risk": bool(p.get("negativeRisk")),
            "raw_data": p,
            
            "wallet_rank": meta.get("rank_in_tier"),
            "composite_score": meta.get("composite_score"),
            "trader_tier": meta.get("tier"),
            "win_rate": meta.get("win_rate"),
            "roi": meta.get("roi"),
            
            "created_at": now,
            "updated_at": now
        }

    def _parse_iso_date(self, date_str: str) -> Optional[datetime]:
        if not date_str: return None
        try:
            clean_str = date_str.replace("Z", "+00:00")
            if "T" not in clean_str and "+" not in clean_str:
                 return datetime.fromisoformat(clean_str).replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(clean_str)
        except:
            return None

    async def _bulk_write_to_db(self, positions: List[Dict]) -> int:
        if not positions: return 0
        total_saved = 0
        
        # Cleanup nesting before write
        for p in positions:
             if '_trader_meta' in p: del p['_trader_meta']
             if '_trader_meta' in p['raw_data']: del p['raw_data']['_trader_meta']

        for i in range(0, len(positions), self.DB_WRITE_BATCH):
            chunk = positions[i : i + self.DB_WRITE_BATCH]
            try:
                self.db.supabase.table("elite_open_positions").upsert(chunk).execute()
                total_saved += len(chunk)
            except Exception as e:
                logger.error(f"❌ DB Write Error: {e}")
        return total_saved

    async def _fetch_trader_batch(self, limit: int, offset: int) -> List[Dict]:
        # UPDATED: Added .neq("tier", "D") to filter out D traders
        res = (self.db.supabase.table("elite_traders")
               .select("*")
               .neq("tier", "D")
               .order("composite_score", desc=True)
               .range(offset, offset + limit - 1).execute())
        return res.data or []

    async def _cleanup_dead_positions(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        self.db.supabase.table("elite_open_positions").delete().lt("event_end_date", today).execute()
        self.db.supabase.table("elite_open_positions").delete().or_("current_price.gte.0.98,current_price.lte.0.02").execute()
        self.db.supabase.table("elite_open_positions").delete().eq("redeemable", True).execute()

if __name__ == "__main__":
    asyncio.run(WalletOpenPositionsCollector().run())