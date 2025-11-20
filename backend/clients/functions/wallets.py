import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from config.settings import Settings
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.functions.retrieve_wallets import RetrieveTradesWalletsCollector

logger = logging.getLogger(__name__)

class PolymarketWalletCollector:
    def __init__(self):
        from database.client import MarketDatabase
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.api = PolymarketDataAPI()
        self.gamma = PolymarketGamma()
        self.retrieve_collector = RetrieveTradesWalletsCollector()
        self._event_cache: Dict[str, Dict] = {} # Memory cache for event metadata

    # --- 1. DISCOVERY ENDPOINT LOGIC ---
    async def enrich_unenriched_events(self, max_events: int = 50) -> Dict[str, Any]:
        """
        Discovers NEW wallets from ACTIVE events.
        Ignores closed events as requested.
        """
        logger.info(f"🔎 Starting Discovery Phase (Max Events: {max_events})")
        
        # Force is_closed=False to only look at active markets
        report = await self.retrieve_collector.retrieve_wallets_from_events(
            max_events=max_events, 
            is_closed=False
        )
        
        logger.info(f"✅ Discovery Complete: {report}")
        return report

    # --- 2. ENRICHMENT ENDPOINT LOGIC ---
    async def enrich_wallets_positions(self, max_wallets: int = 50) -> Dict[str, Any]:
        """
        Deep Scans existing wallets in the DB that have enriched=False.
        Priority: Not Enriched -> High Volume (if available).
        """
        logger.info(f"💎 Starting Enrichment Phase (Target: {max_wallets} Wallets)")
        start = datetime.now(timezone.utc)

        # A. Fetch Candidates
        # Only fetch wallets that are NOT enriched yet
        res = self.db.supabase.table("wallets")\
            .select("proxy_wallet")\
            .eq("enriched", False)\
            .limit(max_wallets)\
            .execute()
            
        candidates = [r['proxy_wallet'] for r in res.data]
        if not candidates:
            logger.info("💤 No unenriched wallets found.")
            return {"processed": 0}

        logger.info(f"🎯 Processing {len(candidates)} wallets...")

        # B. Process Loop
        stats = {"processed": 0, "positions": 0, "events_cached": 0}
        
        for wallet in candidates:
            try:
                # Sync Logic
                pos_count = await self._sync_single_wallet(wallet)
                stats["processed"] += 1
                stats["positions"] += pos_count
                
                # Mark Done
                self.db.supabase.table("wallets").update({
                    "enriched": True, 
                    "enriched_at": datetime.now(timezone.utc).isoformat()
                }).eq("proxy_wallet", wallet).execute()
                
            except Exception as e:
                logger.error(f"❌ Failed to enrich {wallet}: {e}")
                # Don't crash, just skip to next

        stats["duration"] = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(f"✅ Enrichment Batch Complete: {stats}")
        return stats

    async def _sync_single_wallet(self, wallet: str) -> int:
        """
        Fetches ALL closed positions for a wallet, enriches with event data, saves to DB.
        """
        all_positions = []
        offset = 0
        
        while True:
            # API Call
            batch = await self.api.get_closed_positions(user=wallet, limit=50, offset=offset)
            if not batch: break
            
            # Process Batch
            for raw_pos in batch:
                norm_pos = self._normalize_position(raw_pos, wallet)
                
                # Enrich Event Data (Critical Step)
                slug = norm_pos['event_slug']
                if slug:
                    meta = await self._get_event_metadata(slug)
                    if meta:
                        norm_pos['event_id'] = meta['id']
                        norm_pos['event_category'] = meta['category']
                        norm_pos['event_tags'] = meta['tags']
                
                all_positions.append(norm_pos)

            if len(batch) < 50: break
            offset += 50

        # Save to DB
        if all_positions:
            await self._save_positions(all_positions)
            
        return len(all_positions)

    async def _get_event_metadata(self, slug: str) -> Optional[Dict]:
        """Cached metadata fetcher to reduce API load."""
        if slug in self._event_cache:
            return self._event_cache[slug]

        try:
            event = await self.gamma.get_event(slug)
            if event:
                # Minimal metadata needed for analysis
                meta = {
                    "id": event.get("id"),
                    "category": event.get("category"),
                    "tags": event.get("tags", [])
                }
                self._event_cache[slug] = meta
                return meta
        except Exception:
            pass # Metadata fail shouldn't stop position save
        return None

    def _normalize_position(self, p: Dict, wallet: str) -> Dict:
        # (Your provided normalization logic, stripped for brevity but functionality preserved)
        cond_id = p.get("conditionId", "")
        idx = p.get("outcomeIndex", 0)
        pid = hashlib.sha256(f"{wallet}{cond_id}{idx}".encode()).hexdigest()[:32]
        
        return {
            "id": pid,
            "proxy_wallet": wallet,
            "condition_id": cond_id,
            "asset": p.get("asset"),
            "outcome": p.get("outcome"),
            "outcome_index": idx,
            "total_bought": float(p.get("totalBought") or 0),
            "realized_pnl": float(p.get("realizedPnl") or 0),
            "event_slug": p.get("eventSlug") or p.get("slug"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    async def _save_positions(self, positions: List[Dict]):
        # Bulk upsert 
        for i in range(0, len(positions), 100):
            chunk = positions[i:i+100]
            try:
                self.db.supabase.table("wallet_closed_positions").upsert(chunk).execute()
            except Exception as e:
                logger.error(f"DB Save Error: {e}")

# --- Standalone Exports for API ---

async def enrich_unenriched_events(max_events: int = 50) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_unenriched_events(max_events)

async def enrich_wallets_positions(max_wallets: int = 50) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_wallets_positions(max_wallets)