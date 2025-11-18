"""Elite Wallet Enrichment & Historical Performance Orchestration Engine."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.functions.retrieve_wallets import RetrieveTradesWalletsCollector

logger = logging.getLogger(__name__) 

class WalletTracker:
    """
    High-performance tracker defined locally to avoid modifying external files.
    Features: Batch Metadata Fetching, Timestamp Cutoff, Parallel Enrichment.
    """

    def __init__(
        self,
        api: Optional[PolymarketDataAPI] = None,
        gamma: Optional[PolymarketGamma] = None,
    ):
        self.api = api or PolymarketDataAPI()
        self.gamma = gamma or PolymarketGamma()
        self._event_metadata_cache: Dict[str, Dict[str, Any]] = {}

    async def sync_wallet_closed_positions_with_enrichment(
        self,
        proxy_wallet: str,
        save_position_batch: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        save_event_batch: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        last_synced_timestamp: Optional[float] = 0
    ) -> Dict[str, Any]:
        """Optimized sync: fetch API pages, stop at timestamp cutoff, bulk fetch metadata, save in chunks."""
        logger.info(f"Syncing wallet {proxy_wallet}")
        all_positions, offset, events_to_resolve = [], 0, set()

        while True:
            batch = await self.api.get_closed_positions(user=proxy_wallet, limit=50, offset=offset)
            if not batch: break

            new_positions_in_batch = []
            for position in batch:
                pos_timestamp = position.get("timestamp", 0)
                if last_synced_timestamp and pos_timestamp <= last_synced_timestamp: break

                normalized = self._normalize_closed_position(position, proxy_wallet)
                new_positions_in_batch.append(normalized)
                if normalized.get("event_slug") and not normalized.get("event_id"):
                    events_to_resolve.add(normalized["event_slug"])

            if not new_positions_in_batch: break

            if events_to_resolve:
                await self._prefetch_event_metadata(list(events_to_resolve))

            for pos in new_positions_in_batch:
                self._apply_event_metadata_from_cache(pos)

            if save_position_batch:
                await save_position_batch(new_positions_in_batch)

            if save_event_batch and events_to_resolve:
                unique_events = [self._event_metadata_cache[f"slug:{slug}"] for slug in events_to_resolve if self._event_metadata_cache.get(f"slug:{slug}")]
                if unique_events:
                    await save_event_batch(unique_events)

            events_to_resolve.clear()
            all_positions.extend(new_positions_in_batch)

            if len(batch) < 50: break
            offset += 50

        event_ids = list(set(p.get("event_id") for p in all_positions if p.get("event_id")))
        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(all_positions),
            "total_volume": sum(p.get("total_bought", 0) for p in all_positions),
            "realized_pnl": sum(p.get("realized_pnl", 0) for p in all_positions),
            "event_ids": event_ids
        }

    async def _prefetch_event_metadata(self, slugs: List[str]):
        """Fetch metadata for multiple slugs in parallel."""
        missing_slugs = [s for s in slugs if f"slug:{s}" not in self._event_metadata_cache]
        if not missing_slugs: return

        async def fetch_one(slug):
            data = await self.gamma.get_event(slug)
            return self._process_and_cache_event(data, slug) if data else None

        for i in range(0, len(missing_slugs), 10):
            chunk = missing_slugs[i:i+10]
            await asyncio.gather(*[fetch_one(s) for s in chunk])

    def _process_and_cache_event(self, event_data: Dict, slug_key: str) -> Dict:
        """Computes metrics and updates cache synchronously."""
        metrics = self._compute_event_metrics(event_data)
        metadata = {
            "id": str(event_data.get("id")),
            "slug": event_data.get("slug", slug_key),
            "title": event_data.get("title"),
            "description": event_data.get("description"),
            "category": event_data.get("category"),
            "tags": self._extract_tag_labels(event_data.get("tags")),
            "status": "closed" if event_data.get("closed") else event_data.get("status", "active"),
            "start_date": event_data.get("startDate"),
            "end_date": event_data.get("endDate"),
            "market_count": metrics["market_count"],
            "total_liquidity": metrics["total_liquidity"],
            "total_volume": metrics["total_volume"],
            "raw_data": event_data,
        }
        self._event_metadata_cache[f"slug:{slug_key}"] = metadata
        if metadata["id"]:
            self._event_metadata_cache[metadata["id"]] = metadata
        return metadata

    def _apply_event_metadata_from_cache(self, position: Dict[str, Any]):
        """Fast in-memory lookup."""
        metadata = self._event_metadata_cache.get(f"slug:{position.get('event_slug')}")
        if not metadata and position.get("event_id"):
            metadata = self._event_metadata_cache.get(position.get("event_id"))
        if metadata:
            position.update({
                "event_id": metadata.get("id"),
                "event_slug": metadata.get("slug"),
                "event_category": metadata.get("category"),
                "event_tags": metadata.get("tags"),
            })

    def _normalize_closed_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize closed position data."""
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]
        event_category = position.get("category") or position.get("eventCategory")
        event_tags = self._extract_tag_labels(position.get("tags") or position.get("eventTags"))
        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "event_id": None,
            "condition_id": condition_id,
            "asset": position.get("asset"),
            "outcome": position.get("outcome"),
            "outcome_index": outcome_index,
            "total_bought": position.get("totalBought", 0),
            "avg_price": position.get("avgPrice", 0),
            "cur_price": position.get("curPrice", 0),
            "realized_pnl": position.get("realizedPnl", 0),
            "timestamp": position.get("timestamp", 0),
            "end_date": position.get("endDate"),
            "title": position.get("title"),
            "slug": position.get("slug"),
            "event_slug": position.get("eventSlug"),
            "event_category": event_category,
            "event_tags": event_tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": position
        }

    @staticmethod
    def _extract_tag_labels(raw_tags: Any) -> List[str]:
        if not raw_tags: return []
        if isinstance(raw_tags, str):
            try: return json.loads(raw_tags) if raw_tags.startswith("[") else [raw_tags]
            except: return [raw_tags]
        if isinstance(raw_tags, list):
            return [t if isinstance(t, str) else t.get("label", "") for t in raw_tags]
        return []

    @classmethod
    def _compute_event_metrics(cls, event: Dict) -> Dict:
        # Simplified metric extraction
        markets = event.get("markets", []) or []
        total_vol = float(event.get("volume") or 0)
        if not total_vol and markets:
            total_vol = sum(float(m.get("volume") or 0) for m in markets if isinstance(m, dict))
        return {
            "market_count": max(len(markets), int(event.get("marketCount", 0))),
            "total_liquidity": float(event.get("liquidity") or 0),
            "total_volume": total_vol,
        }


# ORCHESTRATOR

class PolymarketWalletCollector:
    """Elite wallet enrichment orchestration engine."""

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        # Use local Optimized Tracker
        self.wallet_tracker = WalletTracker(api=self.data_api, gamma=self.gamma)
        self.max_concurrent = 15
        # Track events already saved in this session to avoid duplicates
        self._saved_events_cache: Set[str] = set()
        # Use RetrieveTradesWalletsCollector for efficient wallet discovery
        self.retrieve_collector = RetrieveTradesWalletsCollector()

    async def enrich_wallets_positions(self, max_wallets: int = 50, skip_existing_positions: bool = True) -> Dict[str, Any]:
        """Optimized enrichment with parallel processing."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting enrichment for {max_wallets} wallets")

        wallets_data = await self.db.get_wallets_needing_enrichment(max_wallets)
        if not wallets_data:
            return {"wallets_processed": 0, "positions_synced": 0, "total_volume": 0.0, "total_pnl": 0.0, "duration_seconds": 0.0}

        wallet_addresses = [w["proxy_wallet"] for w in wallets_data if w.get("proxy_wallet")]
        logger.info(f"Found {len(wallet_addresses)} wallets to process")

        wallet_timestamps = await self._get_wallet_last_sync_timestamps(wallet_addresses)
        logger.info(f"Found timestamps for {len(wallet_timestamps)} wallets")

        valid_results = []
        for wallet_addr in wallet_addresses:
            logger.info(f"Processing wallet {len(valid_results) + 1}/{len(wallet_addresses)}: {wallet_addr[:10]}")
            last_ts = wallet_timestamps.get(wallet_addr, 0)

            async def save_event_batch_filtered(events: List[Dict]):
                new_events = [e for e in events if e.get("id") not in self._saved_events_cache]
                if not new_events: return
                await self._bulk_upsert("events_closed", new_events)
                for event in new_events:
                    if event.get("id"): self._saved_events_cache.add(event["id"])

            enrichment_result = await self.wallet_tracker.sync_wallet_closed_positions_with_enrichment(
                proxy_wallet=wallet_addr,
                save_position_batch=lambda positions: self._bulk_upsert("wallet_closed_positions", positions),
                save_event_batch=save_event_batch_filtered,
                last_synced_timestamp=last_ts
            )

            event_ids = enrichment_result.get("event_ids", [])
            if event_ids:
                discovered_wallets = await self._discover_wallets_from_events_during_enrichment(event_ids)
                logger.info(f"Wallet {wallet_addr[:10]} discovered {len(discovered_wallets)} additional wallets")
                enrichment_result["additional_wallets_discovered"] = len(discovered_wallets)
            else:
                enrichment_result["additional_wallets_discovered"] = 0

            await self._mark_wallet_enriched_single(wallet_addr)
            logger.info(f"Wallet {wallet_addr[:10]} marked as enriched")
            valid_results.append(enrichment_result)

        report = {
            "wallets_processed": len(valid_results),
            "positions_synced": sum(r.get("positions_fetched", 0) for r in valid_results),
            "total_volume": sum(r.get("total_volume", 0.0) for r in valid_results),
            "total_pnl": sum(r.get("realized_pnl", 0.0) for r in valid_results),
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
        }
        logger.info(f"Completed enrichment: {report['wallets_processed']} wallets in {report['duration_seconds']:.2f}s")
        return report

    async def enrich_unenriched_events(
        self,
        max_events: int = 10,
        is_closed: bool = False,
        skip_existing_positions: bool = True
    ) -> Dict[str, Any]:
        """Run the fast (trades) and slow (wallet enrichment) flows in sequence."""
        logger.info(f"Running full enrichment for {max_events} events")
        retriever = RetrieveTradesWalletsCollector()
        fast_report = await retriever.retrieve_wallets_from_events(max_events, is_closed)
        
        target_wallets = max(fast_report.get("wallets_discovered", 0), 1)
        slow_report = await self.enrich_wallets_positions(
            max_wallets=target_wallets,
            skip_existing_positions=skip_existing_positions
        )
        
        # Merge reports
        fast_report.update(slow_report)
        return fast_report

    # --- Helpers ---

    async def _bulk_upsert(self, table: str, rows: List[Dict[str, Any]]) -> None:
        """Bulk upsert with small batches."""
        if not rows: return

        batch_size = 100
        logger.info(f"Bulk upsert: {len(rows)} rows into {table}")

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            self.db.supabase.table(table).upsert(batch).execute()
            logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} rows) into {table}")
            await asyncio.sleep(0.2)

    async def _get_wallet_last_sync_timestamps(self, wallets: List[str]) -> Dict[str, float]:
        """Fetch MAX(timestamp) for wallets via RPC."""
        if not wallets: return {}
        try:
            response = self.db.supabase.rpc("get_wallets_max_timestamp", {"wallets": wallets}).execute()
            return {row['wallet']: float(row['max_ts']) for row in (response.data or []) if row.get('max_ts')}
        except Exception as e:
            # Fall back to 0 (fetch all positions) if RPC fails due to type mismatch
            logger.warning(f"RPC get_wallets_max_timestamp failed, falling back to 0: {e}")
            return {wallet: 0.0 for wallet in wallets}

    async def _discover_wallets_from_events_during_enrichment(self, event_ids: List[str]) -> Set[str]:
        """Discover additional wallets from events during enrichment."""
        if not event_ids: return set()

        discovered_wallets = set()
        logger.info(f"Discovering wallets from {len(event_ids)} events")

        async def discover_from_event(event_id: str) -> Set[str]:
            wallets = await self.retrieve_collector._discover_wallets_only(event_id)
            return wallets

        batch_size = 5
        for i in range(0, len(event_ids), batch_size):
            batch = event_ids[i:i+batch_size]
            batch_results = await asyncio.gather(*[discover_from_event(event_id) for event_id in batch])
            for result in batch_results:
                discovered_wallets.update(result)
            if i + batch_size < len(event_ids):
                await asyncio.sleep(0.1)

        if discovered_wallets:
            existing_wallets = await self.retrieve_collector._check_existing_wallets(list(discovered_wallets))
            new_wallets = discovered_wallets - existing_wallets

            if new_wallets:
                logger.info(f"Discovered {len(new_wallets)} new wallets")
                wallet_rows = [{
                    "proxy_wallet": wallet_addr,
                    "enriched": False,
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                } for wallet_addr in new_wallets]
                await self._bulk_upsert("wallets", wallet_rows)
                return new_wallets

        return set()

    async def _mark_wallet_enriched_single(self, wallet: str):
        """Mark a single wallet as enriched."""
        if not wallet: return
        now = datetime.now(timezone.utc).isoformat()
        self.db.supabase.table("wallets").update({
            "enriched": True,
            "enriched_at": now,
            "updated_at": now
        }).eq("proxy_wallet", wallet).execute()
        logger.debug(f"Marked wallet {wallet[:10]} as enriched")

    async def _run_wallet_metrics(self, wallets: List[str]) -> None:
        """Run wallet metrics calculation."""
        unique_wallets = list(set(wallets))
        if not unique_wallets: return
        self.db.supabase.rpc("run_wallet_metrics", {"p_wallets": unique_wallets}).execute()


# STANDALONE EXPORTS
async def enrich_unenriched_events(
    max_events: int = 10,
    is_closed: bool = False,
    skip_existing_positions: bool = True
) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_unenriched_events(max_events, is_closed, skip_existing_positions)

async def enrich_wallets_positions(
    max_wallets: int = 50,
    skip_existing_positions: bool = True
) -> Dict[str, Any]:
    collector = PolymarketWalletCollector()
    return await collector.enrich_wallets_positions(max_wallets, skip_existing_positions)