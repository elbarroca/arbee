"""Polymarket Wallet Tracker & Copy-Trading Analysis Client."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from clients.polymarket import PolymarketDataAPI, PolymarketGamma

logger = logging.getLogger(__name__)


class WalletTracker:
    """High-level wallet tracking and copy-trading analysis orchestrator."""

    def __init__(
        self,
        api: Optional[PolymarketDataAPI] = None,
        gamma: Optional[PolymarketGamma] = None,
        min_volume: float = 10000.0,
        min_markets: int = 20,
        min_win_rate: float = 0.40
    ):
        self.api = api or PolymarketDataAPI()
        self.gamma = gamma or PolymarketGamma()
        self.min_volume = min_volume
        self.min_markets = min_markets
        self.min_win_rate = min_win_rate
        self._event_metadata_cache: Dict[str, Dict[str, Any]] = {}

    async def sync_wallet_closed_positions_with_enrichment(
        self,
        proxy_wallet: str,
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Sync closed positions with automatic event_id enrichment and early termination."""
        assert proxy_wallet, "proxy_wallet required"
        logger.info("Slow path | wallet=%s | starting closed-position sync", proxy_wallet)

        positions = []
        offset = 0
        seen_position_ids: Set[str] = set()  # Track seen positions for early termination
        POSITION_DUPLICATE_THRESHOLD = 1.0  # 100% duplicates = stop

        while True:
            logger.debug(f"API: get_closed_positions(user={proxy_wallet[:12]}..., offset={offset})")
            batch = await self.api.get_closed_positions(
                user=proxy_wallet, event_id=event_ids, limit=50, offset=offset
            )
            if not batch:
                break
            logger.debug(f"API: Received {len(batch)} positions")

            # Check for duplicates in this batch
            batch_duplicate_count = 0
            new_positions_in_batch = []
            
            for position in batch:
                normalized = self._normalize_closed_position(position, proxy_wallet)
                
                # Create position ID: proxy_wallet + condition_id + outcome_index
                condition_id = normalized.get("condition_id") or ""
                outcome_index = normalized.get("outcome_index", 0)
                position_id = f"{proxy_wallet}_{condition_id}_{outcome_index}"
                
                # Check if we've seen this position before
                if position_id in seen_position_ids:
                    batch_duplicate_count += 1
                else:
                    seen_position_ids.add(position_id)
                    new_positions_in_batch.append(normalized)

            # Early termination: if batch is 100% duplicates, stop fetching
            if len(batch) > 0 and batch_duplicate_count == len(batch):
                logger.info(f"Early termination: Batch at offset {offset} is 100% duplicates ({batch_duplicate_count}/{len(batch)}), stopping fetch")
                break

            # Process new positions
            for normalized in new_positions_in_batch:
                await self._apply_event_metadata(normalized, save_event)

                if save_position:
                    await save_position(normalized)
                positions.append(normalized)

            if len(batch) < 50:
                break
            offset += 50

        total_volume = sum(p.get("total_bought", 0) for p in positions)
        realized_pnl = sum(p.get("realized_pnl", 0) for p in positions)
        enriched_count = sum(1 for p in positions if p.get("event_id"))

        logger.info(
            "Slow path | wallet=%s | positions=%d enriched=%d volume=%.2f pnl=%.2f",
            proxy_wallet,
            len(positions),
            enriched_count,
            total_volume,
            realized_pnl,
        )

        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(positions),
            "positions_enriched": enriched_count,
            "total_volume": total_volume,
            "realized_pnl": realized_pnl
        }

    async def sync_wallets_closed_positions_batch(
        self,
        proxy_wallets: List[str],
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_concurrency: int = 5,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """Sync closed positions for multiple wallets concurrently."""
        logger.info(f"Batch sync: {len(proxy_wallets)} wallets, concurrency={max_concurrency}")
        if not proxy_wallets:
            return {
                "wallets_processed": 0,
                "wallets_with_positions": 0,
                "total_positions": 0,
                "total_enriched": 0,
                "total_volume": 0.0,
                "total_pnl": 0.0,
                "results": {}
            }

        semaphore = asyncio.Semaphore(max_concurrency)
        total_wallets = len(proxy_wallets)

        async def process_wallet(wallet: str, idx: int) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                if progress_callback:
                    progress_callback(wallet, idx, total_wallets)
                logger.info("Slow path | [%d/%d] syncing wallet=%s", idx, total_wallets, wallet)

                result = await self.sync_wallet_closed_positions_with_enrichment(
                    proxy_wallet=wallet,
                    save_position=save_position,
                    save_event=save_event
                )
                logger.info(
                    "Slow path | wallet=%s complete | positions=%d enriched=%d",
                    wallet,
                    result.get("positions_fetched", 0),
                    result.get("positions_enriched", 0),
                )
                return wallet, result

        tasks = [process_wallet(wallet, idx) for idx, wallet in enumerate(proxy_wallets, 1)]
        results = await asyncio.gather(*tasks)

        aggregated = {
            "wallets_processed": 0,
            "wallets_with_positions": 0,
            "total_positions": 0,
            "total_enriched": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "results": {}
        }

        for wallet, wallet_result in results:
            aggregated["wallets_processed"] += 1
            aggregated["results"][wallet] = wallet_result

            if wallet_result.get("positions_fetched", 0) > 0:
                aggregated["wallets_with_positions"] += 1
                aggregated["total_positions"] += wallet_result["positions_fetched"]
                aggregated["total_enriched"] += wallet_result.get("positions_enriched", 0)
                aggregated["total_volume"] += wallet_result.get("total_volume", 0.0)
                aggregated["total_pnl"] += wallet_result.get("realized_pnl", 0.0)

        logger.info(f"Batch complete: {aggregated['wallets_with_positions']}/{aggregated['wallets_processed']} wallets with positions, {aggregated['total_positions']} total positions")
        return aggregated

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        """Return value coerced into a list (for Gamma payloads)."""
        return value if isinstance(value, list) else [value] if value is not None else []

    @staticmethod
    def _to_float(value: Any) -> float:
        """Safely convert Gamma numeric fields into floats."""
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _compute_event_metrics(cls, event: Dict[str, Any]) -> Dict[str, float]:
        """Derive market count, total liquidity, and total volume for an event."""
        markets = [m for m in cls._ensure_list(event.get("markets", [])) if isinstance(m, dict)]
        market_count = max(len(markets), int(event.get("marketCount", 0)))

        # Extract liquidity from markets, event, or series (prioritized)
        total_liquidity = sum(cls._to_float(m.get("liquidityNum") or m.get("liquidityClob") or m.get("liquidity") or 0)
                            for m in markets) or cls._to_float(event.get("liquidity")) or \
                         next((cls._to_float(s.get("liquidity", 0))
                              for s in cls._ensure_list(event.get("series", []))
                              if isinstance(s, dict) and cls._to_float(s.get("liquidity", 0)) > 0), 0.0)

        # Extract volume from event or sum of markets
        total_volume = cls._to_float(event.get("volume")) or \
                      sum(cls._to_float(m.get("volumeNum") or m.get("volumeClob") or m.get("volume_total") or m.get("volume") or 0)
                          for m in markets)

        return {
            "market_count": market_count,
            "total_liquidity": total_liquidity,
            "total_volume": total_volume,
        }

    def _normalize_wallet_from_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Extract wallet profile from trade data."""
        timestamp = trade.get("timestamp", 0)
        timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else None
        now_dt = datetime.now(timezone.utc).isoformat()
        return {
            "proxy_wallet": trade.get("proxyWallet"),
            "enriched": False,
            "name": trade.get("name", ""),
            "pseudonym": trade.get("pseudonym", ""),
            "bio": trade.get("bio", ""),
            "profile_image": trade.get("profileImage", ""),
            "profile_image_optimized": trade.get("profileImageOptimized", ""),
            "display_username_public": trade.get("displayUsernamePublic", False),
            "first_seen_at": timestamp_dt.isoformat() if timestamp_dt else None,
            "last_seen_at": timestamp_dt.isoformat() if timestamp_dt else None,
            "last_sync_at": now_dt,
            "total_trades": 0,
            "total_markets": 0,
            "total_volume": 0.0,
            "created_at": now_dt,
            "updated_at": now_dt,
            "raw_data": trade
        }

    def _normalize_trade(self, trade: Dict[str, Any], event_id: Optional[str] = None) -> Dict[str, Any]:
        """Normalize trade data with assertive ID generation."""
        trade_id = trade.get("id") or trade.get("tradeId") or hashlib.sha256(
            f"{trade.get('transactionHash', '')}{trade.get('asset', '')}{trade.get('conditionId', '')}{trade.get('outcomeIndex', 0)}{trade.get('timestamp', 0)}".encode()
        ).hexdigest()[:32]

        size, price = trade.get("size", 0), trade.get("price", 0)
        return {
            "id": str(trade_id),
            "proxy_wallet": trade.get("proxyWallet"),
            "event_id": event_id,
            "condition_id": trade.get("conditionId"),
            "side": trade.get("side"),
            "asset": trade.get("asset"),
            "size": size,
            "price": price,
            "notional": size * price,
            "timestamp": trade.get("timestamp", 0),
            "transaction_hash": trade.get("transactionHash"),
            "title": trade.get("title"),
            "slug": trade.get("slug"),
            "event_slug": trade.get("eventSlug"),
            "outcome": trade.get("outcome"),
            "outcome_index": trade.get("outcomeIndex"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": trade
        }

    def _normalize_closed_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize closed position data with improved event metadata extraction."""
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]

        # Extract event metadata with multiple fallback strategies
        event_category = position.get("category") or position.get("eventCategory")
        event_tags = self._extract_tag_labels(position.get("tags") or position.get("eventTags"))

        # Try to extract event_slug from various possible fields
        event_slug = (position.get("eventSlug") or
                     position.get("event_slug") or
                     position.get("slug"))  # fallback to market slug if no event slug

        # Try to derive event_id from available data
        event_id = None
        if event_slug:
            # Check if we have cached metadata for this slug
            cached_metadata = self._event_metadata_cache.get(f"slug:{event_slug}")
            if cached_metadata:
                event_id = cached_metadata.get("id")

        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "event_id": event_id,  # Now populated if available from cache
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
            "event_slug": event_slug,  # Improved extraction
            "event_category": event_category,
            "event_tags": event_tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": position
        }

    async def _apply_event_metadata(
        self,
        position: Dict[str, Any],
        save_event: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        metadata = await self._get_event_metadata(position.get("event_id"), position.get("event_slug"))
        if not metadata:
            return

        position.update({
            k: v for k, v in {
                "event_id": metadata.get("id"),
                "event_slug": metadata.get("slug"),
                "event_category": metadata.get("category") if not position.get("event_category") else None,
                "event_tags": metadata.get("tags") if not position.get("event_tags") else None,
            }.items() if v is not None
        })

        logger.info("Slow path | wallet=%s | enriched event metadata id=%s slug=%s",
                   position.get("proxy_wallet"), position.get("event_id"), position.get("event_slug"))

        if save_event:
            await save_event(metadata)

    async def _get_event_metadata(
        self,
        event_id: Optional[str],
        event_slug: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        # Check cache first
        cache_keys = [k for k in [event_id, f"slug:{event_slug}"] if k]
        for key in cache_keys:
            if key in self._event_metadata_cache:
                return self._event_metadata_cache[key]

        # Fetch event data
        event = await self.gamma.get_event(event_slug) if event_slug else \
               (await self.gamma.get_events(limit=1, active=False, id=event_id))[0] if event_id else None

        if not event:
            return None

        metrics = self._compute_event_metrics(event)
        metadata = {
            "id": str(event.get("id", event_id)),
            "slug": event.get("slug", event_slug),
            "title": event.get("title"),
            "description": event.get("description"),
            "category": event.get("category"),
            "tags": self._extract_tag_labels(event.get("tags")),
            "status": "closed" if event.get("closed") else event.get("status", "active"),
            "start_date": event.get("startDate") or event.get("start_time"),
            "end_date": event.get("endDate"),
            "market_count": metrics["market_count"],
            "total_liquidity": metrics["total_liquidity"],
            "total_volume": metrics["total_volume"],
            "platform": "polymarket",
            "raw": event,
        }

        # Cache metadata
        for key in [metadata["id"], metadata["slug"], f"slug:{metadata['slug']}"]:
            if key:
                self._event_metadata_cache[key] = metadata

        return metadata

    @staticmethod
    def _extract_tag_labels(raw_tags: Any) -> List[str]:
        """Normalize tag payloads into a flat list of strings."""
        if not raw_tags:
            return []

        if isinstance(raw_tags, str):
            text = raw_tags.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    items = json.loads(text)
                except json.JSONDecodeError:
                    return []
            else:
                return [text]
        elif isinstance(raw_tags, list):
            items = raw_tags
        else:
            return []

        return [item if isinstance(item, str) else
               (item.get("label") or item.get("slug") or item.get("name") or "")
               for item in items if item and (item if isinstance(item, str) else
               (item.get("label") or item.get("slug") or item.get("name")))]
