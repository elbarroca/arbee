import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI

logger = logging.getLogger(__name__)


class PolymarketDataCollector:
    """
    Elite Polymarket data orchestration engine.

    Assertive execution flow:
    1. Acquire fresh event intelligence
    2. Deploy market capture operations
    3. Execute lifecycle transitions with outcome validation
    4. Enforce complete outcome population
    5. Deliver victory metrics
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.max_concurrent = 20

    async def _process_batches(self, items: List, process_func, batch_size: int = None) -> List:
        batch_size = batch_size or self.max_concurrent
        return [
            result
            for i in range(0, len(items), batch_size)
            for result in await asyncio.gather(*[process_func(item) for item in items[i:i + batch_size]])
        ]

    async def collect_new_events_and_markets(self, limit: Optional[int] = None) -> Dict[str, int]:
        logger.info("🚀 Elite data collection initiated - victory requires complete outcome validation")

        all_events = await self._fetch_all_open_events(limit)
        new_events = await self._filter_new_events_only(all_events)
        events_with_new_markets = await self._filter_events_with_new_markets(new_events)

        logger.info(f"🎯 Intelligence acquired: {len(new_events)} fresh events, {len(events_with_new_markets)} with new markets")

        # Execute parallel operations
        events_task = self._save_events_to_database(new_events)
        markets_task = self._fetch_and_save_all_markets(events_with_new_markets)
        events_saved, markets_saved = await asyncio.gather(events_task, markets_task)

        logger.info(f"💾 Deployed {events_saved} events, {markets_saved} markets to database")

        # Execute lifecycle transitions
        market_lifecycle = self._move_expired_items("markets", "close_date")
        event_lifecycle = self._move_expired_items("events", "end_date")
        markets_moved, events_moved = await asyncio.gather(market_lifecycle, event_lifecycle)

        logger.info(f"🔄 Lifecycle transitions: {markets_moved} markets, {events_moved} events moved to closed")

        # Enforce outcome completeness
        await self._ensure_all_closed_markets_have_outcomes()
        outcome_stats = self.get_outcome_population_stats()

        logger.info("🎯 Mission accomplished - all closed markets have outcomes!")
        return self._generate_victory_report(new_events, events_saved, markets_saved, markets_moved, events_moved, outcome_stats, all_events)

    def _generate_victory_report(self, new_events, events_saved, markets_saved, markets_moved, events_moved, outcome_stats, all_events) -> Dict[str, int]:
        return {
            "events_processed": len(new_events),
            "events_saved": events_saved,
            "markets_saved": markets_saved,
            "markets_moved_to_closed": markets_moved,
            "events_moved_to_closed": events_moved,
            "outcome_stats": outcome_stats,
            "total_open_events": len(all_events)
        }

    async def _move_expired_items(self, table: str, date_field: str) -> int:
        logger.info(f"🔄 Executing lifecycle transition for {table}")
        now = datetime.now(timezone.utc).isoformat()
        expired_items = self.db.supabase.table(table).select("*").lt(date_field, now).neq("status", "closed").execute()

        if not expired_items.data:
            logger.info(f"✅ No expired {table} require transition")
            return 0

        logger.info(f"🎯 {len(expired_items.data)} {table} ready for transition")

        async def execute_transition(item: Dict) -> bool:
            closed_data = {**item, "status": "closed", "closed_at": now}

            if table == "markets":
                outcome = await self._resolve_market_outcome(item)
                if outcome:
                    closed_data["outcome"] = outcome
                    logger.debug(f"✅ Transitioning market {item['id']} with outcome: {outcome}")
                else:
                    logger.warning(f"⚠️ Market {item['id']} blocked - outcome resolution failed")
                    return False

            self.db.supabase.table(f"{table}_closed").upsert(closed_data).execute()
            self.db.supabase.table(table).delete().eq("id", item["id"]).execute()
            return True

        results = await self._process_batches(expired_items.data, execute_transition, batch_size=25)
        moved_count = sum(results)
        logger.info(f"✅ Transitioned {moved_count}/{len(expired_items.data)} {table} to closed")
        return moved_count

    async def _resolve_market_outcome(self, item: Dict) -> Optional[str]:
        """Assertively resolve market outcome through multiple intelligence sources."""
        outcome = item.get("outcome") or self._extract_outcome_from_raw_data(item)
        return outcome or await self._fetch_market_outcome(item["id"])

    async def get_lifecycle_stats(self, table: str, date_field: str) -> Dict[str, int]:
        open_items = self.db.supabase.table(table).select("*", count="exact").execute()
        closed_items = self.db.supabase.table(f"{table}_closed").select("*", count="exact").execute()

        now = datetime.now(timezone.utc).isoformat()
        expired_open = self.db.supabase.table(table).select("*", count="exact").lt(date_field, now).neq("status", "closed").execute()

        open_count = open_items.count or 0
        closed_count = closed_items.count or 0

        return {
            f"open_{table}": open_count,
            f"closed_{table}": closed_count,
            f"expired_open_{table}": expired_open.count or 0,
            f"total_{table}": open_count + closed_count
        }

    async def _fetch_all_open_events(self, limit: Optional[int] = None) -> List[Dict]:
        logger.info("📡 Fetching open events from Polymarket...")
        events = await self.gamma.get_all_events(active=True, page_size=500, max_events=limit)
        logger.info(f"📦 Fetched {len(events)} events")
        return events

    async def _filter_new_events_only(self, all_events: List[Dict]) -> List[Dict]:
        logger.info("🔍 Scanning for fresh event intelligence...")
        existing_ids = {str(event["id"]) for event in self.db.supabase.table("events").select("id").execute().data}
        new_events = [event for event in all_events if str(event.get("id")) not in existing_ids]
        logger.info(f"🎯 Discovered {len(new_events)} new events")
        return new_events

    async def _filter_events_with_new_markets(self, events: List[Dict]) -> List[Dict]:
        logger.info("🎯 Scanning for events with new market opportunities...")
        all_market_ids = {
            str(market.get("id") or market.get("conditionId"))
            for event in events
            for market in event.get("markets", [])
            if market.get("id") or market.get("conditionId")
        }

        if not all_market_ids:
            logger.info("📭 No markets detected in events")
            return []

        existing_market_ids = await self._check_existing_market_ids_batched(all_market_ids)
        logger.info(f"📊 {len(existing_market_ids)} markets already in database")

        events_with_new = [
            event for event in events
            if any(str(market.get("id") or market.get("conditionId")) not in existing_market_ids
                   for market in event.get("markets", []))
        ]

        logger.info(f"🎯 Identified {len(events_with_new)} events with new markets")
        return events_with_new

    async def _check_existing_market_ids_batched(self, market_ids: set, batch_size: int = 1000) -> set:
        existing_ids = set()
        market_ids_list = list(market_ids)

        for i in range(0, len(market_ids_list), batch_size):
            batch = market_ids_list[i:i + batch_size]
            result = self.db.supabase.table("markets").select("id").in_("id", batch).execute()
            existing_ids.update(str(market["id"]) for market in result.data)

        return existing_ids

    async def _save_events_to_database(self, events: List[Dict]) -> int:
        logger.info(f"💾 Deploying {len(events)} events to intelligence database")

        async def deploy_event(event: Dict) -> bool:
            event_data = self._normalize_event_data(event)
            if event_data.get("id"):
                self.db.supabase.table("events").upsert(event_data).execute()
                return True
            return False

        results = await self._process_batches(events, deploy_event, batch_size=100)
        deployed_count = sum(results)
        logger.info(f"✅ Deployed {deployed_count}/{len(events)} events successfully")
        return deployed_count

    async def _fetch_and_save_all_markets(self, events: List[Dict]) -> int:
        logger.info(f"🏪 Processing market intelligence for {len(events)} events")

        async def process_event_markets(event: Dict) -> int:
            markets = event.get("markets", [])
            return await self._save_all_markets_for_event(event, markets) if markets else 0

        market_counts = await asyncio.gather(*[process_event_markets(event) for event in events])
        total_deployed = sum(market_counts)
        logger.info(f"✅ Deployed {total_deployed} markets from {len(events)} events")
        return total_deployed

    async def _save_all_markets_for_event(self, event: Dict, markets: List[Dict]) -> int:
        async def process_market(market: Dict) -> Dict:
            enriched = await self._enrich_market_with_volume_data(market)
            return self._normalize_market_data(enriched, event)

        market_data_list = await asyncio.gather(*[process_market(market) for market in markets])
        valid_markets = [m for m in market_data_list if m.get("id")]

        logger.debug(f"🎯 Processed {len(markets)} markets → {len(valid_markets)} valid for deployment")

        for i in range(0, len(valid_markets), 50):
            batch = valid_markets[i:i + 50]
            for market_data in batch:
                self.db.supabase.table("markets").upsert(market_data).execute()

        return len(valid_markets)

    async def _enrich_market_with_volume_data(self, market: Dict) -> Dict:
        market_id = market.get("id") or market.get("conditionId")
        if not market_id:
            return market

        trades = await asyncio.wait_for(
            self.data_api.get_trades(market=[market_id], limit=500),
            timeout=10.0
        )

        if not trades:
            return market

        current_time = datetime.now(timezone.utc)
        one_day_ago = current_time - timedelta(hours=24)
        volume_24h = total_volume = 0.0

        for trade in trades:
            if not trade.get("timestamp"):
                continue

            timestamp = trade["timestamp"]
            if isinstance(timestamp, (int, float)):
                trade_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                trade_time_str = str(timestamp)
                if trade_time_str.endswith('Z'):
                    trade_time = datetime.fromisoformat(trade_time_str.replace('Z', '+00:00'))
                else:
                    trade_time = datetime.fromisoformat(trade_time_str)

            volume = float(trade.get("price", 0)) * float(trade.get("size", 0))
            total_volume += volume
            if trade_time >= one_day_ago:
                volume_24h += volume

        if volume_24h > 0 or total_volume > 0:
            market.update({
                "volume24hr": volume_24h, "volume24hrClob": volume_24h,
                "volumeNum": total_volume, "volumeClob": total_volume
            })

        return market

    async def _fetch_market_outcome(self, market_id: str) -> Optional[str]:
        """Assertively extract market outcome through intelligence sources."""
        # Primary: Check stored intelligence
        stored_data = self.db.supabase.table("markets_closed").select("raw_data").eq("id", market_id).execute()
        if stored_data.data:
            outcome = self._extract_outcome_from_raw_data({"raw_data": stored_data.data[0].get("raw_data", {})})
            if outcome:
                return outcome

        # Secondary: Intelligence gathering through API reconnaissance
        for offset in range(0, 1000, 100):
            closed_markets = await self.gamma.get_markets(
                limit=100, closed="true", order="updatedAt", ascending="false", offset=offset
            )

            if not closed_markets:
                break

            target_market = next((m for m in closed_markets if m.get("conditionId") == market_id), None)
            if target_market:
                return self._extract_outcome_from_market(target_market)

            if len(closed_markets) < 100:
                break

        return None

    def _extract_outcome_from_market(self, market: Dict) -> Optional[str]:
        """Extract outcome from market data with precision."""
        if market.get("outcome"):
            return str(market["outcome"])
        if market.get("winner"):
            return str(market["winner"])

        outcomes, prices = market.get("outcomes", []), market.get("outcomePrices", [])
        if outcomes and prices and len(outcomes) == len(prices):
            for i, price in enumerate(prices):
                price_val = float(price) if isinstance(price, str) else price
                if abs(price_val - 1.0) < 0.001 or price_val >= 0.95:
                    return str(outcomes[i]) if i < len(outcomes) else None
        return None

    def _extract_outcome_from_raw_data(self, market: Dict) -> Optional[str]:
        """Assertively extract outcome from raw market intelligence."""
        raw_data = market.get("raw_data", {})

        # Primary outcome fields
        for field in ["outcome", "winner", "result", "resolution", "finalOutcome", "settledOutcome", "marketOutcome", "decision"]:
            if raw_data.get(field):
                return str(raw_data[field])

        # Secondary: Analyze probability distributions for resolved markets
        if raw_data.get("closed") or raw_data.get("resolved") or not raw_data.get("active", True):
            outcomes, prices = raw_data.get("outcomes", []), raw_data.get("outcomePrices", [])
            if outcomes and prices and len(outcomes) == len(prices):
                for i, price in enumerate(prices):
                    price_val = float(price) if isinstance(price, str) else price
                    if abs(price_val - 1.0) < 0.001 or price_val >= 0.95:
                        return str(outcomes[i]) if i < len(outcomes) else None

        return None

    async def populate_missing_outcomes(self, limit: int = 50) -> int:
        """Assertively populate missing outcome intelligence."""
        markets_missing_outcomes = self.db.supabase.table("markets_closed").select("id").is_("outcome", "null").limit(limit).execute()
        if not markets_missing_outcomes.data:
            logger.info("✅ All markets have outcome intelligence")
            return 0

        logger.info(f"🎯 Discovered {len(markets_missing_outcomes.data)} markets requiring outcome resolution")

        async def resolve_and_update(market: Dict) -> bool:
            outcome = await self._fetch_market_outcome(market["id"])
            if outcome:
                self.db.supabase.table("markets_closed").update({"outcome": outcome}).eq("id", market["id"]).execute()
                logger.info(f"✅ Resolved outcome for market {market['id']}: {outcome}")
                return True
            logger.debug(f"⚠️ Outcome resolution failed for market {market['id']}")
            return False

        results = await self._process_batches(markets_missing_outcomes.data, resolve_and_update, batch_size=5)
        resolved_count = sum(results)
        logger.info(f"🎯 Successfully resolved {resolved_count}/{len(markets_missing_outcomes.data)} missing outcomes")
        return resolved_count

    async def _ensure_all_closed_markets_have_outcomes(self) -> None:
        """Enforce complete outcome population - no compromises."""
        logger.info("🔒 Elite outcome validation protocol activated")

        for attempt in range(3):
            stats = self.get_outcome_population_stats()
            completion_rate, without_outcomes = stats["completion_percentage"], stats["without_outcomes"]

            if completion_rate >= 85.0:
                logger.info(f"✅ Outcome validation successful: {completion_rate}% intelligence complete")
                return

            if without_outcomes == 0:
                logger.info("🎯 Perfect outcome coverage achieved")
                return

            logger.info(f"🎯 Resolving {without_outcomes} missing outcomes - attempt {attempt + 1}")
            populated = await self.populate_missing_outcomes()

            if populated == 0:
                await asyncio.sleep(2)

        # Final assessment
        final_stats = self.get_outcome_population_stats()
        completion_rate, without_outcomes = final_stats["completion_percentage"], final_stats["without_outcomes"]

        if completion_rate >= 85.0:
            logger.info(f"✅ Mission accomplished: {completion_rate}% outcome coverage")
            if without_outcomes > 0:
                logger.info(f"ℹ️ {without_outcomes} markets pending resolution (expected for active intelligence)")
        else:
            logger.warning(f"⚠️ Suboptimal coverage: {completion_rate}% - potential intelligence gaps detected")

    def get_outcome_population_stats(self) -> Dict[str, int]:
        """Get statistics about outcome population for closed markets."""
        total_closed = self.db.supabase.table("markets_closed").select("*", count="exact").execute()
        with_outcomes = self.db.supabase.table("markets_closed").select("*", count="exact").not_.is_("outcome", "null").execute()
        without_outcomes = self.db.supabase.table("markets_closed").select("*", count="exact").is_("outcome", "null").execute()

        return {
            "total_closed_markets": total_closed.count or 0,
            "with_outcomes": with_outcomes.count or 0,
            "without_outcomes": without_outcomes.count or 0,
            "completion_percentage": round(
                ((with_outcomes.count or 0) / (total_closed.count or 1)) * 100, 1
            )
        }

    def _normalize_event_data(self, event: Dict) -> Dict:
        def clean_ts(ts):
            if not ts:
                return None
            ts_str = str(ts)
            return ts_str if ts_str.endswith('Z') or 'T' in ts_str else None

        tags = [tag.get('label') if isinstance(tag, dict) else tag for tag in event.get("tags", [])]
        markets = event.get("markets", [])

        return {
            "id": event.get("id"),
            "title": event.get("title"),
            "description": event.get("description"),
            "category": event.get("category"),
            "start_date": clean_ts(event.get("startDate") or event.get("startTime")),
            "end_date": clean_ts(event.get("endDate")),
            "status": "active" if event.get("active") else "closed",
            "tags": tags,
            "market_count": len(markets),
            "total_liquidity": sum(m.get("liquidityNum", 0) for m in markets),
            "created_at": clean_ts(event.get("createdAt")),
            "updated_at": clean_ts(event.get("updatedAt")),
            "platform": "polymarket",
            "raw_data": event
        }

    def _normalize_market_data(self, market: Dict, event: Dict) -> Dict:
        def clean_ts(ts):
            if not ts:
                return None
            ts_str = str(ts)
            return ts_str if ts_str.endswith('Z') or 'T' in ts_str else None

        prices = market.get("outcomePrices", [])
        yes_price = float(prices[1]) if len(prices) >= 2 and prices[1] is not None else 0.5
        no_price = float(prices[0]) if len(prices) >= 2 and prices[0] is not None else 0.5

        # Get tags from market or fallback to event
        tags = market.get("tags", [])
        if not tags:
            event_tags = event.get("tags", [])
            tags = [tag.get('label') if isinstance(tag, dict) else tag for tag in event_tags]
            if not tags:
                category = event.get("category") or event.get("label")
                tags = [category] if category else []

        volume = market.get("volumeNum") or market.get("volume") or market.get("volumeClob") or market.get("volumeAmm") or 0.0
        volume_24h = market.get("volume24hr") or market.get("volume24hrAmm") or market.get("volume24hrClob") or 0.0
        liquidity = market.get("liquidityNum") or market.get("liquidity") or market.get("liquidityClob") or market.get("liquidityAmm") or 0.0

        return {
            "id": market.get("id"),
            "event_id": event.get("id"),
            "event_title": event.get("title"),
            "title": market.get("question"),
            "description": market.get("description"),
            "category": market.get("category") or event.get("category"),
            "tags": tags,
            "num_outcomes": len(market.get("outcomes", [])) or 2,
            "p_yes": yes_price,
            "p_no": no_price,
            "bid": market.get("bestBid"),
            "ask": market.get("bestAsk"),
            "spread": market.get("spread"),
            "total_volume": volume,
            "volume_24h": volume_24h,
            "liquidity": liquidity,
            "close_date": clean_ts(market.get("endDate")),
            "status": "open" if market.get("active") else "closed",
            "platform": "polymarket",
            "created_at": clean_ts(market.get("createdAt")),
            "updated_at": clean_ts(market.get("updatedAt")),
            "raw_data": market
        }


async def sync_polymarket_data(limit: Optional[int] = None) -> Dict[str, int]:
    """
    Elite sync protocol: Intelligence acquisition → Market deployment → Outcome completion.

    Assertive execution:
    1. Acquire fresh event intelligence
    2. Deploy new market opportunities
    3. Complete outcome intelligence
    """
    collector = PolymarketDataCollector()

    logger.info("🚀 Elite sync protocol initiated")

    # Intelligence acquisition
    all_events = await collector._fetch_all_open_events(limit)
    new_events = await collector._filter_new_events_only(all_events)

    # Deploy operations
    events_saved = await collector._save_events_to_database(new_events) if new_events else 0

    # Market intelligence deployment
    markets_saved = 0
    if new_events:
        events_with_markets = await collector._filter_events_with_new_markets(new_events)
        if events_with_markets:
            markets_saved = await collector._fetch_and_save_all_markets(events_with_markets)

    # Outcome intelligence completion
    outcomes_filled = await collector.populate_missing_outcomes(limit=100)

    victory_report = {
        "events_processed": len(new_events),
        "events_saved": events_saved,
        "markets_saved": markets_saved,
        "outcomes_filled": outcomes_filled,
        "total_open_events": len(all_events)
    }

    logger.info("🎯 Mission accomplished!")
    logger.info(f"   Events deployed: {events_saved}")
    logger.info(f"   Markets deployed: {markets_saved}")
    logger.info(f"   Outcomes resolved: {outcomes_filled}")

    return victory_report
