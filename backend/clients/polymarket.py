"""
Polymarket API Client

Gamma API (events/markets) + CLOB API (orderbooks/prices).

Goals:
- Fetch and normalize all Polymarket events and markets.
- Enrich markets with probabilities and orderbook metrics.
- Provide a simple interface to persist normalized data into a database.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from importlib.util import find_spec
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from config import settings


logger = logging.getLogger(__name__)


# Optional CLOB client import without try/except
if find_spec("py_clob_client") is not None:
    from py_clob_client.client import ClobClient  # type: ignore[import]
else:
    ClobClient = None  # type: ignore[assignment]


class PolymarketDataAPI:
    """
    High-performance Polymarket Data API wrapper optimized for maximum throughput.

    Official API limits (requests per 10 seconds):
    - GET /trades: 75 req/10s (7.5 req/s)
    - Other endpoints: 200 req/10s (20 req/s)

    Default: Maximum performance at API limits for fastest data retrieval.
    """

    BASE_URL = "https://data-api.polymarket.com"

    def __init__(self, timeout: float = 5.0):
        """
        Initialize Polymarket Data API client with maximum performance.

        Args:
            timeout: HTTP request timeout in seconds (optimized for speed)
        """
        self.base_url = self.BASE_URL
        self.timeout = timeout

        # Maximum rate limits (at API limits)
        self.rate_limit_trades = 7.5    # 75 req/10s = 7.5 req/s
        self.rate_limit_default = 20.0  # 200 req/10s = 20 req/s

        # Rate limiting state - token bucket for concurrent requests
        # Allow up to 8 concurrent trades requests (75/10s allows ~7-8 in flight)
        self._trades_semaphore = asyncio.Semaphore(8)
        self._default_semaphore = asyncio.Semaphore(20)
        
        # Sliding window rate limiter: track request times
        self._trades_request_times = []
        self._default_request_times = []
        self._rate_lock = asyncio.Lock()

    async def _rate_limited_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        is_trades: bool = False
    ) -> Any:
        """Make maximum-throughput rate-limited HTTP request with proper concurrency.
        
        Uses sliding window to allow up to 75 req/10s for trades (7.5 req/s).
        Allows multiple concurrent requests while respecting rate limits.
        """
        semaphore = self._trades_semaphore if is_trades else self._default_semaphore
        request_times = self._trades_request_times if is_trades else self._default_request_times
        max_requests = 75 if is_trades else 200
        window_seconds = 10.0

        async with semaphore:
            # Sliding window rate limiting
            async with self._rate_lock:
                now = asyncio.get_event_loop().time()
                # Remove requests older than window
                request_times[:] = [t for t in request_times if now - t < window_seconds]
                
                # If at limit, wait until oldest request expires
                if len(request_times) >= max_requests:
                    oldest_time = min(request_times)
                    wait_time = window_seconds - (now - oldest_time) + 0.1  # Small buffer
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = asyncio.get_event_loop().time()
                        # Clean up again after wait
                        request_times[:] = [t for t in request_times if now - t < window_seconds]
                
                # Record this request
                request_times.append(now)

            # Fast HTTP request
            url = f"{self.base_url}{endpoint}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params or {})
                response.raise_for_status()
                return response.json()

    async def get_trades(
        self,
        user: Optional[str] = None,
        market: Optional[List[str]] = None,
        event_id: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        side: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch trades from Polymarket Data API.

        Args:
            user: Filter by wallet address (0x-prefixed)
            market: Filter by market condition IDs (CSV)
            event_id: Filter by event IDs (CSV, mutually exclusive with market)
            limit: Results per page (0-10,000)
            offset: Pagination offset (0-10,000)
            side: BUY or SELL
            **kwargs: Additional query parameters

        Returns:
            List of trade objects
        """
        params = {"limit": min(limit, 10000), "offset": offset}
        if user:
            params["user"] = user
        if market:
            params["market"] = ",".join(market)
        if event_id:
            params["eventId"] = ",".join(event_id)
        if side:
            params["side"] = side
        params.update(kwargs)

        return await self._rate_limited_request("/trades", params, is_trades=True)

    async def get_closed_positions(
        self,
        user: str,
        market: Optional[List[str]] = None,
        event_id: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "REALIZEDPNL",
        sort_direction: str = "DESC",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch closed positions for a wallet.

        Args:
            user: Wallet address (0x-prefixed) [REQUIRED]
            market: Filter by market condition IDs
            event_id: Filter by event IDs
            limit: Results per page (1-50)
            offset: Pagination offset (0-100,000)
            sort_by: REALIZEDPNL, TITLE, PRICE, AVGPRICE, TIMESTAMP
            sort_direction: ASC or DESC
            **kwargs: Additional query parameters

        Returns:
            List of closed position objects
        """
        params = {
            "user": user,
            "limit": min(limit, 50),
            "offset": offset,
            "sortBy": sort_by,
            "sortDirection": sort_direction
        }
        if market:
            params["market"] = ",".join(market)
        if event_id:
            params["eventId"] = ",".join(event_id)
        params.update(kwargs)

        return await self._rate_limited_request("/closed-positions", params)

    async def get_positions(
        self,
        user: str,
        market: Optional[List[str]] = None,
        event_id: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch current open positions for a wallet.

        Args:
            user: Wallet address (0x-prefixed) [REQUIRED]
            market: Filter by market condition IDs
            event_id: Filter by event IDs
            limit: Results per page (0-500)
            offset: Pagination offset (0-10,000)
            **kwargs: Additional query parameters

        Returns:
            List of open position objects
        """
        params = {"user": user, "limit": min(limit, 500), "offset": offset}
        if market:
            params["market"] = ",".join(market)
        if event_id:
            params["eventId"] = ",".join(event_id)
        params.update(kwargs)

        return await self._rate_limited_request("/positions", params)

    async def get_holders(
        self,
        market: List[str],
        limit: int = 100,
        min_balance: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch top holders for markets.

        Args:
            market: Market condition IDs (CSV) [REQUIRED]
            limit: Max holders per token (0-500)
            min_balance: Minimum position size (0-999,999)

        Returns:
            List of meta-holder objects (token → holders array)
        """
        params = {
            "market": ",".join(market),
            "limit": min(limit, 500),
            "minBalance": min_balance
        }

        return await self._rate_limited_request("/holders", params)

    async def get_activity(
        self,
        user: str,
        activity_type: Optional[List[str]] = None,
        market: Optional[List[str]] = None,
        event_id: Optional[List[str]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch on-chain activity for a wallet.

        Args:
            user: Wallet address (0x-prefixed) [REQUIRED]
            activity_type: Filter by types (TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION)
            market: Filter by market condition IDs
            event_id: Filter by event IDs
            start: Start timestamp (Unix seconds)
            end: End timestamp (Unix seconds)
            limit: Results per page (0-500)
            offset: Pagination offset (0-10,000)
            **kwargs: Additional query parameters

        Returns:
            List of activity objects
        """
        params = {"user": user, "limit": min(limit, 500), "offset": offset}
        if activity_type:
            params["type"] = ",".join(activity_type)
        if market:
            params["market"] = ",".join(market)
        if event_id:
            params["eventId"] = ",".join(event_id)
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        params.update(kwargs)

        return await self._rate_limited_request("/activity", params)

    async def get_portfolio_value(
        self,
        user: str,
        market: Optional[List[str]] = None
    ) -> float:
        """
        Get total portfolio value for a wallet.

        Args:
            user: Wallet address (0x-prefixed) [REQUIRED]
            market: Filter by specific markets

        Returns:
            Total value in USDC
        """
        params = {"user": user}
        if market:
            params["market"] = ",".join(market)

        data = await self._rate_limited_request("/value", params)
        return data[0]["value"] if data else 0.0


class PolymarketError(Exception):
    """Base Polymarket exception."""


class MarketNotFoundError(PolymarketError):
    """Raised when a specific market or event cannot be found."""


class InvalidOrderbookError(PolymarketError):
    """Raised when an orderbook is structurally invalid or empty."""


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string with optional trailing 'Z'."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _ensure_list(value: Any) -> List[Any]:
    """Return value as list; strings and None become empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _parse_json_list_field(container: Dict[str, Any], field: str) -> List[Any]:
    """
    Parse a JSON-encoded list field in-place and return the parsed list.

    If the field is already a list, it is returned unchanged.
    If the value is a string that looks like a JSON array, json.loads is applied.
    Otherwise the field is set to [].
    """
    raw = container.get(field)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("["):
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        container[field] = parsed
        return parsed
    container[field] = []
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion with a bounded domain for prices and probabilities."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text)


class PolymarketGamma:
    """
    Client for the Polymarket Gamma REST API.

    This client is responsible for:
    - Fetching raw events and markets from /events and /markets endpoints.
    - Supporting efficient pagination for full-universe crawls.
    - Normalizing event/market payloads for downstream processing.
    """

    def __init__(self, *, timeout: float = 30.0, default_page_size: int = 100) -> None:
        base = getattr(settings, "POLYMARKET_GAMMA_URL", "").rstrip("/")
        self.base_url = base or "https://gamma-api.polymarket.com"
        self.timeout = timeout
        self.default_page_size = default_page_size

    async def get_events(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a single page of events.

        Args:
            active: If True, only open events (maps to closed=false).
            limit: Page size (1–1000, clamped).
            offset: Offset for pagination.
            **filters: Additional Gamma query parameters (tag_id, order, ascending, etc.).

        Returns:
            List of normalized event dicts (each may contain embedded markets).
        """
        assert limit > 0
        page_limit = min(limit, 1000)
        params: Dict[str, Any] = {"limit": page_limit, "offset": offset}
        if "closed" in filters:
            params["closed"] = filters.pop("closed")
        elif active:
            params["closed"] = "false"
        params.update(filters)

        data = await self._get("/events", params)
        assert isinstance(data, list)
        return [self._normalize_event(e) for e in data]

    async def get_all_events(
        self,
        *,
        active: bool = True,
        page_size: Optional[int] = None,
        max_events: Optional[int] = None,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all events matching filters via paginated /events calls.

        Args:
            active: If True, restrict to open events (closed=false).
            page_size: Items per page (defaults to self.default_page_size).
            max_events: Hard cap on total events to fetch.
            **filters: Additional Gamma query parameters.

        Returns:
            List of normalized event dicts.
        """
        size = page_size or self.default_page_size
        assert size > 0
        events: List[Dict[str, Any]] = []
        offset = 0

        while True:
            remaining = size
            if max_events is not None:
                remaining = min(remaining, max_events - len(events))
                if remaining <= 0:
                    break

            batch = await self.get_events(
                active=active,
                limit=remaining,
                offset=offset,
                order="id",
                ascending="false",
                **filters,
            )
            if not batch:
                break

            events.extend(batch)
            if len(batch) < remaining:
                break

            offset += remaining

        return events

    async def get_markets(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a single page of markets.

        Args:
            active: If True, only open markets (maps to closed=false).
            limit: Page size (1–1000, clamped).
            offset: Offset for pagination.
            **filters: Additional Gamma query parameters.

        Returns:
            List of normalized market dicts.
        """
        assert limit > 0
        page_limit = min(limit, 1000)
        params: Dict[str, Any] = {"limit": page_limit, "offset": offset}
        if "closed" in filters:
            params["closed"] = filters.pop("closed")
        elif active:
            params["closed"] = "false"
        params.update(filters)

        data = await self._get("/markets", params)
        assert isinstance(data, list)
        return [self._normalize_market(m) for m in data]

    async def get_market(self, slug: str) -> Dict[str, Any]:
        """
        Retrieve a single market by slug via /markets/slug/{slug}.
        """
        assert slug
        data = await self._get(f"/markets/slug/{slug}", None)
        assert isinstance(data, dict)
        market = self._normalize_market(data)
        if not market:
            raise MarketNotFoundError(f"Market not found for slug={slug}")
        return market

    async def get_event(self, slug: str) -> Dict[str, Any]:
        """
        Retrieve a single event by slug via /events/slug/{slug}.
        """
        assert slug
        data = await self._get(f"/events/slug/{slug}", None)
        assert isinstance(data, dict)
        event = self._normalize_event(data)
        if not event:
            raise MarketNotFoundError(f"Event not found for slug={slug}")
        return event

    async def get_market_by_condition_id(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single market by its conditionId.

        Note: Due to Polymarket API inconsistency, querying by conditionId may not
        return resolved outcomePrices. This method tries multiple query strategies
        to get the most accurate data.

        Args:
            condition_id: The market condition ID (hex string like 0x...)

        Returns:
            Normalized market dict if found, None otherwise.
        """
        assert condition_id

        # Strategy 1: Query with conditionId filter
        params = {"conditionId": condition_id, "limit": 1}
        data = await self._get("/markets", params)

        if isinstance(data, list) and len(data) > 0:
            market = self._normalize_market(data[0])

            # Check if we need resolution data but got ["0", "0"]
            if market.get("closed"):
                prices = market.get("outcomePrices", [])
                # If prices are all zeros, try fetching with closed=true filter for better data
                if prices and all(str(p) == "0" for p in prices):
                    # Strategy 2: Query closed markets and find our specific one
                    # This is slower but returns accurate resolution data
                    try:
                        closed_params = {
                            "closed": "true",
                            "conditionId": condition_id,
                            "limit": 1,
                            "order": "updatedAt",
                            "ascending": "false"
                        }
                        closed_data = await self._get("/markets", closed_params)
                        if isinstance(closed_data, list) and len(closed_data) > 0:
                            closed_market = self._normalize_market(closed_data[0])
                            # Merge resolution data if better
                            closed_prices = closed_market.get("outcomePrices", [])
                            if closed_prices and not all(str(p) == "0" for p in closed_prices):
                                market["outcomePrices"] = closed_prices
                    except Exception:
                        pass  # Use original data if fallback fails

            return market

        return None

    async def _get(self, path: str, params: Optional[Dict[str, Any]]) -> Any:
        """Low-level GET helper."""
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize an Event payload.

        - Ensures 'markets' and 'tags' are always lists.
        - Fills event['category'] from tags when missing.
        - Normalizes embedded markets.
        """
        markets = _ensure_list(event.get("markets"))
        tags = _ensure_list(event.get("tags"))

        if not event.get("category") and tags:
            head = tags[0]
            if isinstance(head, dict):
                event["category"] = head.get("label") or head.get("slug")

        event["markets"] = [self._normalize_market(m) for m in markets]
        event["tags"] = tags
        return event

    def _normalize_market(self, market: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Market payload.

        - Parses JSON-encoded string fields ('outcomes', 'outcomePrices', 'clobTokenIds').
        - Converts numeric fields to float.
        - Propagates category/tags from embedded event when missing.
        - Adds convenience aliases (close_date, volume_24h, volume_total).
        """
        outcomes = _parse_json_list_field(market, "outcomes")
        prices = _parse_json_list_field(market, "outcomePrices")
        clob_ids = _parse_json_list_field(market, "clobTokenIds")

        market["outcomes"] = outcomes
        market["outcomePrices"] = prices
        market["clobTokenIds"] = clob_ids

        numeric_fields = (
            "volumeNum",
            "liquidityNum",
            "volume",
            "liquidity",
            "volume24hr",
            "volume1wk",
            "volume1mo",
            "volume1yr",
        )
        for name in numeric_fields:
            if name in market:
                market[name] = _to_float(market.get(name))

        if not market.get("category"):
            events = _ensure_list(market.get("events"))
            if events:
                parent = events[0]
                if isinstance(parent, dict):
                    category = parent.get("category")
                    if not category:
                        parent_tags = _ensure_list(parent.get("tags"))
                        if parent_tags and isinstance(parent_tags[0], dict):
                            category = parent_tags[0].get("label") or parent_tags[0].get("slug")
                    market["category"] = category

        if not market.get("tags"):
            events = _ensure_list(market.get("events"))
            if events and isinstance(events[0], dict):
                market["tags"] = _ensure_list(events[0].get("tags"))
            else:
                market["tags"] = []

        if not market.get("close_date") and market.get("endDate"):
            market["close_date"] = market["endDate"]

        if not market.get("volume_24h") and market.get("volume24hr") is not None:
            market["volume_24h"] = _to_float(market.get("volume24hr"))

        if not market.get("volume_total") and market.get("volumeNum") is not None:
            market["volume_total"] = _to_float(market.get("volumeNum"))

        market.setdefault("active", True)
        market.setdefault("slug", "")
        market.setdefault("question", "")

        return market


class PolymarketCLOB:
    """
    Wrapper around the Polymarket CLOB client.

    Responsible for:
    - Fetching orderbooks for a given token_id.
    - Computing best bid/ask, spreads, and depth-based liquidity.
    """

    def __init__(self) -> None:
        base = getattr(settings, "POLYMARKET_CLOB_URL", "").rstrip("/")
        host = base or "https://clob.polymarket.com"
        chain_id = getattr(settings, "POLYMARKET_CHAIN_ID", 137)

        if ClobClient is None:
            self.client = None
        else:
            key = getattr(settings, "POLYMARKET_PRIVATE_KEY", "").strip()
            if key and len(key) > 16 and " " not in key:
                self.client = ClobClient(host, key=key, chain_id=chain_id)
            else:
                self.client = ClobClient(host, chain_id=chain_id)

    def has_client(self) -> bool:
        """Return True if the underlying CLOB client is available."""
        return self.client is not None

    def get_orderbook(self, token_id: str, depth: int = 50) -> Dict[str, Any]:
        """
        Fetch and normalize the orderbook for a given token_id.

        Args:
            token_id: CLOB token id (YES or NO token).
            depth: Maximum number of levels for bids/asks.

        Returns:
            Dict with bids/asks and best bid/ask metrics.

        Raises:
            InvalidOrderbookError: If client is missing or book is empty/invalid.
        """
        if not self.client:
            raise InvalidOrderbookError("CLOB client unavailable")
        assert token_id
        assert depth > 0

        orderbook_obj = self.client.get_order_book(token_id)
        if not hasattr(orderbook_obj, "__dict__"):
            raise InvalidOrderbookError("Orderbook response is not a valid object")

        raw: Dict[str, Any] = dict(orderbook_obj.__dict__)
        bids = self._convert_orders(raw.get("bids", []))
        asks = self._convert_orders(raw.get("asks", []))

        if not bids or not asks:
            raise InvalidOrderbookError("Orderbook has no bids or asks")

        bids = bids[:depth]
        asks = asks[:depth]

        best_bid = _to_float(bids[0].get("price"))
        best_ask = _to_float(asks[0].get("price"))
        assert 0 <= best_bid <= 1
        assert 0 <= best_ask <= 1
        assert best_ask >= best_bid

        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = spread * 10_000.0

        bid_liq = sum(_to_float(o.get("price")) * _to_float(o.get("size")) for o in bids[:20])
        ask_liq = sum(_to_float(o.get("price")) * _to_float(o.get("size")) for o in asks[:20])
        total_liquidity = min(bid_liq, ask_liq)

        return {
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": spread_bps,
            "total_liquidity": total_liquidity,
        }

    def _convert_orders(self, orders: Iterable[Any]) -> List[Dict[str, Any]]:
        """Coerce CLOB order objects into plain dicts."""
        normalized: List[Dict[str, Any]] = []
        for order in orders:
            if isinstance(order, dict):
                normalized.append(order)
            elif hasattr(order, "__dict__"):
                normalized.append(dict(order.__dict__))
        return normalized

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """
        Fetch the current CLOB price for a token on a given side.

        Args:
            token_id: CLOB token id.
            side: "BUY" or "SELL".

        Returns:
            Price as float in [0, 1].
        """
        if not self.client:
            raise InvalidOrderbookError("CLOB client unavailable")
        assert token_id
        price = self.client.get_price(token_id, side=side)
        value = _to_float(price)
        assert 0.0 <= value <= 1.0
        return value


class PolymarketClient:
    """
    High-level Polymarket client.

    Responsibilities:
    - Expose simple methods to fetch events/markets and enrich them with CLOB data.
    - Provide formatting helpers for display.
    - Provide utilities to sync normalized events/markets/orderbooks into a database.
    """

    def __init__(self) -> None:
        self.gamma = PolymarketGamma()
        self.clob = PolymarketCLOB()

    async def get_event(self, slug: str, with_orderbooks: bool = False) -> Dict[str, Any]:
        """
        Retrieve a single event by slug and format its markets.

        Args:
            slug: Event slug from the Polymarket URL.
            with_orderbooks: If True, each market is enriched with best-bid/ask liquidity.

        Returns:
            Dict with event metadata and formatted markets under "event" key.
        """
        event = await self.gamma.get_event(slug)
        return await self._format_event(event, with_orderbooks)

    async def get_events(
        self,
        limit: int = 10,
        with_orderbooks: bool = False,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve multiple events and format each.

        Args:
            limit: Max number of events to return.
            with_orderbooks: If True, markets are enriched with orderbooks.
            **filters: Additional Gamma filters (tag_id, closed, etc.).

        Returns:
            List of formatted event dicts.
        """
        assert limit > 0
        events_raw = await self.gamma.get_all_events(
            active=filters.pop("active", True),
            page_size=min(limit, 100),
            max_events=limit,
            **filters,
        )
        return [await self._format_event(ev, with_orderbooks) for ev in events_raw]

    async def get_markets(
        self,
        limit: int = 200,
        min_liquidity: float = 100.0,
        max_spread: float = 0.98,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve markets enriched with CLOB data.

        Args:
            limit: Maximum number of markets in the final result.
            min_liquidity: Minimum notional liquidity (USD) required to keep a market.
            max_spread: Maximum allowed spread in absolute price terms (0–1).
            **filters: Additional Gamma filters (tag_id, closed, etc.).

        Returns:
            List of markets with liquidity, spread and yes-probability fields.
        """
        assert limit > 0
        assert min_liquidity >= 0
        assert 0.0 < max_spread <= 1.0

        page_size = min(limit * 5, 1000)
        raw_markets = await self.gamma.get_markets(
            active=filters.pop("active", True),
            limit=page_size,
            **filters,
        )

        enriched: List[Dict[str, Any]] = []
        for market in raw_markets:
            if len(enriched) >= limit:
                break
            enriched_market = await self._enrich_market(market, min_liquidity, max_spread)
            if enriched_market is not None:
                enriched.append(enriched_market)

        logger.info("polymarket.get_markets: %d markets enriched", len(enriched))
        return enriched

    async def sync_all_events_and_markets(
        self,
        save_event: Callable[[Dict[str, Any]], None],
        save_market: Callable[[Dict[str, Any]], None],
        save_orderbook: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        only_active: bool = True,
        page_size: int = 100,
        max_events: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Crawl Gamma for events + markets and persist them through callbacks.

        Args:
            save_event: Callable that persists a normalized event record.
            save_market: Callable that persists a normalized market record.
            save_orderbook: Optional callable to persist orderbook metrics.
            only_active: If True, restrict to open events.
            page_size: Gamma page size for crawling.
            max_events: Optional hard cap on events to ingest.

        Returns:
            Summary dict with counts and an integrity report.
        """
        events = await self.gamma.get_all_events(
            active=only_active,
            page_size=page_size,
            max_events=max_events,
        )

        market_records: List[Dict[str, Any]] = []
        orderbook_records: List[Dict[str, Any]] = []

        for event in events:
            event_id = event.get("id")

            # Re-normalize the event to ensure category is properly set
            normalized_event = self.gamma._normalize_event(event)

            # Extract market count from embedded markets
            markets_list = _ensure_list(normalized_event.get("markets", []))
            market_count = len(markets_list)

            # Calculate total liquidity from all markets in the event
            total_liquidity = 0.0
            for market in markets_list:
                # Normalize each market too
                normalized_market = self.gamma._normalize_market(market)
                market_liquidity = normalized_market.get("liquidityNum") or normalized_market.get("liquidity") or 0.0
                total_liquidity += _to_float(market_liquidity)

            # Extract tags - handle both nested and flat structures
            tags = _ensure_list(normalized_event.get("tags", []))
            if tags and isinstance(tags[0], dict):
                # Tags are objects with label/slug
                tag_labels = [tag.get("label") or tag.get("slug", "") for tag in tags if isinstance(tag, dict)]
            else:
                # Tags are already strings
                tag_labels = [str(tag) for tag in tags]

            # Ensure we have a clean list of strings
            tag_labels = [tag for tag in tag_labels if tag and isinstance(tag, str)]

            # Ensure category is set from tags if missing
            category = normalized_event.get("category")
            if not category and tag_labels:
                # Use first tag as category
                category = tag_labels[0]

            save_event(
                {
                    "id": event_id,
                    "slug": normalized_event.get("slug") or normalized_event.get("ticker"),
                    "title": normalized_event.get("title"),
                    "description": normalized_event.get("description"),
                    "category": category,  # Use the properly extracted category
                    "start_date": normalized_event.get("startDate") or normalized_event.get("startTime"),
                    "end_date": normalized_event.get("endDate"),
                    "active": normalized_event.get("active"),
                    "closed": normalized_event.get("closed"),
                    "liquidity": normalized_event.get("liquidity"),
                    "volume": normalized_event.get("volume"),
                    "open_interest": normalized_event.get("openInterest"),
                    "tags": tag_labels,
                    "market_count": market_count,  # Now properly calculated
                    "total_liquidity": total_liquidity,  # Now properly calculated
                    "created_at": normalized_event.get("createdAt"),
                    "updated_at": normalized_event.get("updatedAt"),
                    "raw": normalized_event,
                }
            )

        for market in normalized_event.get("markets", []):
            yes_price, no_price = self._extract_yes_no_prices(market)
            clob_ids = _ensure_list(market.get("clobTokenIds"))
            yes_token_id = clob_ids[1] if len(clob_ids) >= 2 else None

            # Extract bid/ask from orderbook if available
            bid = market.get("bestBid")
            ask = market.get("bestAsk")

            # Extract tags from market (usually empty but handle if present)
            # Markets inherit tags from their parent event
            market_tags = _ensure_list(market.get("tags", []))
            if market_tags and isinstance(market_tags[0], dict):
                market_tag_labels = [tag.get("label") or tag.get("slug", "") for tag in market_tags if isinstance(tag, dict)]
            else:
                market_tag_labels = [str(tag) for tag in market_tags]

            # If market has no tags, inherit from parent event
            if not market_tag_labels:
                market_tag_labels = tag_labels.copy()

            # Ensure we have a clean list of strings
            market_tag_labels = [tag for tag in market_tag_labels if tag and isinstance(tag, str)]

            market_record = {
                "id": market.get("id"),
                "event_id": event_id,
                "event_title": event.get("title"),  # Add event title
                "slug": market.get("slug"),
                "question": market.get("question"),
                "description": market.get("description"),  # Add description
                "category": market.get("category") or event.get("category"),
                "tags": market_tag_labels,  # Add tags
                "outcomes": market.get("outcomes"),
                "yes_price": yes_price,
                "no_price": no_price,
                "bid": bid,  # Add bid price
                "ask": ask,  # Add ask price
                "volume_total": market.get("volume_total") or market.get("volume"),
                "volume_24h": market.get("volume_24h") or market.get("volume24hr"),
                "liquidity_num": market.get("liquidityNum") or market.get("liquidity"),
                "end_date": market.get("endDate"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "clob_token_ids": clob_ids,
                "created_at": market.get("createdAt"),  # Add created_at
                "updated_at": market.get("updatedAt"),  # Add updated_at
                "raw": market,
            }
            save_market(market_record)
            market_records.append(market_record)

            if save_orderbook and yes_token_id and self.clob.has_client():
                ob = self.clob.get_orderbook(yes_token_id, depth=20)
                orderbook_record = {
                    "market_id": market.get("id"),
                    "event_id": event_id,
                    "token_id": yes_token_id,
                    "best_bid": ob["best_bid"],
                    "best_ask": ob["best_ask"],
                    "mid_price": ob["mid_price"],
                    "spread": ob["spread"],
                    "spread_bps": ob["spread_bps"],
                    "total_liquidity": ob["total_liquidity"],
                    "bids": ob["bids"],
                    "asks": ob["asks"],
                }
                save_orderbook(orderbook_record)
                orderbook_records.append(orderbook_record)

        integrity = self.integrity_report(events, market_records, orderbook_records)

        return {
            "events_saved": len(events),
            "markets_saved": len(market_records),
            "orderbooks_saved": len(orderbook_records),
            "integrity": integrity,
        }

    def integrity_report(
        self,
        events: Sequence[Dict[str, Any]],
        markets: Sequence[Dict[str, Any]],
        orderbooks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a simple integrity report over events, markets and orderbooks.

        Metrics include:
        - Coverage ratios (events with markets, markets with category, markets with CLOB ids).
        - Volume and liquidity anomalies.
        - Probability vs mid-price discrepancies for markets that have both.
        """
        total_events = len(events)
        total_markets = len(markets)
        total_orderbooks = len(orderbooks)

        events_with_markets = sum(1 for e in events if _ensure_list(e.get("markets")))
        events_without_category = sum(1 for e in events if not e.get("category"))

        markets_with_category = sum(1 for m in markets if m.get("category"))
        markets_with_clob_ids = sum(1 for m in markets if _ensure_list(m.get("clob_token_ids")))
        markets_zero_volume = sum(
            1
            for m in markets
            if _to_float(m.get("volume_total")) == 0.0 and _to_float(m.get("volume_24h")) == 0.0
        )

        suspicious: List[Dict[str, Any]] = []
        ob_index: Dict[str, Dict[str, Any]] = {}
        for ob in orderbooks:
            market_id = ob.get("market_id")
            if market_id:
                ob_index[str(market_id)] = ob

        for m in markets:
            market_id = str(m.get("id"))
            yes_price = _to_float(m.get("yes_price"), default=-1.0)
            if not (0.0 <= yes_price <= 1.0):
                continue
            ob = ob_index.get(market_id)
            if not ob:
                continue
            mid = _to_float(ob.get("mid_price"), default=-1.0)
            if not (0.0 <= mid <= 1.0):
                continue
            delta = abs(yes_price - mid)
            if delta >= 0.15:
                suspicious.append(
                    {
                        "market_id": market_id,
                        "slug": m.get("slug"),
                        "question": m.get("question"),
                        "yes_price": yes_price,
                        "mid_price": mid,
                        "deviation": delta,
                    }
                )

        return {
            "events": {
                "total": total_events,
                "with_markets": events_with_markets,
                "without_markets": total_events - events_with_markets,
                "without_category": events_without_category,
            },
            "markets": {
                "total": total_markets,
                "with_category": markets_with_category,
                "without_category": total_markets - markets_with_category,
                "with_clob_ids": markets_with_clob_ids,
                "without_clob_ids": total_markets - markets_with_clob_ids,
                "zero_volume": markets_zero_volume,
            },
            "orderbooks": {
                "total": total_orderbooks,
            },
            "suspicious_pricing": {
                "count": len(suspicious),
                "examples": suspicious[:20],
            },
        }

    async def get_related_markets(
        self,
        main_slug: str,
        question_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Discover related threshold markets for a given main market.

        Strategy:
        - Try to use the event that contains the main market and inspect sibling markets.
        - If event is unavailable, fall back to text-similarity search across markets.

        Returns:
            List of dicts with (slug, threshold, market_price, market_data, question) sorted by threshold.
        """
        assert main_slug
        assert question_text

        main_market = await self.gamma.get_market(main_slug)
        related: List[Dict[str, Any]] = []

        event_id = main_market.get("eventId") or main_market.get("event_id")
        if event_id:
            events = await self.gamma.get_events(limit=1, id=event_id)
            if events:
                event = events[0]
                for market in _ensure_list(event.get("markets")):
                    normalized = self.gamma._normalize_market(market)
                    slug = normalized.get("slug", "")
                    if slug == main_slug:
                        continue
                    candidate = self._build_threshold_market(normalized)
                    if candidate is not None:
                        related.append(candidate)

        if not related:
            base = self._extract_base_question(question_text)
            all_markets = await self.gamma.get_markets(active=True, limit=500)
            for market in all_markets:
                normalized = self.gamma._normalize_market(market)
                slug = normalized.get("slug", "")
                if slug == main_slug:
                    continue
                question = normalized.get("question", "") or ""
                if base.lower() in question.lower() or question.lower() in base.lower():
                    candidate = self._build_threshold_market(normalized)
                    if candidate is not None:
                        related.append(candidate)

        related.sort(key=lambda m: m.get("threshold", 0))
        return related

    def _build_threshold_market(self, market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper to construct a threshold market descriptor if possible."""
        question = market.get("question", "") or ""
        threshold = self._extract_threshold(question)
        if threshold is None:
            for outcome in _ensure_list(market.get("outcomes")):
                threshold = self._extract_threshold(str(outcome))
                if threshold is not None:
                    break
        if threshold is None:
            return None

        yes_price, _ = self._extract_yes_no_prices(market)
        if not (0.0 <= yes_price <= 1.0):
            clob_ids = _ensure_list(market.get("clobTokenIds"))
            if len(clob_ids) >= 2 and self.clob.has_client():
                orderbook = self.clob.get_orderbook(clob_ids[1], depth=5)
                yes_price = _to_float(orderbook.get("mid_price"), default=-1.0)
        if not (0.0 <= yes_price <= 1.0):
            return None

        return {
            "slug": market.get("slug", ""),
            "threshold": threshold,
            "market_price": yes_price,
            "market_data": market,
            "question": question,
        }

    def _extract_yes_no_prices(self, market: Dict[str, Any]) -> Tuple[float, float]:
        """Extract YES/NO prices from market['outcomePrices'] with sane defaults."""
        prices = _ensure_list(market.get("outcomePrices"))
        yes = 0.5
        no = 0.5
        if len(prices) >= 2:
            no = _to_float(prices[0], default=0.5)
            yes = _to_float(prices[1], default=0.5)
        elif len(prices) == 1:
            yes = _to_float(prices[0], default=0.5)
            no = 1.0 - yes
        yes = max(0.0, min(1.0, yes))
        no = max(0.0, min(1.0, no))
        return yes, no

    def _extract_threshold(self, text: str) -> Optional[int]:
        """
        Extract a percentage threshold from free text.

        Examples:
            "30%+" -> 30
            "40% or higher" -> 40
            "Score 50%+" -> 50
        """
        if not text:
            return None

        patterns = (
            r"(\d+)%\s*\+",
            r"(\d+)\s*%\s*or\s*higher",
            r"score\s*(\d+)%",
            r"(\d+)%",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match and match.group(1):
                threshold = int(match.group(1))
                if 0 <= threshold <= 100:
                    return threshold
        return None

    def _extract_base_question(self, question_text: str) -> str:
        """
        Remove threshold-specific fragments to get a base question.

        Example:
            "Google Gemini 3 score 30%+ on Humanity's Last Exam"
            -> "Google Gemini 3 score on Humanity's Last Exam"
        """
        base = re.sub(r"\d+%\s*\+", "", question_text, flags=re.IGNORECASE)
        base = re.sub(r"\d+\s*%\s*or\s*higher", "", base, flags=re.IGNORECASE)
        base = re.sub(r"score\s*\d+%", "score", base, flags=re.IGNORECASE)
        base = re.sub(r"\s+", " ", base)
        return base.strip()

    async def _format_event(self, event: Dict[str, Any], with_orderbooks: bool) -> Dict[str, Any]:
        """Format a raw event into a compact, display-oriented JSON structure."""
        markets_raw = _ensure_list(event.get("markets"))
        formatted_markets: List[Dict[str, Any]] = []
        total_volume = 0.0

        for market in markets_raw:
            yes_price, no_price = self._extract_yes_no_prices(market)
            volume_raw = _to_float(market.get("volume_total") or market.get("volume"))
            total_volume += volume_raw

            market_dict: Dict[str, Any] = {
                "outcome": market.get("question", ""),
                "slug": market.get("slug", ""),
                "probability": round(yes_price * 100.0, 0),
                "probability_pct": f"{yes_price * 100.0:.0f}%",
                "volume": self._fmt_vol(volume_raw),
                "volume_raw": volume_raw,
                "yes_price": yes_price,
                "no_price": no_price,
                "buy_yes_display": f"{yes_price * 100.0:.1f}¢",
                "buy_no_display": f"{no_price * 100.0:.1f}¢",
                "active": market.get("active", True),
            }

            if with_orderbooks:
                clob_ids = _ensure_list(market.get("clobTokenIds"))
                if clob_ids and self.clob.has_client():
                    orderbook = self.clob.get_orderbook(clob_ids[0], depth=10)
                    market_dict["orderbook"] = {
                        "best_bid": orderbook["best_bid"],
                        "best_ask": orderbook["best_ask"],
                        "spread": f"{orderbook['spread'] * 100.0:.1f}%",
                        "liquidity": self._fmt_vol(orderbook["total_liquidity"]),
                    }

            formatted_markets.append(market_dict)

        formatted_markets.sort(key=lambda m: m.get("volume_raw", 0.0), reverse=True)

        return {
            "event": {
                "id": event.get("id", ""),
                "slug": event.get("slug", ""),
                "title": event.get("title", ""),
                "description": event.get("description", ""),
                "total_volume": self._fmt_vol(total_volume),
                "total_volume_raw": total_volume,
                "market_count": len(formatted_markets),
                "end_date": self._fmt_date(event.get("endDate")),
                "status": "active" if event.get("active", True) else "closed",
                "category": event.get("category", ""),
                "markets": formatted_markets,
            }
        }

    def _fmt_vol(self, volume: float) -> str:
        """Human-readable USD volume formatting."""
        if volume >= 1_000_000.0:
            return f"${volume / 1_000_000.0:.2f}M"
        if volume >= 1_000.0:
            return f"${volume / 1_000.0:.0f}K"
        return f"${volume:,.0f}"

    def _fmt_date(self, iso_date: Optional[str]) -> str:
        """Format ISO date string into a compact human-readable date."""
        dt = _parse_iso_datetime(iso_date)
        if dt is None:
            return "N/A"
        return dt.strftime("%b %d, %Y")