"""
polyrouter_client.py

Asynchronous client and arbitrage utilities for the PolyRouter API.

This module wraps PolyRouter's REST endpoints to list markets, events, and
series across prediction platforms (e.g., Polymarket, Kalshi). It also
provides compact aggregation and simple binary arbitrage detection.

Public API (signatures preserved from prior version):
- class PolyRouterClient
    - async list_markets(...)
    - async list_events(...)
    - async get_market_details(market_id)
    - async list_series(...)
  (New convenience methods added; originals unchanged.)
- find_arbitrage_opportunities(markets, threshold=0.0)
- aggregate_events(events_with_markets)
- aggregate_series(series_list)
- integrity_report(markets=None, events=None)

Notes
-----
Arbitrage logic is conservative and fee-agnostic: an opportunity is flagged
when (best_yes_price + best_no_price) < 1 - threshold. Consider platform fees,
slippage, liquidity, and settlement risk before acting on results.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

__all__ = [
    "PolyRouterClient",
    "find_arbitrage_opportunities",
    "aggregate_events",
    "aggregate_series",
    "integrity_report",
]

logger = logging.getLogger(__name__)


# ------------------------------- Helpers --------------------------------- #
def _as_float(x: Any) -> Optional[float]:
    """Return float(x) or None if not castable."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _price_from(prices: Dict[str, Any], side: str) -> Optional[float]:
    """
    Extract a price for a given side ('yes' or 'no') from a market price dict.
    Accepts case variations and either {'price': v} or direct numeric values.
    """
    assert side in {"yes", "no"}
    for key in (side, side.upper(), side.capitalize()):
        if key in prices:
            val = prices[key]
            if isinstance(val, dict) and "price" in val:
                return _as_float(val.get("price"))
            return _as_float(val)
    return None


def _best_two_sides(markets: Iterable[Dict[str, Any]]) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
    """
    Scan markets and return ((platform, best_yes_price), (platform, best_no_price)).
    Returns (None, None) if respective side not found.
    """
    best_yes: Optional[Tuple[str, float]] = None
    best_no: Optional[Tuple[str, float]] = None
    for m in markets:
        platform = m.get("platform")
        if not platform:
            continue
        prices = m.get("current_prices") or {}
        yp, np = _price_from(prices, "yes"), _price_from(prices, "no")
        if yp is not None and (best_yes is None or yp < best_yes[1]):
            best_yes = (platform, yp)
        if np is not None and (best_no is None or np < best_no[1]):
            best_no = (platform, np)
    return best_yes, best_no


def _event_key_for_grouping(m: Dict[str, Any]) -> Optional[str]:
    """Robust event key fallback chain for market grouping."""
    return (
        m.get("event_name")
        or m.get("title")
        or m.get("event_slug")
        or m.get("event_id")
        or m.get("series_event_id")
    )


# ------------------------------ HTTP Client ------------------------------ #
class PolyRouterClient:
    """Async client for PolyRouter REST API.

    Parameters
    ----------
    api_key : str
        PolyRouter API key (sent via 'X-API-Key' header). Required.
    base_url : str, optional
        API base URL. Default: "https://api.polyrouter.io/functions/v1".
    timeout : float, optional
        Per-request timeout in seconds. Default: 30.0.

    Notes
    -----
    - Methods with pagination helpers (`fetch_all_*`) are additive; original
      method signatures and behavior are preserved.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polyrouter.io/functions/v1",
        timeout: float = 30.0,
    ) -> None:
        assert isinstance(api_key, str) and api_key.strip(), "api_key must be a non-empty string"
        assert isinstance(base_url, str) and base_url.startswith("http"), "base_url must be an http(s) URL"
        assert timeout > 0, "timeout must be positive"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.info("PolyRouterClient initialized")

    async def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform an HTTP request. Raises RuntimeError on HTTP errors."""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-API-Key": self.api_key}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(method, url, params=params, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            logger.info("PolyRouter API request failed")  # lifecycle/failure log
            raise RuntimeError(f"PolyRouter API request failed: {exc}") from exc

    # ------------------------------ Markets ------------------------------ #
    async def list_markets(
        self,
        platform: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return a page of markets.

        Parameters
        ----------
        platform : Optional[str]
            Filter to a single platform (e.g., "polymarket", "kalshi").
        status : str
            Market status (e.g., "open", "closed", "settled"). Default "open".
        limit : int
            Page size (API-dependent max). Default 100.
        offset : int
            Pagination offset. Default 0.

        Returns
        -------
        List[Dict[str, Any]]
            Normalized market dicts; schema is API-controlled.
        """
        assert limit >= 0 and offset >= 0, "limit/offset must be non-negative"
        params: Dict[str, Any] = {"status": status, "limit": limit, "offset": offset}
        if platform:
            params["platform"] = platform
        result = await self._request("GET", "/markets", params=params)
        return result.get("markets", [])  # type: ignore[return-value]

    async def fetch_all_markets(
        self,
        platform: Optional[str] = None,
        status: str = "open",
        page_size: int = 500,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all markets by following pagination until exhausted.

        Parameters
        ----------
        platform : Optional[str]
            Filter to a single platform or None for all.
        status : str
            Market status filter.
        page_size : int
            Items per page (bounded by API). Default 500.
        max_pages : Optional[int]
            Safety cap on pages followed; None disables. Default None.

        Returns
        -------
        List[Dict[str, Any]]
            All collected markets.
        """
        assert page_size > 0 and (max_pages is None or max_pages > 0)
        out: List[Dict[str, Any]] = []
        pages, offset = 0, 0
        while True:
            params: Dict[str, Any] = {"status": status, "limit": page_size, "offset": offset}
            if platform:
                params["platform"] = platform
            result = await self._request("GET", "/markets", params=params)
            items = result.get("markets", [])
            out.extend(items if isinstance(items, list) else [])
            next_offset = result.get("next_offset")
            pages += 1
            if not isinstance(next_offset, int) or next_offset <= offset:
                break
            if max_pages and pages >= max_pages:
                break
            offset = next_offset
        return out

    async def get_market_details(self, market_id: str) -> Dict[str, Any]:
        """Return a single market's detailed info (orderbook, volume, raw data)."""
        assert isinstance(market_id, str) and market_id.strip(), "market_id must be a non-empty string"
        data = await self._request("GET", f"/markets/{market_id}")
        return data.get("market", {})  # type: ignore[return-value]

    # ------------------------------- Events ------------------------------- #
    async def list_events(
        self,
        platform: Optional[str] = None,
        include_raw: bool = False,
        with_nested_markets: bool = False,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return a page of events (optionally with nested markets).

        Parameters
        ----------
        platform : Optional[str]
            Filter to a single platform or None for all.
        include_raw : bool
            Include upstream raw payloads.
        with_nested_markets : bool
            Attach markets for each event (API-permitting).
        limit : int
            Items per page.
        cursor : Optional[str]
            Pagination cursor for next page.

        Returns
        -------
        List[Dict[str, Any]]
            Event objects; may include a 'markets' key if requested.
        """
        assert limit >= 0, "limit must be non-negative"
        params: Dict[str, Any] = {"limit": limit}
        if platform:
            params["platform"] = platform
        if include_raw:
            params["include_raw"] = True
        if with_nested_markets:
            params["with_nested_markets"] = True
        if cursor:
            params["cursor"] = cursor
        result = await self._request("GET", "/events", params=params)
        return result.get("events", [])  # type: ignore[return-value]

    async def fetch_all_events(
        self,
        platform: Optional[str] = None,
        include_raw: bool = False,
        with_nested_markets: bool = True,
        page_size: int = 100,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all events by following cursors until exhausted."""
        assert page_size > 0 and (max_pages is None or max_pages > 0)
        out: List[Dict[str, Any]] = []
        pages, cursor = 0, None
        while True:
            params: Dict[str, Any] = {"limit": page_size}
            if platform:
                params["platform"] = platform
            if include_raw:
                params["include_raw"] = True
            if with_nested_markets:
                params["with_nested_markets"] = True
            if cursor:
                params["cursor"] = cursor
            result = await self._request("GET", "/events", params=params)
            items = result.get("events", [])
            out.extend(items if isinstance(items, list) else [])
            cursor = result.get("next_cursor") or result.get("cursor") or result.get("next")  # tolerant
            pages += 1
            if not cursor or (max_pages and pages >= max_pages):
                break
        return out

    # ------------------------------- Series ------------------------------- #
    async def list_series(
        self,
        platform: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return a page of series (groupings of related events/markets)."""
        assert limit >= 0 and offset >= 0, "limit/offset must be non-negative"
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if platform:
            params["platform"] = platform
        result = await self._request("GET", "/series", params=params)
        return result.get("series", [])  # type: ignore[return-value]

    async def fetch_all_series(
        self,
        platform: Optional[str] = None,
        page_size: int = 100,
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all series by paginating via offset."""
        assert page_size > 0 and (max_pages is None or max_pages > 0)
        out: List[Dict[str, Any]] = []
        pages, offset = 0, 0
        while True:
            params: Dict[str, Any] = {"limit": page_size, "offset": offset}
            if platform:
                params["platform"] = platform
            result = await self._request("GET", "/series", params=params)
            items = result.get("series", [])
            out.extend(items if isinstance(items, list) else [])
            next_offset = result.get("next_offset")
            pages += 1
            if not isinstance(next_offset, int) or next_offset <= offset:
                break
            if max_pages and pages >= max_pages:
                break
            offset = next_offset
        return out


# --------------------------- Aggregation & Arb --------------------------- #
def find_arbitrage_opportunities(markets: List[Dict[str, Any]], threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Identify simple binary arbitrage opportunities across platforms.

    Logic
    -----
    1) Group markets by their event key.
    2) For each group, take the minimum 'yes' and minimum 'no' price across platforms.
    3) If (yes + no) < 1 - threshold, record an opportunity.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        Market objects (e.g., from client.list_markets or fetch_all_markets).
    threshold : float, default 0.0
        Minimum required margin (e.g., 0.02 requires at least 2% cushion).

    Returns
    -------
    List[Dict[str, Any]]
        Sorted (desc by margin) opportunity dicts with:
        - event_name
        - best_yes_platform, best_yes_price
        - best_no_platform,  best_no_price
        - margin (1 - yes - no)
    """
    assert 0.0 <= threshold < 1.0, "threshold must be in [0, 1)"
    events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in markets:
        k = _event_key_for_grouping(m)
        if k:
            events[k].append(m)

    out: List[Dict[str, Any]] = []
    for event_name, group in events.items():
        best_yes, best_no = _best_two_sides(group)
        if best_yes and best_no:
            margin = 1.0 - (best_yes[1] + best_no[1])
            if margin > threshold:
                out.append(
                    {
                        "event_name": event_name,
                        "best_yes_platform": best_yes[0],
                        "best_yes_price": best_yes[1],
                        "best_no_platform": best_no[0],
                        "best_no_price": best_no[1],
                        "margin": margin,
                    }
                )
    out.sort(key=lambda x: x["margin"], reverse=True)
    return out


def _aggregate_single_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate one event's markets and compute best-side arbitrage summary.

    Returns a dict:
    {
      'id','title','event_slug','event_start_at','event_end_at','total_volume',
      'markets': [{'platform','id','slug','title','yes_price','no_price',
                   'volume_total','liquidity','market_type'}, ...],
      'arbitrage': {'best_yes_platform','best_yes_price','best_no_platform',
                    'best_no_price','margin'} | None
    }
    """
    markets = event.get("markets", []) or []
    market_summaries: List[Dict[str, Any]] = []
    for m in markets:
        prices = m.get("current_prices") or {}
        yes_price = _price_from(prices, "yes")
        no_price = _price_from(prices, "no")
        market_summaries.append(
            {
                "platform": m.get("platform"),
                "id": m.get("id") or m.get("market_id"),
                "slug": m.get("market_slug") or m.get("slug"),
                "title": m.get("title") or m.get("market_title"),
                "yes_price": yes_price,
                "no_price": no_price,
                "volume_total": m.get("volume_total"),
                "liquidity": m.get("liquidity"),
                "market_type": m.get("market_type"),
            }
        )

    best_yes, best_no = _best_two_sides(markets)
    arbitrage = None
    if best_yes and best_no:
        arbitrage = {
            "best_yes_platform": best_yes[0],
            "best_yes_price": best_yes[1],
            "best_no_platform": best_no[0],
            "best_no_price": best_no[1],
            "margin": 1.0 - (best_yes[1] + best_no[1]),
        }

    return {
        "id": event.get("id"),
        "title": event.get("title"),
        "event_slug": event.get("event_slug"),
        "event_start_at": event.get("event_start_at") or event.get("event_start"),
        "event_end_at": event.get("event_end_at") or event.get("event_end"),
        "total_volume": event.get("total_volume") or 0,
        "markets": market_summaries,
        "arbitrage": arbitrage,
    }


def aggregate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate events (with nested markets) and attach per-event arbitrage."""
    assert isinstance(events, list), "events must be a list"
    return [_aggregate_single_event(e) for e in events]


def aggregate_series(series_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate series by grouping contained markets under their events."""
    assert isinstance(series_list, list), "series_list must be a list"
    aggregated: List[Dict[str, Any]] = []
    for series in series_list:
        markets_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m in series.get("markets", []) or []:
            eid = m.get("event_id")
            if eid:
                markets_by_event[eid].append(m)
        for ev in series.get("events", []) or []:
            eid = ev.get("id")
            if not eid:
                continue
            synthetic = dict(ev)
            synthetic["markets"] = markets_by_event.get(eid, [])
            aggregated.append(_aggregate_single_event(synthetic))
    return aggregated


# ------------------------------ Integrity -------------------------------- #
def integrity_report(
    markets: Optional[List[Dict[str, Any]]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    arb_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Compute compact data-quality and risk metrics.

    Parameters
    ----------
    markets : Optional[List[Dict[str, Any]]]
        Flat market list (e.g., from list/fetch_all_markets).
    events : Optional[List[Dict[str, Any]]]
        Event list with nested markets (e.g., from list/fetch_all_events).
    arb_threshold : float, default 0.0
        Margin threshold for counting arbitrage opportunities.

    Returns
    -------
    Dict[str, Any]
        {
          'totals': {'events','markets'},
          'platform_coverage': {'<platform>': count, ...},
          'price_coverage': {'yes_pct','no_pct','both_pct'},
          'arbitrage': {'count','sample':[... up to 5]},
          'violations': [str, ...]  # non-fatal data issues
        }
    """
    assert 0.0 <= arb_threshold < 1.0
    flat_markets: List[Dict[str, Any]] = []
    if markets:
        flat_markets.extend(markets)
    if events:
        for e in events:
            flat_markets.extend(e.get("markets", []) or [])
    total_markets = len(flat_markets)
    total_events = len(events or [])

    platform_counts: Dict[str, int] = defaultdict(int)
    yes_present = no_present = both_present = 0
    violations: List[str] = []

    for m in flat_markets:
        p = m.get("platform")
        if p:
            platform_counts[p] += 1
        prices = m.get("current_prices") or {}
        yp = _price_from(prices, "yes")
        np = _price_from(prices, "no")
        yes_present += int(yp is not None)
        no_present += int(np is not None)
        both_present += int(yp is not None and np is not None)
        if yp is not None and not (0.0 <= yp <= 1.0):
            violations.append("yes_price_out_of_bounds")
        if np is not None and not (0.0 <= np <= 1.0):
            violations.append("no_price_out_of_bounds")

    arbs = find_arbitrage_opportunities(flat_markets, threshold=arb_threshold) if flat_markets else []
    sample = arbs[:5]

    pct = (lambda n: (n / total_markets * 100.0) if total_markets else 0.0)
    return {
        "totals": {"events": total_events, "markets": total_markets},
        "platform_coverage": dict(platform_counts),
        "price_coverage": {
            "yes_pct": round(pct(yes_present), 2),
            "no_pct": round(pct(no_present), 2),
            "both_pct": round(pct(both_present), 2),
        },
        "arbitrage": {"count": len(arbs), "sample": sample},
        "violations": violations,
    }