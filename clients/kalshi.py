"""
kalshi_market_extractor.py

Async utilities to fetch and enrich Kalshi prediction-market data.

- KalshiClient: minimal API client (events, markets, orderbooks, prices)
- KalshiMarketService: high-level enrichment (filters, liquidity, spreads)
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx  # type: ignore

logger = logging.getLogger(__name__)

try:
    from config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None


# =========================
# Low-level Kalshi API
# =========================
class KalshiClient:
    """Lightweight client for the Kalshi trading API (read-only).

    Auth:
      - If KALSHI_API_KEY is set -> Bearer
      - Else if KALSHI_API_KEY_ID + PRIVATE_KEY -> RSA signature headers
      - Else -> unauthenticated (public data)

    Only public reads (no trading). Suitable for enrichment pipelines.
    """

    def __init__(self) -> None:
        base = getattr(settings, "KALSHI_API_URL", "https://trading-api.kalshi.com")
        self.base_url = base.rstrip("/")
        self.api_key_id: Optional[str] = getattr(settings, "KALSHI_API_KEY_ID", None)
        self.api_key: Optional[str] = getattr(settings, "KALSHI_API_KEY", None)
        self.private_key_path: Optional[str] = getattr(settings, "KALSHI_PRIVATE_KEY_PATH", None)
        self._privkey = None
        if self.private_key_path:
            self._load_private_key()
        logger.info("KalshiClient ready")  # 1/2 module log lines

    # --- auth helpers ---
    def _load_private_key(self) -> None:
        """Load RSA private key if present (optional)."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend

            with open(self.private_key_path, "rb") as f:  # type: ignore[arg-type]
                self._privkey = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Kalshi private key not loaded: {exc}")

    def _sign_request(self, method: str, path: str, ts_ms: int) -> Optional[str]:
        """Return base64 RSA-PSS signature or None if unavailable."""
        if not self._privkey:
            return None
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            msg = f"{ts_ms}{method}{path}".encode()
            sig = self._privkey.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return base64.b64encode(sig).decode()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Sign error: {exc}")
            return None

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Build auth headers for the given request."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        if self.api_key_id and self._privkey:
            ts = int(datetime.utcnow().timestamp() * 1000)
            sig = self._sign_request(method, path, ts)
            if sig:
                return {
                    "KALSHI-ACCESS-KEY": self.api_key_id,
                    "KALSHI-ACCESS-TIMESTAMP": str(ts),
                    "KALSHI-ACCESS-SIGNATURE": sig,
                }
        return {}

    # --- HTTP helpers ---
    async def _get(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers("GET", path)
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.get(url, params=params, headers=headers)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as exc:
                logger.debug(f"GET {path} failed: {exc}")
                return None

    # --- API methods ---
    async def get_events(
        self, *, series_ticker: Optional[str] = None, status: str = "open", limit: int = 100
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = await self._get("/events", params=params) or {}
        return data.get("events", []) or []

    async def get_event(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        data = await self._get(f"/events/{event_ticker}")
        return (data or {}).get("event")

    async def get_markets(
        self,
        *,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = await self._get("/markets", params=params) or {}
        return data.get("markets", []) or []

    async def get_market(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        data = await self._get(f"/markets/{market_ticker}")
        return (data or {}).get("market")

    async def get_orderbook(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        data = await self._get(f"/markets/{market_ticker}/orderbook")
        return (data or {}).get("orderbook")

    async def get_market_price(self, market_ticker: str) -> Optional[float]:
        """Return implied YES probability in [0,1] (mid or best available)."""
        m = await self.get_market(market_ticker)
        if not m:
            return None

        # Preferred: midpoint of yes_bid_dollars / no_ask_dollars
        ybd, nad = m.get("yes_bid_dollars"), m.get("no_ask_dollars")
        if ybd is not None and nad is not None:
            return (float(ybd) + float(nad)) / 2.0

        # Fallbacks (cents -> dollars)
        def cents_to_d(v: Any) -> Optional[float]:
            return float(v) / 100.0 if v is not None else None

        for pair in (
            (cents_to_d(m.get("yes_bid")), cents_to_d(m.get("no_ask"))),
            (cents_to_d(m.get("last_price")), None),
        ):
            a, b = pair
            if a is not None and b is not None:
                return (a + b) / 2.0
            if a is not None:
                return a

        # Dollar fallbacks
        for k in ("last_price_dollars", "yes_ask_dollars", "yes_bid_dollars"):
            v = m.get(k)
            if v is not None:
                return float(v)

        return None


# =========================
# High-level enrichment
# =========================
class KalshiMarketService:
    """Enrich markets with orderbooks and computed stats (probability, liquidity, spread)."""

    def __init__(self, client: KalshiClient) -> None:
        self.client = client
        logger.info("KalshiMarketService ready")  # 2/2 module log lines

    async def get_markets(
        self,
        *,
        limit: int = 200,
        min_liquidity: float = 100.0,
        max_spread: float = 0.98,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Return up to `limit` enriched markets, sorted by volume (desc).

        Filters:
          - end_date_min / end_date_max: YYYY-MM-DD (defaults: today, +365d)
        """
        assert limit >= 1, "limit must be >= 1"
        now = datetime.utcnow()
        start_date = filters.get("end_date_min") or now.strftime("%Y-%m-%d")
        end_date = filters.get("end_date_max") or (now + timedelta(days=365)).strftime("%Y-%m-%d")

        raw = await self.client.get_markets(limit=limit * 3)
        filtered = []
        for m in raw:
            end_str = m.get("end_date") or m.get("endDate") or m.get("close_time")
            if end_str:
                try:
                    d = end_str.split("T")[0]
                    if not (start_date <= d <= end_date):
                        continue
                except (AttributeError, IndexError, ValueError) as e:
                    logger.debug(f"Failed to parse end_date '{end_str}': {e}")
            filtered.append(m)

        filtered.sort(key=lambda x: float(x.get("volumeNum", x.get("volume", 0)) or 0), reverse=True)

        tasks = [
            asyncio.create_task(self._enrich_market(m, min_liquidity, max_spread))
            for m in filtered[: limit * 10]
        ]
        enriched: List[Dict[str, Any]] = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res:
                enriched.append(res)
                if len(enriched) >= limit:
                    break
        return enriched

    async def _enrich_market(
        self, market: Dict[str, Any], min_liq: float, max_spread: float
    ) -> Optional[Dict[str, Any]]:
        """Attach orderbook + metrics; drop markets failing liquidity/spread checks."""
        ticker = market.get("market_ticker") or market.get("ticker") or market.get("id") or market.get("slug")
        if not ticker:
            return None
        ob = await self.client.get_orderbook(str(ticker))
        if not ob:
            return None

        bids, asks = ob.get("bids", []), ob.get("asks", [])
        if not bids or not asks:
            return None

        try:
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.debug(f"Failed to parse orderbook prices: {e}")
            return None

        spread = best_ask - best_bid
        if spread < 0 or spread > 1 or spread > max_spread:
            return None

        # Liquidity: min(notional on top-10 sides) * 1000 (notional dollars)
        bid_liq = sum(float(b["price"]) * float(b["size"]) for b in bids[:10])
        ask_liq = sum(float(a["price"]) * float(a["size"]) for a in asks[:10])
        total_liq = min(bid_liq, ask_liq) * 1000
        if total_liq < min_liq:
            return None

        prices = market.get("outcomePrices") or market.get("outcome_prices")
        yes_prob = float(prices[1]) if (isinstance(prices, list) and len(prices) >= 2) else best_ask
        volume_raw = float(market.get("volumeNum", market.get("volume", 0)) or 0)

        return {
            "market_slug": market.get("slug") or ticker,
            "question": market.get("question", "") or market.get("title", ""),
            "category": market.get("category", ""),
            "yes_probability": round(yes_prob * 100, 2),
            "total_liquidity": total_liq,
            "total_volume": volume_raw,
            "spread": round(spread * 100, 2),
            "spread_bps": round(spread * 10000, 0),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "midpoint_price": round((best_bid + best_ask) / 2, 4),
            "end_date": market.get("end_date") or market.get("endDate") or market.get("close_time", ""),
            "orderbook": {
                "bids": [{"price": b["price"], "size": b["size"]} for b in bids[:15]],
                "asks": [{"price": a["price"], "size": a["size"]} for a in asks[:15]],
            },
        }

    # ---------- Optional event formatter ----------
    async def format_event(self, event: Dict[str, Any], *, with_orderbooks: bool = False) -> Dict[str, Any]:
        """Return a compact event dict with aggregated volume (optionally with orderbooks)."""
        markets: List[Dict[str, Any]] = event.get("markets", []) or []
        total_vol = sum(float(m.get("volumeNum", m.get("volume", 0)) or 0) for m in markets)

        formatted: List[Dict[str, Any]] = []
        for m in markets:
            prices = m.get("outcomePrices") or m.get("outcome_prices") or []
            yes = float(prices[1]) if len(prices) >= 2 else 0.5
            no = float(prices[0]) if len(prices) >= 1 else 0.5
            md: Dict[str, Any] = {
                "outcome": m.get("question", "") or m.get("title", ""),
                "slug": m.get("slug", "") or m.get("id", ""),
                "probability": round(yes * 100, 0),
                "probability_pct": f"{yes * 100:.0f}%",
                "volume": self._fmt_vol(float(m.get("volumeNum", m.get("volume", 0)) or 0)),
                "volume_raw": float(m.get("volumeNum", m.get("volume", 0)) or 0),
                "yes_price": yes,
                "no_price": no,
                "buy_yes_display": f"{yes * 100:.1f}¢",
                "buy_no_display": f"{no * 100:.1f}¢",
                "active": m.get("active", True),
            }
            if with_orderbooks:
                tick = m.get("market_ticker") or m.get("ticker") or m.get("slug") or m.get("id")
                if tick:
                    ob = await self.client.get_orderbook(str(tick))
                    if ob:
                        bb, ba = ob.get("best_bid"), ob.get("best_ask")
                        if isinstance(bb, (int, float)) and isinstance(ba, (int, float)):
                            spr = ba - bb
                            bids, asks = ob.get("bids", []), ob.get("asks", [])
                            bid_liq = sum(float(b["price"]) * float(b["size"]) for b in bids[:10])
                            ask_liq = sum(float(a["price"]) * float(a["size"]) for a in asks[:10])
                            md["orderbook"] = {
                                "best_bid": bb,
                                "best_ask": ba,
                                "spread": f"{spr * 100:.1f}%",
                                "liquidity": self._fmt_vol(min(bid_liq, ask_liq) * 1000),
                            }
            formatted.append(md)

        formatted.sort(key=lambda x: x.get("volume_raw", 0), reverse=True)
        return {
            "event": {
                "id": event.get("id", ""),
                "slug": event.get("slug", ""),
                "title": event.get("title", ""),
                "description": event.get("description", ""),
                "total_volume": self._fmt_vol(total_vol),
                "total_volume_raw": total_vol,
                "market_count": len(formatted),
                "end_date": self._fmt_date(event.get("endDate", "")),
                "status": "active" if event.get("active", True) else "closed",
                "category": event.get("category", ""),
                "markets": formatted,
            }
        }

    # ---------- utils ----------
    @staticmethod
    def _fmt_vol(v: float) -> str:
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"{v/1_000:.2f}K"
        return f"{v:.0f}"

    @staticmethod
    def _fmt_date(s: str) -> str:
        return s.split("T")[0] if s else ""
