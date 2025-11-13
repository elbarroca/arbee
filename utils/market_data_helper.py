"""
Enhanced Polymarket Client Helper

Provides robust market data retrieval with orderbook fallback for accurate pricing.
Ensures data consistency through strict validation and defensive programming.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from clients.polymarket import PolymarketClient

logger = logging.getLogger(__name__)


def _parse_json_field(raw_value: Any) -> list:
    """Parse JSON string to list, return empty list on failure."""
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            pass
    return []


def _extract_prices_from_outcome(prices: list) -> tuple[Optional[float], Optional[float]]:
    """Extract YES/NO prices from outcomePrices, validate they sum to near 1.0."""
    if not prices or len(prices) < 2:
        return None, None

    try:
        yes_price = float(prices[1])
        no_price = float(prices[0])
        # Validate price consistency (should sum near 1.0 for binary markets)
        assert 0.9 <= yes_price + no_price <= 1.1, f"Invalid price sum: {yes_price + no_price}"
        # Ensure both prices are positive and reasonable
        assert 0 <= yes_price <= 1 and 0 <= no_price <= 1, f"Unreasonable prices: YES={yes_price}, NO={no_price}"
        return yes_price, no_price
    except (ValueError, TypeError, AssertionError):
        return None, None


def _get_orderbook_prices(client: PolymarketClient, token_id: str) -> tuple[Optional[float], float, Optional[float]]:
    """Fetch mid-price from orderbook with liquidity and spread data."""
    orderbook = client.clob.get_orderbook(token_id, depth=20)
    best_bid = orderbook.get('best_bid', 0.5)
    best_ask = orderbook.get('best_ask', 0.5)
    mid_price = orderbook.get('mid_price', (best_bid + best_ask) / 2)

    # Validate orderbook data
    assert 0 < best_bid <= best_ask < 1, f"Invalid orderbook: bid={best_bid}, ask={best_ask}"
    assert 0 < mid_price < 1, f"Invalid mid price: {mid_price}"

    liquidity = orderbook.get('total_liquidity', 0.0)
    spread = orderbook.get('spread', best_ask - best_bid)

    return mid_price, liquidity, spread


def _extract_volume(market: Dict[str, Any]) -> float:
    """Extract volume from market data, preferring volumeNum over volume."""
    volume = market.get('volumeNum') or market.get('volume') or 0
    return float(volume)


def _extract_liquidity(market: Dict[str, Any], orderbook_liquidity: float = 0.0) -> float:
    """Extract liquidity from market data, preferring orderbook over market fields."""
    if orderbook_liquidity > 0:
        return orderbook_liquidity
    liquidity = market.get('liquidityNum') or market.get('liquidity') or 0
    return float(liquidity)


def _build_market_result(market: Dict[str, Any], yes_price: Optional[float], no_price: Optional[float],
                        volume: float, liquidity: float, spread: Optional[float], price_source: str) -> Dict[str, Any]:
    """Construct standardized market result dictionary with all required fields."""
    assert yes_price is not None and no_price is not None, "Prices must be available"
    assert 0 <= yes_price <= 1 and 0 <= no_price <= 1, f"Invalid price range: YES={yes_price}, NO={no_price}"

    return {
        "market_slug": market.get('slug', ''),
        "market_question": market.get('question', ''),
        "provider": "polymarket",
        "market_id": market.get('id', market.get('_id', '')),
        "yes_price": yes_price,
        "no_price": no_price,
        "yes_probability": yes_price * 100,
        "no_probability": no_price * 100,
        "volume": volume,
        "liquidity": liquidity,
        "spread": spread if spread is not None else abs(yes_price - no_price),
        "active": market.get('active', True),
        "category": market.get('category', ''),
        "end_date": market.get('endDate', market.get('end_date', '')),
        "price_source": price_source,
    }


async def get_market_with_real_prices(client: PolymarketClient, slug: str) -> Dict[str, Any]:
    """
    Retrieve market data with accurate prices from outcomePrices or orderbook fallback.

    Ensures price consistency and validates data integrity. For binary markets,
    prices should sum to approximately 1.0. Falls back to orderbook mid-prices
    when outcomePrices are missing or invalid.

    Args:
        client: Initialized PolymarketClient instance
        slug: Market slug identifier

    Returns:
        Dict containing market data with validated prices, probabilities, volume, and liquidity

    Raises:
        AssertionError: If critical data validation fails
        ValueError: If market data cannot be retrieved or parsed
    """
    market = await client.gamma.get_market(slug)
    assert market, f"Market not found for slug: {slug}"

    # Parse and validate price data
    prices = _parse_json_field(market.get('outcomePrices', []))
    clob_token_ids = _parse_json_field(market.get('clobTokenIds', []))

    # Try outcomePrices first
    yes_price, no_price = _extract_prices_from_outcome(prices)
    price_source = "outcomePrices"

    # Fallback to orderbook if prices unavailable
    if (yes_price is None or yes_price == 0) and clob_token_ids and len(clob_token_ids) >= 2:
        yes_token = clob_token_ids[1]
        yes_price, liquidity, spread = _get_orderbook_prices(client, yes_token)
        no_price = 1.0 - yes_price if yes_price else None
        price_source = "orderbook"
        logger.info(f"Using orderbook prices for {slug}: YES={yes_price:.4f}")

    # Extract volume and liquidity
    volume = _extract_volume(market)
    liquidity = _extract_liquidity(market)

    return _build_market_result(market, yes_price, no_price, volume, liquidity, spread if 'spread' in locals() else None, price_source)


async def get_active_markets_with_prices(client: PolymarketClient, limit: int = 10) -> list[Dict[str, Any]]:
    """
    Retrieve active markets with validated prices from multiple sources.

    Filters markets to ensure only those with reliable price data are returned.
    Markets without valid prices (from outcomePrices or orderbook) are excluded.

    Args:
        client: Initialized PolymarketClient instance
        limit: Maximum number of markets to return (default: 10)

    Returns:
        List of market dictionaries with complete price and metadata validation

    Raises:
        AssertionError: If market data integrity checks fail
    """
    assert limit > 0, "Limit must be positive"

    today = datetime.now()
    future_cutoff = (today + timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch markets with broad criteria, then filter
    markets = await client.gamma.get_markets(
        active=True,
        limit=limit * 3,  # Fetch more to account for filtering
        end_date_min=today.strftime('%Y-%m-%d'),
        end_date_max=future_cutoff
    )

    validated_markets = []
    for market in markets[:limit * 2]:  # Check up to 2x limit for valid markets
        slug = market.get('slug', '')
        if not slug:
            continue

        market_data = await get_market_with_real_prices(client, slug)
        # Ensure market has valid positive prices
        if market_data.get('yes_price') and market_data['yes_price'] > 0:
            validated_markets.append(market_data)
            if len(validated_markets) >= limit:
                break

    return validated_markets