"""
Token-to-Market Resolver for Polymarket
Maps CTF token IDs to market slugs using Gamma API.
"""
import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from config.settings import settings

logger = logging.getLogger(__name__)


class TokenResolverCache:
    """
    Simple in-memory cache for token-to-market mappings.

    Cache structure:
        {token_id: {"market_slug": str, "market_data": dict, "expires_at": datetime}}
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 5 minutes)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached market data for a token ID.

        Args:
            token_id: CTF token ID

        Returns:
            Cached market data or None if not found/expired
        """
        if token_id not in self.cache:
            return None

        entry = self.cache[token_id]
        if datetime.utcnow() > entry["expires_at"]:
            # Expired, remove from cache
            del self.cache[token_id]
            return None

        return entry

    def set(self, token_id: str, market_slug: str, market_data: Dict[str, Any]) -> None:
        """
        Store market data in cache.

        Args:
            token_id: CTF token ID
            market_slug: Polymarket market slug
            market_data: Full market metadata from Gamma API
        """
        self.cache[token_id] = {
            "market_slug": market_slug,
            "market_data": market_data,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        }
        logger.debug(f"Cached token {token_id} -> market {market_slug}")

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Token resolver cache cleared")

    def size(self) -> int:
        """Return number of entries in cache"""
        return len(self.cache)


class TokenResolver:
    """
    Resolves CTF token IDs to Polymarket market slugs.

    Uses Gamma API to query markets and find matching token IDs.
    Implements caching to reduce API calls.
    """

    def __init__(
        self,
        gamma_url: Optional[str] = None,
        cache_ttl: int = 300,
        timeout: float = 10.0
    ):
        """
        Initialize token resolver.

        Args:
            gamma_url: Gamma API base URL (default from settings)
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
            timeout: HTTP request timeout in seconds
        """
        self.gamma_url = gamma_url or settings.POLYMARKET_GAMMA_URL
        self.timeout = timeout
        self.cache = TokenResolverCache(ttl_seconds=cache_ttl)

    async def resolve_token_to_market(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a CTF token ID to a market slug and metadata.

        Args:
            token_id: CTF token ID (without 0x prefix or with)

        Returns:
            Dict with keys:
                - market_slug: str
                - market_id: str
                - question: str
                - active: bool
                - clobTokenIds: List[str]
                - tokens: List[Dict] (with side, outcome, etc.)
                - ... (other market metadata)
            or None if not found
        """
        # Normalize token ID (remove 0x prefix if present)
        token_id = token_id.lower().replace("0x", "")

        # Check cache first
        cached = self.cache.get(token_id)
        if cached:
            logger.debug(f"Cache hit for token {token_id}")
            return cached["market_data"]

        # Query Gamma API
        try:
            market_data = await self._query_gamma_for_token(token_id)

            if market_data:
                # Cache the result
                market_slug = market_data.get("slug") or market_data.get("id")
                self.cache.set(token_id, market_slug, market_data)
                logger.info(f"Resolved token {token_id} -> market {market_slug}")
                return market_data
            else:
                logger.warning(f"Token {token_id} not found in any market")
                return None
        except Exception as e:
            logger.error(f"Error resolving token {token_id}: {e}", exc_info=True)
            return None

    async def _query_gamma_for_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Query Gamma API to find market containing the token ID.

        Strategy:
        1. Search active markets
        2. Check clobTokenIds for match
        3. Return first matching market

        Args:
            token_id: Normalized token ID (no 0x prefix)

        Returns:
            Market metadata or None
        """
        url = f"{self.gamma_url}/markets"

        # Query parameters: get active markets, limit to reasonable number
        params = {
            "active": "true",
            "closed": "false",
            "limit": 100  # Adjust if needed
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # Gamma API returns list of markets
                markets = data if isinstance(data, list) else data.get("data", [])

                # Search for matching token ID
                for market in markets:
                    clob_token_ids = market.get("clobTokenIds", [])

                    # Normalize all clob token IDs
                    normalized_clob_ids = [
                        tid.lower().replace("0x", "") for tid in clob_token_ids
                    ]

                    if token_id in normalized_clob_ids:
                        # Found a match!
                        logger.info(f"Found market {market.get('slug')} for token {token_id}")

                        # Enrich with token details
                        tokens = market.get("tokens", [])
                        token_index = normalized_clob_ids.index(token_id)

                        if token_index < len(tokens):
                            token_details = tokens[token_index]
                            market["matched_token"] = {
                                "index": token_index,
                                "outcome": token_details.get("outcome"),
                                "side": "YES" if token_details.get("outcome") == market.get("outcomes", ["YES", "NO"])[0] else "NO",
                                "token_id": f"0x{token_id}"
                            }

                        return market

                # If we're here, token not found in active markets
                # Try querying all markets (could be slow, use sparingly)
                logger.debug(f"Token {token_id} not found in active markets, trying all markets")
                return await self._query_all_markets_for_token(token_id)

            except httpx.HTTPError as e:
                logger.error(f"HTTP error querying Gamma API: {e}")
                return None

    async def _query_all_markets_for_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback: Query all markets (active and closed) for token.

        This is slower but ensures we find the market even if it's recently closed.

        Args:
            token_id: Normalized token ID

        Returns:
            Market metadata or None
        """
        url = f"{self.gamma_url}/markets"

        params = {
            "limit": 500  # Larger limit for comprehensive search
        }

        async with httpx.AsyncClient(timeout=self.timeout * 2) as client:  # Double timeout
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                markets = data if isinstance(data, list) else data.get("data", [])

                for market in markets:
                    clob_token_ids = market.get("clobTokenIds", [])
                    normalized_clob_ids = [
                        tid.lower().replace("0x", "") for tid in clob_token_ids
                    ]

                    if token_id in normalized_clob_ids:
                        logger.info(f"Found market {market.get('slug')} in all markets query")
                        tokens = market.get("tokens", [])
                        token_index = normalized_clob_ids.index(token_id)

                        if token_index < len(tokens):
                            token_details = tokens[token_index]
                            market["matched_token"] = {
                                "index": token_index,
                                "outcome": token_details.get("outcome"),
                                "side": "YES" if token_details.get("outcome") == market.get("outcomes", ["YES", "NO"])[0] else "NO",
                                "token_id": f"0x{token_id}"
                            }

                        return market

                return None
            except httpx.HTTPError as e:
                logger.error(f"HTTP error in all markets query: {e}")
                return None

    async def resolve_multiple_tokens(self, token_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Resolve multiple token IDs to markets (batch operation).

        Args:
            token_ids: List of CTF token IDs

        Returns:
            Dict mapping token_id -> market_data (or None if not found)
        """
        import asyncio

        tasks = [self.resolve_token_to_market(token_id) for token_id in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        resolved = {}
        for token_id, result in zip(token_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Error resolving token {token_id}: {result}")
                resolved[token_id] = None
            else:
                resolved[token_id] = result

        return resolved

    async def is_market_active(self, market_slug: str) -> bool:
        """
        Check if a market is currently active.

        Args:
            market_slug: Market slug or ID

        Returns:
            True if market is active, False otherwise
        """
        url = f"{self.gamma_url}/markets/{market_slug}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                market = response.json()

                return market.get("active", False) and not market.get("closed", True)
            except httpx.HTTPError as e:
                logger.error(f"Error checking market status for {market_slug}: {e}")
                return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size and hit/miss stats
        """
        return {
            "size": self.cache.size(),
            "ttl_seconds": self.cache.ttl_seconds
        }


# Singleton instance for global use
_resolver: Optional[TokenResolver] = None


def get_token_resolver(
    gamma_url: Optional[str] = None,
    cache_ttl: int = 300
) -> TokenResolver:
    """
    Get or create the global TokenResolver instance.

    Args:
        gamma_url: Gamma API URL (only used on first call)
        cache_ttl: Cache TTL in seconds (only used on first call)

    Returns:
        TokenResolver singleton instance
    """
    global _resolver
    if _resolver is None:
        _resolver = TokenResolver(gamma_url=gamma_url, cache_ttl=cache_ttl)
    return _resolver


def reset_token_resolver() -> None:
    """Reset the global TokenResolver instance (primarily for tests)"""
    global _resolver
    _resolver = None
