"""
Market Data Caching Infrastructure

In-memory cache with TTL for market data providers.
Reduces redundant API calls and improves performance.
"""
import asyncio
import time
import hashlib
import json
import logging
from typing import Any, Optional, Dict, Callable, Tuple
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata"""
    value: Any
    created_at: float
    ttl: int  # seconds
    hits: int = 0
    provider: str = "unknown"

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return (time.time() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics"""
    provider: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_sets: int = 0
    cache_evictions: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0


class MarketDataCache:
    """
    In-memory cache for market data with TTL and statistics.

    Features:
    - Configurable TTL per data type (events, markets, prices, orderbooks)
    - Automatic eviction of expired entries
    - Cache statistics per provider
    - Thread-safe operations
    - Max size limit with LRU eviction
    """

    def __init__(
        self,
        *,
        events_ttl: int = 300,  # 5 minutes
        markets_ttl: int = 60,  # 1 minute
        prices_ttl: int = 10,  # 10 seconds
        orderbook_ttl: int = 5,  # 5 seconds
        max_size: int = 10000,  # Maximum cache entries
        eviction_check_interval: int = 60  # Check for expired entries every 60s
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._stats: Dict[str, CacheStats] = {}
        self._lock = asyncio.Lock()

        # TTL configuration
        self.ttl_config = {
            "events": events_ttl,
            "event": events_ttl,
            "markets": markets_ttl,
            "market": markets_ttl,
            "price": prices_ttl,
            "orderbook": orderbook_ttl,
            "default": 60
        }

        self.max_size = max_size
        self.eviction_check_interval = eviction_check_interval
        self._last_eviction_check = time.time()

        logger.info(
            f"MarketDataCache initialized: "
            f"events_ttl={events_ttl}s, markets_ttl={markets_ttl}s, "
            f"prices_ttl={prices_ttl}s, orderbook_ttl={orderbook_ttl}s, "
            f"max_size={max_size}"
        )

    def _get_cache_key(
        self,
        provider: str,
        data_type: str,
        *args,
        **kwargs
    ) -> str:
        """
        Generate cache key from provider, data type, and parameters.

        Examples:
            polymarket:markets:active=True:limit=100
            kalshi:market:BTC-100K-2025
            polymarket:orderbook:0x123abc
        """
        # Build key components
        key_parts = [provider, data_type]

        # Add args
        for arg in args:
            key_parts.append(str(arg))

        # Add kwargs (sorted for consistency)
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            # Skip None values
            if v is not None:
                key_parts.append(f"{k}={v}")

        key_str = ":".join(key_parts)

        # Hash long keys to prevent memory bloat
        if len(key_str) > 200:
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"{provider}:{data_type}:{key_hash}"

        return key_str

    def _get_ttl(self, data_type: str) -> int:
        """Get TTL for data type"""
        return self.ttl_config.get(data_type, self.ttl_config["default"])

    def _get_stats(self, provider: str) -> CacheStats:
        """Get or create stats for provider"""
        if provider not in self._stats:
            self._stats[provider] = CacheStats(provider=provider)
        return self._stats[provider]

    async def get(
        self,
        provider: str,
        data_type: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            provider: Provider name (polymarket, kalshi, etc.)
            data_type: Data type (events, market, price, orderbook)
            *args: Positional args for cache key
            **kwargs: Keyword args for cache key

        Returns:
            Cached value or None if not found/expired
        """
        key = self._get_cache_key(provider, data_type, *args, **kwargs)
        stats = self._get_stats(provider)

        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                stats.cache_misses += 1
                logger.debug(f"Cache MISS: {key}")
                return None

            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                stats.cache_misses += 1
                stats.cache_evictions += 1
                logger.debug(f"Cache EXPIRED: {key} (age: {entry.age_seconds:.1f}s)")
                return None

            # Cache hit
            entry.hits += 1
            stats.cache_hits += 1
            logger.debug(
                f"Cache HIT: {key} (age: {entry.age_seconds:.1f}s, hits: {entry.hits})"
            )
            return entry.value

    async def set(
        self,
        provider: str,
        data_type: str,
        value: Any,
        *args,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Set value in cache.

        Args:
            provider: Provider name
            data_type: Data type
            value: Value to cache
            *args: Positional args for cache key
            ttl: Custom TTL (overrides default)
            **kwargs: Keyword args for cache key
        """
        key = self._get_cache_key(provider, data_type, *args, **kwargs)
        stats = self._get_stats(provider)

        if ttl is None:
            ttl = self._get_ttl(data_type)

        async with self._lock:
            # Check max size and evict if needed
            if len(self._cache) >= self.max_size:
                await self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl,
                provider=provider
            )
            stats.cache_sets += 1

            logger.debug(f"Cache SET: {key} (ttl: {ttl}s)")

    async def invalidate(
        self,
        provider: Optional[str] = None,
        data_type: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            provider: If specified, only invalidate this provider's entries
            data_type: If specified, only invalidate this data type

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if provider is None and data_type is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cache invalidated: {count} entries")
                return count

            # Selective invalidation
            keys_to_remove = []
            for key in self._cache.keys():
                parts = key.split(":")
                if len(parts) < 2:
                    continue

                key_provider = parts[0]
                key_data_type = parts[1]

                should_remove = True
                if provider and key_provider != provider:
                    should_remove = False
                if data_type and key_data_type != data_type:
                    should_remove = False

                if should_remove:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            logger.info(
                f"Cache invalidated: {len(keys_to_remove)} entries "
                f"(provider={provider}, data_type={data_type})"
            )
            return len(keys_to_remove)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return

        # Find entry with lowest hits and oldest
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hits, -self._cache[k].created_at)
        )

        entry = self._cache[lru_key]
        stats = self._get_stats(entry.provider)
        stats.cache_evictions += 1

        del self._cache[lru_key]
        logger.debug(f"Cache LRU eviction: {lru_key}")

    async def evict_expired(self) -> int:
        """
        Evict all expired entries.

        Returns:
            Number of entries evicted
        """
        async with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in keys_to_remove:
                entry = self._cache[key]
                stats = self._get_stats(entry.provider)
                stats.cache_evictions += 1
                del self._cache[key]

            if keys_to_remove:
                logger.info(f"Cache expired eviction: {len(keys_to_remove)} entries")

            return len(keys_to_remove)

    async def _maybe_evict_expired(self) -> None:
        """Check and evict expired entries if interval elapsed"""
        now = time.time()
        if (now - self._last_eviction_check) >= self.eviction_check_interval:
            await self.evict_expired()
            self._last_eviction_check = now

    def get_stats(self, provider: Optional[str] = None) -> Dict[str, CacheStats]:
        """
        Get cache statistics.

        Args:
            provider: If specified, get stats for specific provider

        Returns:
            Dict of provider -> CacheStats
        """
        if provider:
            return {provider: self._get_stats(provider)}
        return dict(self._stats)

    def get_size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

    def get_info(self) -> Dict[str, Any]:
        """Get cache info"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization_pct": (len(self._cache) / self.max_size * 100),
            "ttl_config": self.ttl_config,
            "providers": list(self._stats.keys())
        }


# ========== Decorator ==========

def cache_result(
    data_type: str,
    *,
    ttl: Optional[int] = None,
    key_args: Optional[Tuple[str, ...]] = None
):
    """
    Decorator to cache async function results.

    Usage:
        @cache_result("markets", ttl=60, key_args=("active", "limit"))
        async def get_markets(self, active=True, limit=100):
            ...

    Args:
        data_type: Data type for cache key and TTL lookup
        ttl: Custom TTL (overrides default for data_type)
        key_args: Tuple of argument names to include in cache key
                  (default: all args)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get cache from instance (assumes self.cache exists)
            cache: Optional[MarketDataCache] = getattr(self, "_cache", None)
            if cache is None:
                # No cache available, call function directly
                return await func(self, *args, **kwargs)

            # Get provider name (assumes self.provider_name exists)
            provider = getattr(self, "provider_name", "unknown")

            # Build cache key kwargs
            if key_args:
                # Use only specified args
                cache_kwargs = {k: kwargs[k] for k in key_args if k in kwargs}
            else:
                # Use all kwargs
                cache_kwargs = kwargs

            # Try to get from cache
            start_time = time.time()
            cached = await cache.get(provider, data_type, *args, **cache_kwargs)

            if cached is not None:
                latency_ms = (time.time() - start_time) * 1000
                stats = cache._get_stats(provider)
                stats.total_requests += 1
                stats.total_latency_ms += latency_ms
                return cached

            # Cache miss - call function
            try:
                result = await func(self, *args, **kwargs)

                # Store in cache
                await cache.set(
                    provider,
                    data_type,
                    result,
                    *args,
                    ttl=ttl,
                    **cache_kwargs
                )

                latency_ms = (time.time() - start_time) * 1000
                stats = cache._get_stats(provider)
                stats.total_requests += 1
                stats.total_latency_ms += latency_ms

                return result

            except Exception as e:
                stats = cache._get_stats(provider)
                stats.errors += 1
                stats.total_requests += 1
                raise

        return wrapper
    return decorator


# ========== Global Cache Instance ==========

# Shared cache instance for all providers
_global_cache: Optional[MarketDataCache] = None


def get_cache() -> MarketDataCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketDataCache()
    return _global_cache


def set_cache(cache: MarketDataCache) -> None:
    """Set global cache instance (for testing/customization)"""
    global _global_cache
    _global_cache = cache
