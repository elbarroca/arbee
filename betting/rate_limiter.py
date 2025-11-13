"""
Rate Limiting Infrastructure

Token bucket algorithm for API rate limiting per provider.
Prevents exceeding provider rate limits and automatic backoff.
"""
import asyncio
import time
import logging
from typing import Dict, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a provider"""
    requests_per_second: float
    burst_size: int  # Max tokens (burst capacity)
    timeout: float = 30.0  # Max wait time for token

    @property
    def refill_rate(self) -> float:
        """Tokens added per second"""
        return self.requests_per_second


@dataclass
class RateLimitStats:
    """Rate limit statistics"""
    provider: str
    total_requests: int = 0
    throttled_requests: int = 0
    total_wait_time_ms: float = 0.0
    rate_limit_hits: int = 0
    last_reset: float = time.time()

    @property
    def throttle_rate(self) -> float:
        """Percentage of requests throttled"""
        return (
            self.throttled_requests / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

    @property
    def avg_wait_time_ms(self) -> float:
        """Average wait time per throttled request"""
        return (
            self.total_wait_time_ms / self.throttled_requests
            if self.throttled_requests > 0
            else 0.0
        )


class TokenBucket:
    """
    Token bucket for rate limiting.

    Tokens are added at a constant rate (refill_rate).
    Each request consumes 1 token.
    If no tokens available, request waits until token is available.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = float(config.burst_size)  # Start full
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

        logger.debug(
            f"TokenBucket initialized: {config.requests_per_second} req/s, "
            f"burst={config.burst_size}"
        )

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Max wait time (None = use config default)

        Returns:
            True if acquired, False if timeout

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        if timeout is None:
            timeout = self.config.timeout

        start_time = time.time()

        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_refill
                refill_amount = elapsed * self.config.refill_rate

                self.tokens = min(
                    self.tokens + refill_amount,
                    self.config.burst_size
                )
                self.last_refill = now

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time for next token
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.config.refill_rate

                # Check timeout
                elapsed_total = time.time() - start_time
                if elapsed_total + wait_time > timeout:
                    raise asyncio.TimeoutError(
                        f"Rate limit timeout after {elapsed_total:.2f}s"
                    )

                # Wait for tokens to refill
                logger.debug(
                    f"Rate limit: waiting {wait_time:.2f}s for {tokens_needed:.1f} tokens"
                )
                await asyncio.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """Get current number of available tokens"""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.config.refill_rate

        return min(
            self.tokens + refill_amount,
            self.config.burst_size
        )

    def reset(self) -> None:
        """Reset bucket to full"""
        self.tokens = float(self.config.burst_size)
        self.last_refill = time.time()


class RateLimiter:
    """
    Multi-provider rate limiter with statistics.

    Manages token buckets for multiple providers.
    """

    # Default rate limits per provider
    DEFAULT_LIMITS = {
        "polymarket": RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20
        ),
        "kalshi": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10
        ),
        "valyu": RateLimitConfig(
            requests_per_second=2.0,
            burst_size=5
        ),
        "default": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10
        )
    }

    def __init__(self, custom_limits: Optional[Dict[str, RateLimitConfig]] = None):
        self._buckets: Dict[str, TokenBucket] = {}
        self._stats: Dict[str, RateLimitStats] = {}
        self._custom_limits = custom_limits or {}

        logger.info("RateLimiter initialized")

    def _get_bucket(self, provider: str) -> TokenBucket:
        """Get or create token bucket for provider"""
        if provider not in self._buckets:
            # Get config (custom or default)
            config = (
                self._custom_limits.get(provider) or
                self.DEFAULT_LIMITS.get(provider) or
                self.DEFAULT_LIMITS["default"]
            )

            self._buckets[provider] = TokenBucket(config)
            logger.info(
                f"Created rate limiter for {provider}: "
                f"{config.requests_per_second} req/s, burst={config.burst_size}"
            )

        return self._buckets[provider]

    def _get_stats(self, provider: str) -> RateLimitStats:
        """Get or create stats for provider"""
        if provider not in self._stats:
            self._stats[provider] = RateLimitStats(provider=provider)
        return self._stats[provider]

    async def acquire(
        self,
        provider: str,
        tokens: int = 1,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire tokens for provider.

        Args:
            provider: Provider name
            tokens: Number of tokens to acquire
            timeout: Max wait time

        Returns:
            True if acquired

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        bucket = self._get_bucket(provider)
        stats = self._get_stats(provider)

        stats.total_requests += 1
        start_time = time.time()

        try:
            # Check if we need to wait
            available = bucket.get_available_tokens()
            if available < tokens:
                stats.throttled_requests += 1

            # Acquire token (may wait)
            result = await bucket.acquire(tokens, timeout)

            # Track wait time
            wait_time_ms = (time.time() - start_time) * 1000
            if wait_time_ms > 10:  # Only count significant waits
                stats.total_wait_time_ms += wait_time_ms
                logger.debug(
                    f"Rate limit wait: {provider} waited {wait_time_ms:.0f}ms"
                )

            return result

        except asyncio.TimeoutError:
            stats.rate_limit_hits += 1
            logger.warning(
                f"Rate limit timeout for {provider} after "
                f"{time.time() - start_time:.2f}s"
            )
            raise

    def get_available_tokens(self, provider: str) -> float:
        """Get available tokens for provider"""
        bucket = self._get_bucket(provider)
        return bucket.get_available_tokens()

    def reset(self, provider: Optional[str] = None) -> None:
        """
        Reset rate limiter.

        Args:
            provider: If specified, reset only this provider
        """
        if provider:
            if provider in self._buckets:
                self._buckets[provider].reset()
                logger.info(f"Reset rate limiter for {provider}")
        else:
            for bucket in self._buckets.values():
                bucket.reset()
            logger.info("Reset all rate limiters")

    def get_stats(self, provider: Optional[str] = None) -> Dict[str, RateLimitStats]:
        """
        Get rate limit statistics.

        Args:
            provider: If specified, get stats for specific provider

        Returns:
            Dict of provider -> RateLimitStats
        """
        if provider:
            return {provider: self._get_stats(provider)}
        return dict(self._stats)

    def get_info(self) -> Dict[str, any]:
        """Get rate limiter info"""
        return {
            "providers": list(self._buckets.keys()),
            "limits": {
                provider: {
                    "requests_per_second": bucket.config.requests_per_second,
                    "burst_size": bucket.config.burst_size,
                    "available_tokens": bucket.get_available_tokens()
                }
                for provider, bucket in self._buckets.items()
            }
        }


# ========== Decorator ==========

def rate_limit(provider_attr: str = "provider_name"):
    """
    Decorator to rate limit async function calls.

    Usage:
        @rate_limit()
        async def get_markets(self):
            ...

        @rate_limit(provider_attr="name")
        async def fetch_data(self):
            ...

    Args:
        provider_attr: Attribute name to get provider from self
                       (default: "provider_name")
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get rate limiter from instance (assumes self._rate_limiter exists)
            limiter: Optional[RateLimiter] = getattr(self, "_rate_limiter", None)
            if limiter is None:
                # No rate limiter, call function directly
                return await func(self, *args, **kwargs)

            # Get provider name
            provider = getattr(self, provider_attr, "unknown")

            # Acquire token (may wait)
            await limiter.acquire(provider)

            # Call function
            return await func(self, *args, **kwargs)

        return wrapper
    return decorator


# ========== Global Rate Limiter Instance ==========

_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set global rate limiter instance (for testing/customization)"""
    global _global_rate_limiter
    _global_rate_limiter = limiter
