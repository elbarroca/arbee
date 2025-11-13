"""
Circuit Breaker Pattern

Prevents cascading failures by detecting provider issues and failing fast.
Implements the circuit breaker pattern with automatic recovery.
"""
import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 60.0  # Seconds to wait before half-open
    error_rate_threshold: float = 0.5  # Error rate to open (0-1)
    min_requests: int = 10  # Min requests before checking error rate


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    provider: str
    state: CircuitState = CircuitState.CLOSED
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: float = time.time()
    open_count: int = 0  # Times circuit has opened
    half_open_count: int = 0  # Times circuit has gone half-open

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        total = self.successful_requests + self.failed_requests
        return self.failed_requests / total if total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.error_rate

    @property
    def uptime_seconds(self) -> float:
        """Get uptime since last state change"""
        return time.time() - self.last_state_change


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker for a single provider.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    Transitions:
    - CLOSED -> OPEN: After N failures or high error rate
    - OPEN -> HALF_OPEN: After timeout period
    - HALF_OPEN -> CLOSED: After N successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(self, provider: str, config: CircuitBreakerConfig):
        self.provider = provider
        self.config = config
        self.stats = CircuitBreakerStats(provider=provider)
        self._lock = asyncio.Lock()

        logger.info(
            f"CircuitBreaker created for {provider}: "
            f"failure_threshold={config.failure_threshold}, "
            f"timeout={config.timeout}s"
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional args for func
            **kwargs: Keyword args for func

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from func
        """
        async with self._lock:
            # Check circuit state
            await self._check_state()

            if self.stats.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker OPEN for {self.provider} "
                    f"(failures: {self.stats.consecutive_failures}, "
                    f"error_rate: {self.stats.error_rate:.2%})"
                )

        # Execute function
        self.stats.total_requests += 1

        try:
            result = await func(*args, **kwargs)

            # Success
            await self._record_success()
            return result

        except Exception as e:
            # Failure
            await self._record_failure(e)
            raise

    async def _check_state(self) -> None:
        """Check and update circuit state"""
        if self.stats.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self.stats.last_failure_time:
                elapsed = time.time() - self.stats.last_failure_time
                if elapsed >= self.config.timeout:
                    # Transition to half-open
                    await self._transition_to(CircuitState.HALF_OPEN)

    async def _record_success(self) -> None:
        """Record successful request"""
        async with self._lock:
            self.stats.successful_requests += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                # Check if we should close
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, error: Exception) -> None:
        """Record failed request"""
        async with self._lock:
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker failure for {self.provider}: "
                f"{type(error).__name__}: {error} "
                f"(consecutive: {self.stats.consecutive_failures})"
            )

            if self.stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open -> open
                await self._transition_to(CircuitState.OPEN)

            elif self.stats.state == CircuitState.CLOSED:
                # Check if we should open
                should_open = False

                # Check consecutive failures
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    should_open = True
                    logger.warning(
                        f"Circuit breaker opening due to consecutive failures: "
                        f"{self.stats.consecutive_failures}"
                    )

                # Check error rate (only after min requests)
                total = self.stats.successful_requests + self.stats.failed_requests
                if total >= self.config.min_requests:
                    if self.stats.error_rate >= self.config.error_rate_threshold:
                        should_open = True
                        logger.warning(
                            f"Circuit breaker opening due to error rate: "
                            f"{self.stats.error_rate:.2%}"
                        )

                if should_open:
                    await self._transition_to(CircuitState.OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state"""
        old_state = self.stats.state
        self.stats.state = new_state
        self.stats.last_state_change = time.time()

        if new_state == CircuitState.OPEN:
            self.stats.open_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.stats.half_open_count += 1
            self.stats.consecutive_successes = 0  # Reset for testing

        logger.info(
            f"Circuit breaker {self.provider}: "
            f"{old_state.value} -> {new_state.value}"
        )

    def get_state(self) -> CircuitState:
        """Get current state"""
        return self.stats.state

    def reset(self) -> None:
        """Reset circuit breaker"""
        self.stats = CircuitBreakerStats(provider=self.provider)
        logger.info(f"Circuit breaker reset for {self.provider}")


class CircuitBreakerManager:
    """
    Manages circuit breakers for multiple providers.
    """

    # Default configurations per provider
    DEFAULT_CONFIGS = {
        "polymarket": CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
            error_rate_threshold=0.5,
            min_requests=10
        ),
        "kalshi": CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
            error_rate_threshold=0.5,
            min_requests=10
        ),
        "default": CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
            error_rate_threshold=0.5,
            min_requests=10
        )
    }

    def __init__(
        self,
        custom_configs: Optional[Dict[str, CircuitBreakerConfig]] = None
    ):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._custom_configs = custom_configs or {}

        logger.info("CircuitBreakerManager initialized")

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider"""
        if provider not in self._breakers:
            # Get config (custom or default)
            config = (
                self._custom_configs.get(provider) or
                self.DEFAULT_CONFIGS.get(provider) or
                self.DEFAULT_CONFIGS["default"]
            )

            self._breakers[provider] = CircuitBreaker(provider, config)

        return self._breakers[provider]

    async def call(
        self,
        provider: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Call function through circuit breaker.

        Args:
            provider: Provider name
            func: Async function to call
            *args: Positional args for func
            **kwargs: Keyword args for func

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        breaker = self._get_breaker(provider)
        return await breaker.call(func, *args, **kwargs)

    def get_state(self, provider: str) -> CircuitState:
        """Get circuit state for provider"""
        breaker = self._get_breaker(provider)
        return breaker.get_state()

    def get_stats(
        self,
        provider: Optional[str] = None
    ) -> Dict[str, CircuitBreakerStats]:
        """
        Get circuit breaker statistics.

        Args:
            provider: If specified, get stats for specific provider

        Returns:
            Dict of provider -> CircuitBreakerStats
        """
        if provider:
            breaker = self._get_breaker(provider)
            return {provider: breaker.stats}

        return {
            provider: breaker.stats
            for provider, breaker in self._breakers.items()
        }

    def reset(self, provider: Optional[str] = None) -> None:
        """
        Reset circuit breaker.

        Args:
            provider: If specified, reset only this provider
        """
        if provider:
            if provider in self._breakers:
                self._breakers[provider].reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")

    def get_info(self) -> Dict[str, Any]:
        """Get circuit breaker info"""
        return {
            "providers": list(self._breakers.keys()),
            "states": {
                provider: {
                    "state": breaker.stats.state.value,
                    "error_rate": breaker.stats.error_rate,
                    "consecutive_failures": breaker.stats.consecutive_failures
                }
                for provider, breaker in self._breakers.items()
            }
        }


# ========== Decorator ==========

def circuit_breaker(provider_attr: str = "provider_name"):
    """
    Decorator to wrap async function with circuit breaker.

    Usage:
        @circuit_breaker()
        async def get_markets(self):
            ...

        @circuit_breaker(provider_attr="name")
        async def fetch_data(self):
            ...

    Args:
        provider_attr: Attribute name to get provider from self
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get circuit breaker manager (assumes self._circuit_breaker exists)
            manager: Optional[CircuitBreakerManager] = getattr(
                self, "_circuit_breaker", None
            )
            if manager is None:
                # No circuit breaker, call function directly
                return await func(self, *args, **kwargs)

            # Get provider name
            provider = getattr(self, provider_attr, "unknown")

            # Call through circuit breaker
            return await manager.call(provider, func, self, *args, **kwargs)

        return wrapper
    return decorator


# ========== Global Circuit Breaker Manager ==========

_global_circuit_breaker: Optional[CircuitBreakerManager] = None


def get_circuit_breaker() -> CircuitBreakerManager:
    """Get or create global circuit breaker manager"""
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreakerManager()
    return _global_circuit_breaker


def set_circuit_breaker(manager: CircuitBreakerManager) -> None:
    """Set global circuit breaker manager (for testing/customization)"""
    global _global_circuit_breaker
    _global_circuit_breaker = manager
