"""Token-bucket rate limiter for outbound media API calls.

Single responsibility: pace calls to a configured requests-per-minute limit.
It implements the classic token bucket so short bursts are allowed while the
long-run average never exceeds the limit. The limiter knows nothing about
Groq, caching or analyzers.
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket with refill per second.

    capacity equals the per-minute allowance; tokens refill continuously at
    capacity/60 per second, so a call is only delayed when the bucket is
    empty. acquire() blocks the calling thread until a token is available.
    """

    def __init__(self, rate_per_minute: int) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be positive, got {rate_per_minute}")
        self._capacity = float(rate_per_minute)
        self._tokens = float(rate_per_minute)
        self._refill_per_second = rate_per_minute / 60.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self._refill()
            while self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                self._lock.release()
                time.sleep(deficit / self._refill_per_second)
                self._lock.acquire()
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens
