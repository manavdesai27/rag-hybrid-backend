"""Per-route Redis-backed rate limiting with refilling token buckets and concurrency caps.

Usage:
    from app.rate_limit import rag_rate_limited

    @app.post("/ask", response_model=AskResponse)
    @rag_rate_limited(
        # Optional: enable per-question cooldown; key derived from request model
        question_key=lambda args, kwargs: (
            # compatible with both positional and keyword 'req'
            (kwargs.get("req") or (args[0] if args else None)).question.strip().lower()
            + "|max="
            + str((kwargs.get("req") or (args[0] if args else None)).max_tokens)
        )
    )
    def ask(req: AskRequest, db: Session = Depends(get_db), request: Request) -> AskResponse:
        ...

Only routes decorated with @rag_rate_limited are limited. No global middleware is registered.

Limits enforced (defaults from app.config.settings, all overridable per decorator call):
- Refilling token buckets per-IP:
    - per_minute: 20 req/min
    - burst_5s:   5 req/5s
    - per_day:    500 req/day
- Optional per-question cooldown: 1 req per 5 seconds for same (IP + question_key)
- Concurrency/backpressure:
    - global_concurrency = 8
    - per_ip_concurrency = 2
    - max_queue_size = 50 (queue of waiters across instance)
- 429 responses include Retry-After, with JSON:
    - {"detail": "rate limit exceeded"} for bucket/cooldown
    - {"detail": "server busy, try later"} for queue/concurrency saturation
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from typing import Any, Callable, Optional, Tuple

from fastapi import Request
from starlette.responses import JSONResponse

from app.cache import get_redis
from app.config import settings


def _get_request_from_args_kwargs(args: Tuple[Any, ...], kwargs: dict) -> Optional[Request]:
    """Try to retrieve the starlette Request object from endpoint args/kwargs."""
    req = kwargs.get("request")
    if isinstance(req, Request):
        return req
    for a in args:
        if isinstance(a, Request):
            return a
    # As a last resort, inspect signature to map args to param names;
    # but FastAPI typically passes as kwargs so this is rarely needed.
    return None


def _get_client_ip(request: Optional[Request], trust_xff: bool, real_ip_header: str) -> str:
    """Derive client IP from Request, honoring X-Forwarded-For when configured."""
    if request is None:
        return "unknown"
    try:
        if trust_xff:
            xff = request.headers.get(real_ip_header)
            if xff:
                # Use first IP in X-Forwarded-For, trimming spaces
                return xff.split(",")[0].strip()
        # Fallback to connection host
        host = request.client.host if request.client else None
        return host or "unknown"
    except Exception:
        return "unknown"


# Lua script: token-bucket with refill and consume
# KEYS[1] = tokens_key
# KEYS[2] = ts_key
# ARGV[1] = capacity
# ARGV[2] = refill_rate (tokens per second, can be fractional)
# ARGV[3] = now (seconds, float/number)
TOKEN_BUCKET_LUA = r"""
local tokens_key = KEYS[1]
local ts_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local tokens = tonumber(redis.call('GET', tokens_key))
local last_ts = tonumber(redis.call('GET', ts_key))

if tokens == nil then
    tokens = capacity
end
if last_ts == nil then
    last_ts = now
end

local elapsed = now - last_ts
if elapsed < 0 then
    elapsed = 0
end

local new_tokens = tokens + (elapsed * refill_rate)
if new_tokens > capacity then
    new_tokens = capacity
end

local allowed = 0
local retry_after = 0

if new_tokens >= 1.0 then
    new_tokens = new_tokens - 1.0
    allowed = 1
else
    allowed = 0
    local needed = 1.0 - new_tokens
    if refill_rate > 0 then
        retry_after = math.ceil(needed / refill_rate)
    else
        retry_after = 1
    end
end

-- Persist state
redis.call('SET', tokens_key, tostring(new_tokens))
redis.call('SET', ts_key, tostring(now))

-- Set TTL to ~2 full bucket refill times (avoid stale keys)
local ttl = math.max(5, math.ceil((capacity / math.max(refill_rate, 0.000001)) * 2))
redis.call('EXPIRE', tokens_key, ttl)
redis.call('EXPIRE', ts_key, ttl)

return {allowed, retry_after}
"""


# Lua script: reserve concurrency only if below limit (atomic)
# KEYS[1] = counter_key
# ARGV[1] = limit
# Returns 1 on success (reserved), 0 if at/over limit
RESERVE_CONCURRENCY_LUA = r"""
local key = KEYS[1]
local limit = tonumber(ARGV[1])

local v = redis.call('INCR', key)
if v == 1 then
    redis.call('EXPIRE', key, 60)
end

if v > limit then
    redis.call('DECR', key)
    return 0
else
    return 1
end
"""

# Lua script: release concurrency (never go below zero)
# KEYS[1] = counter_key
RELEASE_CONCURRENCY_LUA = r"""
local key = KEYS[1]
local v = tonumber(redis.call('GET', key) or '0')
if v > 0 then
    redis.call('DECR', key)
end
return 1
"""


def _token_bucket_allow(base_key: str, capacity: float, refill_rate: float, now: float) -> Tuple[bool, int]:
    """Consume 1 token from the bucket; returns (allowed, retry_after_seconds)."""
    r = get_redis()
    tokens_key = f"{base_key}:tokens"
    ts_key = f"{base_key}:ts"
    res = r.eval(TOKEN_BUCKET_LUA, 2, tokens_key, ts_key, str(capacity), str(refill_rate), str(now))
    try:
        allowed = int(float(res[0])) == 1
    except Exception:
        allowed = bool(res[0])
    try:
        retry_after = int(float(res[1]))
    except Exception:
        retry_after = 1
    return allowed, retry_after


def _reserve_concurrency(counter_key: str, limit: int) -> bool:
    r = get_redis()
    res = r.eval(RESERVE_CONCURRENCY_LUA, 1, counter_key, str(limit))
    try:
        return int(float(res)) == 1
    except Exception:
        return bool(res)


def _release_concurrency(counter_key: str) -> None:
    r = get_redis()
    try:
        r.eval(RELEASE_CONCURRENCY_LUA, 1, counter_key)
    except Exception:
        # best-effort
        pass


def _json_429(detail: str, retry_after: int) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": detail}, headers={"Retry-After": str(max(0, retry_after))})


def rag_rate_limited(
    *,
    # Buckets
    per_minute: Optional[int] = None,
    burst_5s: Optional[int] = None,
    per_day: Optional[int] = None,
    # Cooldown
    cooldown_seconds: Optional[int] = None,
    question_key: Optional[Callable[[Tuple[Any, ...], dict], Optional[str]]] = None,
    # Concurrency/Queue
    global_concurrency: Optional[int] = None,
    per_ip_concurrency: Optional[int] = None,
    max_queue_size: Optional[int] = None,
    # IP extraction
    trust_xff: Optional[bool] = None,
    real_ip_header: Optional[str] = None,
    # Queue wait config
    queue_wait_timeout_seconds: float = 2.0,
    queue_retry_sleep_seconds: float = 0.05,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to rate-limit a FastAPI route. Only decorated routes are limited.

    Parameters default to values from app.config.settings if None.
    - per_minute: capacity in the 60s bucket
    - burst_5s: capacity in the 5s bucket
    - per_day: capacity in the 24h bucket
    - cooldown_seconds: if provided with question_key, enforces a short cooldown
    - question_key: callable mapping (args, kwargs) to a string key (e.g., normalized question)
    - global_concurrency: max in-flight requests across the instance(s)
    - per_ip_concurrency: max in-flight per client IP
    - max_queue_size: maximum queued waiters before returning 429 immediately
    - trust_xff: whether to trust X-Forwarded-For
    - real_ip_header: header name for XFF
    - queue_wait_timeout_seconds: maximum wait to acquire a concurrency slot
    - queue_retry_sleep_seconds: sleep between retry attempts
    """
    # Resolve defaults from settings at decoration time (reads env-loaded values)
    per_minute = per_minute if per_minute is not None else settings.RATE_LIMIT_PER_MINUTE
    burst_5s = burst_5s if burst_5s is not None else settings.RATE_LIMIT_BURST_5S
    per_day = per_day if per_day is not None else settings.RATE_LIMIT_PER_DAY
    cooldown_seconds = (
        cooldown_seconds if cooldown_seconds is not None else settings.RATE_LIMIT_PER_QUESTION_INTERVAL_SECONDS
    )
    global_concurrency = global_concurrency if global_concurrency is not None else settings.MAX_CONCURRENCY
    per_ip_concurrency = per_ip_concurrency if per_ip_concurrency is not None else settings.PER_IP_MAX_CONCURRENCY
    max_queue_size = max_queue_size if max_queue_size is not None else settings.MAX_QUEUE_SIZE
    trust_xff = trust_xff if trust_xff is not None else settings.TRUST_X_FORWARDED_FOR
    real_ip_header = real_ip_header if real_ip_header is not None else settings.REAL_IP_HEADER

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        is_coro = inspect.iscoroutinefunction(func)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _get_request_from_args_kwargs(args, kwargs)
            ip = _get_client_ip(request, trust_xff=trust_xff, real_ip_header=real_ip_header)

            now = time.time()
            r = get_redis()

            # Queue handling
            waiting_key = "rag:queue:waiting"
            queued = False
            try:
                if max_queue_size and max_queue_size > 0:
                    waiting = r.incr(waiting_key)
                    queued = True
                    # Keep waiting counter from lingering
                    r.expire(waiting_key, 5)
                    if waiting > max_queue_size:
                        # Too many queued already
                        r.decr(waiting_key)
                        queued = False
                        return _json_429("server busy, try later", retry_after=1)

                # Concurrency reservation with short retries
                global_key = "rag:inflight:global"
                ip_key = f"rag:inflight:ip:{ip}"

                deadline = time.time() + queue_wait_timeout_seconds
                have_global = False
                have_ip = False
                while True:
                    have_global = _reserve_concurrency(global_key, int(global_concurrency))
                    if have_global:
                        have_ip = _reserve_concurrency(ip_key, int(per_ip_concurrency))
                        if have_ip:
                            break
                        # free global and retry
                        _release_concurrency(global_key)
                        have_global = False

                    if time.time() >= deadline:
                        return _json_429("server busy, try later", retry_after=1)
                    await asyncio.sleep(queue_retry_sleep_seconds)

                # Token buckets per-IP
                # minute bucket: capacity=per_minute, refill_rate=per_minute / 60
                # burst 5s: capacity=burst_5s, refill_rate=burst_5s / 5
                # day: capacity=per_day, refill_rate=per_day / 86400
                retry_afters = []

                if per_minute and per_minute > 0:
                    allowed, ra = _token_bucket_allow(
                        base_key=f"rag:rl:minute:{ip}", capacity=float(per_minute), refill_rate=float(per_minute) / 60.0, now=now
                    )
                    if not allowed:
                        retry_afters.append(ra)

                if burst_5s and burst_5s > 0:
                    allowed, ra = _token_bucket_allow(
                        base_key=f"rag:rl:burst5s:{ip}", capacity=float(burst_5s), refill_rate=float(burst_5s) / 5.0, now=now
                    )
                    if not allowed:
                        retry_afters.append(ra)

                if per_day and per_day > 0:
                    allowed, ra = _token_bucket_allow(
                        base_key=f"rag:rl:day:{ip}", capacity=float(per_day), refill_rate=float(per_day) / 86400.0, now=now
                    )
                    if not allowed:
                        retry_afters.append(ra)

                if retry_afters:
                    return _json_429("rate limit exceeded", retry_after=max(retry_afters))

                # Optional per-question cooldown
                if question_key is not None and cooldown_seconds and cooldown_seconds > 0:
                    try:
                        qkey = question_key(args, kwargs)
                    except Exception:
                        qkey = None
                    if qkey:
                        h = hashlib.sha256(f"{ip}|{qkey}".encode("utf-8")).hexdigest()
                        cd_key = f"rag:cooldown:{h}"
                        # SET if not exists with TTL; if already exists, fetch TTL for Retry-After
                        ok = r.set(cd_key, "1", nx=True, ex=int(cooldown_seconds))
                        if not ok:
                            ttl = r.ttl(cd_key)
                            # ttl might be -1 or -2; fall back to small delay
                            retry_after = ttl if ttl is not None and ttl > 0 else 1
                            return _json_429("rate limit exceeded", retry_after=retry_after)

                # Call the underlying endpoint (sync or async)
                if is_coro:
                    return await func(*args, **kwargs)
                # Sync callable: run in threadpool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            finally:
                # release queue counter
                if queued:
                    try:
                        r = get_redis()
                        r.decr(waiting_key)
                    except Exception:
                        pass
                # release concurrency reservations if held
                try:
                    _release_concurrency("rag:inflight:global")
                except Exception:
                    pass
                try:
                    _release_concurrency(f"rag:inflight:ip:{ip}")
                except Exception:
                    pass

        return wrapper

    return decorator
