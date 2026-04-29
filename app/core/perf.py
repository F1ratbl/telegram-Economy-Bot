import logging
import time
from contextlib import contextmanager
from functools import wraps


def log_timing(logger_name: str = "economy-assistant-bot"):
    logger = logging.getLogger(logger_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            logger.info("START %s", func.__qualname__)
            try:
                result = func(*args, **kwargs)
            except Exception:
                duration_ms = (time.perf_counter() - started_at) * 1000
                logger.exception("ERROR %s | duration_ms=%.2f", func.__qualname__, duration_ms)
                raise
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info("END %s | duration_ms=%.2f", func.__qualname__, duration_ms)
            return result

        return wrapper

    return decorator


@contextmanager
def timed_block(name: str, logger_name: str = "economy-assistant-bot"):
    logger = logging.getLogger(logger_name)
    started_at = time.perf_counter()
    logger.info("START %s", name)
    try:
        yield
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception("ERROR %s | duration_ms=%.2f", name, duration_ms)
        raise
    duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info("END %s | duration_ms=%.2f", name, duration_ms)
