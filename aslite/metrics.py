"""
Timings for /stats: each web worker counts how long requests take per kind of page (and how
long model fits take) in memory, as a histogram per minute, and writes them to metrics.sqlite
about once a minute, so /stats sees every worker. Background jobs record their runs there
directly (db.put_job). Nothing about who asked or what: only the kind of page and how long.
"""

import math
import time
import threading
from collections import Counter

from aslite import db

BUCKET = 1.25  # histogram buckets grow by 25%: bucket i holds BUCKET**i .. BUCKET**(i+1) ms
FLUSH = 60  # seconds

_lock = threading.Lock()
_counts = {}  # (minute, name) -> [n, total seconds, Counter of buckets]
_flushed = time.time()


def bucket(seconds):
    return max(0, int(math.log(max(seconds * 1000, 1)) / math.log(BUCKET)))


def record(name, seconds):
    global _flushed
    minute = int(time.time() // 60 * 60)
    with _lock:
        c = _counts.setdefault((minute, name), [0, 0.0, Counter()])
        c[0] += 1
        c[1] += seconds
        c[2][bucket(seconds)] += 1
        due = time.time() - _flushed > FLUSH
        if due:
            rows = [(t, name, n, total, dict(h)) for (t, name), (n, total, h) in _counts.items()]
            _counts.clear()
            _flushed = time.time()
    if due:
        try:
            db.put_timings(rows)
        except Exception as e:  # only statistics
            print("storing timings failed: %s" % (e,))


def flush():
    """writes what this worker counted so far (for /stats itself and tests)"""
    global _flushed
    with _lock:
        rows = [(t, name, n, total, dict(h)) for (t, name), (n, total, h) in _counts.items()]
        _counts.clear()
        _flushed = time.time()
    if rows:
        db.put_timings(rows)


def summary(since):
    """{name: dict(n, mean, p50, p95, max)} in seconds, over every worker since then"""
    merged = {}
    for _, name, n, total, hist in db.get_timings(since):
        m = merged.setdefault(name, [0, 0.0, Counter()])
        m[0] += n
        m[1] += total
        m[2].update({int(k): v for k, v in hist.items()})
    out = {}
    for name, (n, total, hist) in merged.items():

        def pct(q, hist=hist, n=n):
            seen = 0
            for b in sorted(hist):
                seen += hist[b]
                if seen >= q * n:
                    return BUCKET ** (b + 0.5) / 1000  # the middle of the bucket
            return 0.0

        out[name] = dict(n=n, mean=total / n, p50=pct(0.5), p95=pct(0.95), max=BUCKET ** (max(hist) + 1) / 1000)
    return out
