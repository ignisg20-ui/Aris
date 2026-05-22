"""Prometheus metrics.

Exports:
* Gauges: train loss, eval loss, learning rate, tokens/s, GPU mem & util.
* Counters: requests served, tokens generated, requests refused.
* Histograms: prompt length, completion length, latency.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_DEFAULT_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


@dataclass
class MetricRegistry:
    namespace: str = "aris"
    registry: CollectorRegistry | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or REGISTRY
        self._lock = threading.Lock()
        self._gauges: dict[str, Gauge] = {}
        self._counters: dict[str, Counter] = {}
        self._hists: dict[str, Histogram] = {}

    def gauge(self, name: str, description: str = "") -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(f"{self.namespace}_{name}", description or name, registry=self.registry)
        return self._gauges[name]

    def counter(self, name: str, description: str = "", labelnames: tuple[str, ...] = ()) -> Counter:
        key = name + "::" + ",".join(labelnames)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(
                    f"{self.namespace}_{name}",
                    description or name,
                    labelnames=list(labelnames),
                    registry=self.registry,
                )
        return self._counters[key]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: tuple[float, ...] = _DEFAULT_BUCKETS,
        labelnames: tuple[str, ...] = (),
    ) -> Histogram:
        key = name + "::" + ",".join(labelnames)
        with self._lock:
            if key not in self._hists:
                self._hists[key] = Histogram(
                    f"{self.namespace}_{name}",
                    description or name,
                    buckets=list(buckets),
                    labelnames=list(labelnames),
                    registry=self.registry,
                )
        return self._hists[key]


def metrics_app(registry: CollectorRegistry | None = None):
    """Returns an ASGI app exposing ``GET /metrics`` for Prometheus scraping."""

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":  # pragma: no cover
            return
        body = generate_latest(registry or REGISTRY)
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", CONTENT_TYPE_LATEST.encode())],
        })
        await send({"type": "http.response.body", "body": body})

    return app
