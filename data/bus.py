"""Kafka-style message bus for streaming grid telemetry.

Provides a unified pub/sub interface that works in two modes:
  - LOCAL:  in-process asyncio queues (dev/testing, no infrastructure)
  - KAFKA:  real Kafka broker (production)

All ingest modules publish to topics on this bus.  The feature service
subscribes and converts raw signals into GridState vectors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Topic(str, Enum):
    SCADA = "grid.scada"
    PMU = "grid.pmu"
    WEATHER = "grid.weather"
    MARKET = "grid.market"
    ALERTS = "grid.alerts"


@dataclass
class Message:
    topic: str
    key: str                         # e.g. bus_id, line_id, station_id
    value: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "topic": self.topic,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, raw: str) -> Message:
        d = json.loads(raw)
        return cls(**d)


# Type alias for subscriber callbacks
Subscriber = Callable[[Message], None]


class MessageBus:
    """Unified message bus — local queues or Kafka, same interface."""

    def __init__(self, mode: str = "local", kafka_config: dict | None = None) -> None:
        self.mode = mode
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._buffer: dict[str, list[Message]] = defaultdict(list)
        self._kafka_producer = None
        self._kafka_consumer = None

        if mode == "kafka" and kafka_config:
            self._init_kafka(kafka_config)

        logger.info("MessageBus ready  mode=%s", mode)

    def _init_kafka(self, config: dict) -> None:
        from confluent_kafka import Consumer, Producer

        self._kafka_producer = Producer({
            "bootstrap.servers": config["bootstrap_servers"],
            "client.id": "conducto-data-ingest",
        })
        self._kafka_consumer = Consumer({
            "bootstrap.servers": config["bootstrap_servers"],
            "group.id": config.get("group_id", "conducto-features"),
            "auto.offset.reset": "latest",
        })

    # ── Publish ──────────────────────────────────────────

    def publish(self, msg: Message) -> None:
        """Publish a message to a topic."""
        if self.mode == "kafka" and self._kafka_producer:
            self._kafka_producer.produce(
                msg.topic,
                key=msg.key.encode(),
                value=msg.to_json().encode(),
            )
            self._kafka_producer.poll(0)
        else:
            # Local mode: buffer + notify subscribers
            self._buffer[msg.topic].append(msg)
            for sub in self._subscribers.get(msg.topic, []):
                sub(msg)

    def flush(self) -> None:
        if self._kafka_producer:
            self._kafka_producer.flush()

    # ── Subscribe ────────────────────────────────────────

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        """Register a callback for messages on a topic."""
        self._subscribers[topic].append(callback)
        logger.debug("Subscribed to %s", topic)

    # ── Read buffered (local mode) ──────────────────────

    def read_latest(self, topic: str, n: int = 1) -> list[Message]:
        """Read the n most recent messages from a topic (local mode)."""
        return self._buffer.get(topic, [])[-n:]

    def read_since(self, topic: str, since: float) -> list[Message]:
        """Read all messages after a timestamp (local mode)."""
        return [m for m in self._buffer.get(topic, []) if m.timestamp > since]

    @property
    def topics(self) -> list[str]:
        return list(self._buffer.keys())

    def stats(self) -> dict[str, int]:
        return {topic: len(msgs) for topic, msgs in self._buffer.items()}
