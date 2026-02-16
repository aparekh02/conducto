"""Market signal ingestion.

Ingests electricity market data that influences dispatch economics:
  - Locational Marginal Prices (LMPs) per bus/node
  - Day-ahead and real-time prices
  - Ancillary service prices (regulation, reserves)
  - Congestion charges
  - Carbon prices / emission allowances

Sources: ISO/RTO feeds (CAISO OASIS, PJM Data Miner, etc.)

Publishes to topic: grid.market
"""

from __future__ import annotations

import logging
import random
import time

from data.bus import Message, MessageBus, Topic

logger = logging.getLogger(__name__)


class MarketIngest:
    """Ingests market pricing signals."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus

    def publish_lmp(
        self,
        node_id: str,
        lmp_energy: float,
        lmp_congestion: float,
        lmp_loss: float,
        market_type: str = "real_time",
    ) -> None:
        """Publish a Locational Marginal Price for a pricing node."""
        self.bus.publish(Message(
            topic=Topic.MARKET,
            key=f"lmp:{node_id}",
            value={
                "type": "lmp",
                "node_id": node_id,
                "market_type": market_type,
                "lmp_total": lmp_energy + lmp_congestion + lmp_loss,
                "lmp_energy": lmp_energy,
                "lmp_congestion": lmp_congestion,
                "lmp_loss": lmp_loss,
                "currency": "USD/MWh",
            },
        ))

    def publish_ancillary_prices(
        self,
        region_id: str,
        regulation_up: float,
        regulation_down: float,
        spinning_reserve: float,
        non_spinning_reserve: float,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.MARKET,
            key=f"anc:{region_id}",
            value={
                "type": "ancillary",
                "region_id": region_id,
                "regulation_up_usd_mw": regulation_up,
                "regulation_down_usd_mw": regulation_down,
                "spinning_reserve_usd_mw": spinning_reserve,
                "non_spinning_reserve_usd_mw": non_spinning_reserve,
            },
        ))

    def publish_carbon_price(self, price_usd_per_ton: float) -> None:
        self.bus.publish(Message(
            topic=Topic.MARKET,
            key="carbon",
            value={
                "type": "carbon",
                "price_usd_per_ton": price_usd_per_ton,
            },
        ))

    def generate_demo_snapshot(self, node_ids: list[str]) -> None:
        """Generate realistic market data for testing."""
        base_energy = random.uniform(25, 80)
        for nid in node_ids:
            self.publish_lmp(
                node_id=nid,
                lmp_energy=base_energy + random.gauss(0, 5),
                lmp_congestion=max(0, random.gauss(2, 8)),
                lmp_loss=random.uniform(0.5, 3),
            )
        self.publish_ancillary_prices(
            region_id="REGION-1",
            regulation_up=random.uniform(5, 25),
            regulation_down=random.uniform(3, 15),
            spinning_reserve=random.uniform(4, 20),
            non_spinning_reserve=random.uniform(2, 10),
        )
        self.publish_carbon_price(random.uniform(15, 55))
