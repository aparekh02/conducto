"""Weather feed ingestion.

Ingests weather data that drives renewable generation output and
load forecasting:
  - Solar irradiance (W/m²) — drives PV generation
  - Wind speed and direction — drives wind turbine output
  - Temperature — drives HVAC load
  - Cloud cover — predicts solar ramps
  - Precipitation — affects line ratings

Sources: NWS API, OpenWeather, utility-owned weather stations.

Publishes to topic: grid.weather
"""

from __future__ import annotations

import logging
import math
import random
import time

from data.bus import Message, MessageBus, Topic

logger = logging.getLogger(__name__)


class WeatherIngest:
    """Ingests weather observations and forecasts."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus

    def publish_observation(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        solar_irradiance_w_m2: float,
        wind_speed_m_s: float,
        wind_direction_deg: float,
        temperature_c: float,
        humidity_pct: float,
        cloud_cover_pct: float,
        precipitation_mm_hr: float = 0.0,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.WEATHER,
            key=f"wx:{station_id}",
            value={
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "solar_irradiance_w_m2": solar_irradiance_w_m2,
                "wind_speed_m_s": wind_speed_m_s,
                "wind_direction_deg": wind_direction_deg,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "cloud_cover_pct": cloud_cover_pct,
                "precipitation_mm_hr": precipitation_mm_hr,
            },
        ))

    def publish_forecast(
        self,
        station_id: str,
        horizon_hours: int,
        solar_forecast_w_m2: list[float],
        wind_forecast_m_s: list[float],
        temp_forecast_c: list[float],
    ) -> None:
        """Publish an hourly forecast out to horizon_hours."""
        self.bus.publish(Message(
            topic=Topic.WEATHER,
            key=f"wx_fcst:{station_id}",
            value={
                "type": "forecast",
                "station_id": station_id,
                "horizon_hours": horizon_hours,
                "solar_w_m2": solar_forecast_w_m2[:horizon_hours],
                "wind_m_s": wind_forecast_m_s[:horizon_hours],
                "temperature_c": temp_forecast_c[:horizon_hours],
            },
        ))

    def generate_demo_observation(
        self, station_id: str, hour_of_day: float,
    ) -> None:
        """Generate a realistic weather observation for testing."""
        solar = max(0, 900 * math.sin(
            math.pi * max(0, min(hour_of_day - 6, 12)) / 12
        )) if 6 <= hour_of_day <= 18 else 0.0
        solar *= random.uniform(0.6, 1.0)

        self.publish_observation(
            station_id=station_id,
            latitude=37.78,
            longitude=-122.42,
            solar_irradiance_w_m2=solar,
            wind_speed_m_s=max(0, random.gauss(8, 3)),
            wind_direction_deg=random.uniform(0, 360),
            temperature_c=random.gauss(28, 5),
            humidity_pct=random.uniform(30, 80),
            cloud_cover_pct=random.uniform(0, 60),
        )
