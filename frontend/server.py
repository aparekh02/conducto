"""Conducto Web Dashboard — real-time grid visualization + VESSL monitoring.

Serves a web frontend that shows:
  - Live grid topology with power flows, voltages, and line loading
  - Agent decisions and model improvement metrics
  - VESSL A100 GPU monitoring (nvidia-smi via SSH)
  - Energy readings log from each bus

Usage:
    Started automatically by main.py (--web flag), or standalone:
        python -m frontend.server
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Conducto Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Shared state — updated by main.py, read by WebSocket ──
class DashboardState:
    """Thread-safe shared state between the main loop and the web server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "cycle": 0,
            "episode": 1,
            "step_in_episode": 0,
            "grid": {},
            "agent": {},
            "metrics": {},
            "vessl": {"gpu_util": 0, "gpu_mem_used": 0, "gpu_mem_total": 80,
                      "status": "idle", "training_progress": ""},
            "energy_log": [],
            "topology": {},
        }
        self._clients: list[WebSocket] = []

    def update(self, data: dict) -> None:
        with self._lock:
            self._state.update(data)

    def get(self) -> dict:
        with self._lock:
            return self._state.copy()

    def add_client(self, ws: WebSocket) -> None:
        self._clients.append(ws)

    def remove_client(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self) -> None:
        state = self.get()
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(state)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_client(ws)


dashboard_state = DashboardState()


# ── Routes ─────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    dashboard_state.add_client(ws)
    # Send initial state
    await ws.send_json(dashboard_state.get())
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        dashboard_state.remove_client(ws)


@app.get("/api/vessl-gpu")
async def vessl_gpu():
    """Poll GPU status from the VESSL workspace via SSH."""
    return dashboard_state.get().get("vessl", {})


# ── VESSL GPU monitor (runs in background thread) ─────────

def _poll_vessl_gpu(ssh_host: str, ssh_key: str, ssh_port: int) -> dict:
    """SSH into VESSL workspace and run nvidia-smi."""
    try:
        key_path = str(Path(ssh_key).expanduser())
        cmd = [
            "ssh", "-i", key_path,
            "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            ssh_host,
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            parts = [p.strip() for p in parts]
            return {
                "gpu_util": float(parts[0]),
                "gpu_mem_used": float(parts[1]) / 1024,  # MB → GB
                "gpu_mem_total": float(parts[2]) / 1024,
                "gpu_temp": float(parts[3]),
                "gpu_power_w": float(parts[4]),
                "status": "online",
                "last_poll": time.time(),
            }
    except Exception as e:
        logger.debug("VESSL GPU poll failed: %s", e)
    return {"status": "offline", "last_poll": time.time()}


def start_vessl_monitor(ssh_host: str, ssh_key: str, ssh_port: int, interval: float = 10.0) -> None:
    """Background thread that polls GPU status every N seconds."""
    def _monitor():
        while True:
            gpu_data = _poll_vessl_gpu(ssh_host, ssh_key, ssh_port)
            # Preserve training status from main loop updates
            current = dashboard_state.get().get("vessl", {})
            gpu_data["training_progress"] = current.get("training_progress", "")
            gpu_data["finetune_status"] = current.get("finetune_status", "idle")
            gpu_data["finetune_runs"] = current.get("finetune_runs", 0)
            dashboard_state.update({"vessl": gpu_data})
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True, name="vessl-gpu-monitor")
    t.start()
    logger.info("VESSL GPU monitor started (polling every %.0fs)", interval)


# ── Broadcast loop (pushes state to all WebSocket clients) ─

async def _broadcast_loop(interval: float = 1.0):
    while True:
        await dashboard_state.broadcast()
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_broadcast_loop(interval=1.0))


# ── Standalone entry point ─────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    uvicorn.run("frontend.server:app", host="0.0.0.0", port=8080, reload=True)
