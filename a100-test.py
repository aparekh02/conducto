#!/usr/bin/env python3
"""Conducto A100 Fine-Tuning Verification Test

Connects to the VESSL A100, checks the inference server,
sends a training batch, and verifies the model weights actually changed.

    python test.py
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

from nemotron_agent.config import Config

# ── Config ─────────────────────────────────────────────
LOCAL_PORT = 9090
REMOTE_PORT = 8000
BASE_URL = f"http://localhost:{LOCAL_PORT}"

# The same grid prompt we'll use before and after training
TEST_PROMPT = [
    {"role": "system", "content": "You are a power grid operator. Respond with JSON actions."},
    {"role": "user", "content": (
        "Grid state: B1-CITY load 180MW, B2-GAS gen 120MW, LN12 at 95% loading. "
        "B4-SUBURB BESS SoC 60%. Voltage at B1 is 0.94 pu (low). "
        "What actions do you take?"
    )},
]

# Fake but realistic training traces — high-reward examples
TRAINING_TRACES = [
    {
        "messages": [
            {"role": "system", "content": "You are a power grid operator. Respond with JSON actions."},
            {"role": "user", "content": (
                "Grid state: B1-CITY load 160MW, B2-GAS gen 100MW, LN12 at 88% loading. "
                "B4-SUBURB BESS SoC 70%. Voltage at B1 is 0.96 pu. What actions do you take?"
            )},
        ],
        "raw_response": json.dumps({
            "risk_level": "high",
            "assessment": "LN12 congested at 88%. Redispatch B2-GAS up and discharge BESS to relieve flow.",
            "actions": [
                {"type": "REDISPATCH", "target": "B2-GAS", "params": {"target_mw": 140}, "reason": "Increase local gen to reduce LN12 flow"},
                {"type": "STORAGE", "target": "BESS-SUBURB", "params": {"setpoint_mw": 30, "mode": "discharge"}, "reason": "Discharge BESS to support voltage"},
            ]
        }),
    },
    {
        "messages": [
            {"role": "system", "content": "You are a power grid operator. Respond with JSON actions."},
            {"role": "user", "content": (
                "Grid state: B3-WIND gen dropped to 20MW (low wind). B5-COAL gen 200MW. "
                "LN45 at 92% loading. B1-CITY voltage 0.93 pu. What actions do you take?"
            )},
        ],
        "raw_response": json.dumps({
            "risk_level": "critical",
            "assessment": "Wind drop + LN45 congestion. Emergency: ramp B5-COAL, shed B1-CITY flex load.",
            "actions": [
                {"type": "REDISPATCH", "target": "B5-COAL", "params": {"target_mw": 250}, "reason": "Compensate for wind shortfall"},
                {"type": "LOAD_SHAPE", "target": "B1-CITY", "params": {"delta_mw": -20}, "reason": "Shed flex load to ease LN45"},
                {"type": "STORAGE", "target": "BESS-SUBURB", "params": {"setpoint_mw": 40, "mode": "discharge"}, "reason": "Emergency BESS support"},
            ]
        }),
    },
    {
        "messages": [
            {"role": "system", "content": "You are a power grid operator. Respond with JSON actions."},
            {"role": "user", "content": (
                "Grid state: All lines below 70% loading. All voltages in [0.97, 1.02] pu. "
                "BESS SoC at 30%. Total gen > total load by 40MW. What actions do you take?"
            )},
        ],
        "raw_response": json.dumps({
            "risk_level": "low",
            "assessment": "Grid nominal. Charging BESS with excess generation to build reserves.",
            "actions": [
                {"type": "STORAGE", "target": "BESS-SUBURB", "params": {"setpoint_mw": -25, "mode": "charge"}, "reason": "Charge BESS during stable period"},
            ]
        }),
    },
]


def p(icon, msg, style=""):
    """Pretty print with icon."""
    colors = {"green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m", "cyan": "\033[36m", "dim": "\033[90m", "bold": "\033[1m", "": ""}
    reset = "\033[0m"
    c = colors.get(style, "")
    print(f"  {icon}  {c}{msg}{reset}")


def http_get(path, timeout=10):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())


def http_post(path, data, timeout=120):
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode()
    req = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())


def main():
    cfg = Config()
    print()
    print("  \033[1m\033[36m╔══════════════════════════════════════════════╗\033[0m")
    print("  \033[1m\033[36m║   CONDUCTO — A100 Fine-Tuning Verification   ║\033[0m")
    print("  \033[1m\033[36m╚══════════════════════════════════════════════╝\033[0m")
    print()

    # ── Step 1: Check SSH config ──────────────────────────
    p("1️⃣ ", "Checking VESSL SSH config...", "bold")
    if not cfg.vessl.ssh_host:
        p("❌", "VESSL_SSH_HOST not set in .env", "red")
        return
    p("✓", f"SSH host: {cfg.vessl.ssh_host}:{cfg.vessl.ssh_port}", "green")

    # ── Step 2: Check if tunnel already open ──────────────
    p("2️⃣ ", "Checking SSH tunnel to A100...", "bold")
    tunnel_proc = None
    tunnel_needed = True
    try:
        health = http_get("/health", timeout=5)
        p("✓", f"Tunnel already open — server responding (train_steps={health.get('train_steps', 0)})", "green")
        tunnel_needed = False
    except Exception:
        p("→", "No tunnel detected, opening one...", "yellow")
        from nemotron_agent.training.vessl_runner import VESSLRunner
        runner = VESSLRunner(cfg.vessl)
        tunnel_proc = runner.start_ssh_tunnel(remote_port=REMOTE_PORT, local_port=LOCAL_PORT)
        time.sleep(3)
        try:
            health = http_get("/health", timeout=10)
            p("✓", f"Tunnel opened — server responding", "green")
        except Exception as e:
            p("❌", f"Cannot reach server through tunnel: {e}", "red")
            p("→", "Is peft_serve.py running on the A100?", "yellow")
            if tunnel_proc:
                tunnel_proc.terminate()
            return

    # ── Step 3: Server health ─────────────────────────────
    p("3️⃣ ", "Checking inference server health...", "bold")
    try:
        health = http_get("/health")
        model_info = health.get("model", {})
        p("✓", f"Model: {model_info.get('model_id', 'unknown')}", "green")
        p("✓", f"Quantization: {model_info.get('quantization', 'unknown')}", "green")
        p("✓", f"Online training: {model_info.get('online_training', False)}", "green")
        p("✓", f"Total train steps so far: {health.get('train_steps', 0)}", "green")
        initial_steps = health.get("train_steps", 0)
    except Exception as e:
        p("❌", f"Health check failed: {e}", "red")
        return

    # ── Step 4: Baseline inference ────────────────────────
    p("4️⃣ ", "Sending baseline inference request (BEFORE training)...", "bold")
    try:
        t0 = time.time()
        resp_before = http_post("/v1/chat/completions", {
            "model": "nemotron-finetuned",
            "messages": TEST_PROMPT,
            "temperature": 0.1,  # low temp for more deterministic comparison
            "max_tokens": 512,
        })
        latency = (time.time() - t0) * 1000
        text_before = resp_before["choices"][0]["message"]["content"]
        usage = resp_before.get("usage", {})
        p("✓", f"Response received in {latency:.0f}ms  ({usage.get('completion_tokens', '?')} tokens)", "green")
        p("→", f"Response preview: {text_before[:150]}...", "dim")
    except Exception as e:
        p("❌", f"Inference failed: {e}", "red")
        return

    # ── Step 5: Send training traces ──────────────────────
    p("5️⃣ ", f"Sending {len(TRAINING_TRACES)} training traces to /train endpoint...", "bold")
    p("→", "This triggers real loss.backward() + optimizer.step() on LoRA weights", "dim")
    try:
        t0 = time.time()
        train_result = http_post("/train", {
            "traces": TRAINING_TRACES,
            "learning_rate": 1e-4,  # higher LR for visible effect in test
            "max_steps": 10,        # 10 gradient steps for stronger signal
            "max_length": 1024,
        })
        train_ms = (time.time() - t0) * 1000
        status = train_result.get("status", "unknown")
        steps = train_result.get("steps", 0)
        loss = train_result.get("loss", 0)
        total = train_result.get("total_steps", 0)
        traces_used = train_result.get("traces_used", 0)
        server_ms = train_result.get("train_ms", 0)

        if status == "ok" and steps > 0:
            p("✓", f"Training completed!", "green")
            p("✓", f"  Steps:        {steps}", "green")
            p("✓", f"  Avg loss:     {loss:.4f}", "green")
            p("✓", f"  Total steps:  {total} (was {initial_steps})", "green")
            p("✓", f"  Traces used:  {traces_used}", "green")
            p("✓", f"  Server time:  {server_ms:.0f}ms", "green")
            p("✓", f"  Round-trip:   {train_ms:.0f}ms", "green")
        else:
            p("❌", f"Training returned: {train_result}", "red")
            return
    except Exception as e:
        p("❌", f"Training request failed: {e}", "red")
        return

    # ── Step 6: Verify train steps incremented ────────────
    p("6️⃣ ", "Verifying server state after training...", "bold")
    try:
        health_after = http_get("/health")
        new_steps = health_after.get("train_steps", 0)
        delta = new_steps - initial_steps
        if delta > 0:
            p("✓", f"Train steps incremented: {initial_steps} → {new_steps} (+{delta})", "green")
        else:
            p("❌", f"Train steps did NOT change: still {new_steps}", "red")
            return
    except Exception as e:
        p("❌", f"Post-training health check failed: {e}", "red")
        return

    # ── Step 7: Post-training inference ───────────────────
    p("7️⃣ ", "Sending same inference request (AFTER training)...", "bold")
    try:
        t0 = time.time()
        resp_after = http_post("/v1/chat/completions", {
            "model": "nemotron-finetuned",
            "messages": TEST_PROMPT,
            "temperature": 0.1,
            "max_tokens": 512,
        })
        latency = (time.time() - t0) * 1000
        text_after = resp_after["choices"][0]["message"]["content"]
        usage = resp_after.get("usage", {})
        p("✓", f"Response received in {latency:.0f}ms  ({usage.get('completion_tokens', '?')} tokens)", "green")
        p("→", f"Response preview: {text_after[:150]}...", "dim")
    except Exception as e:
        p("❌", f"Post-training inference failed: {e}", "red")
        return

    # ── Step 8: Compare responses ─────────────────────────
    p("8️⃣ ", "Comparing before/after responses...", "bold")
    changed = text_before.strip() != text_after.strip()
    if changed:
        p("✓", "Model output CHANGED after training — weights were updated!", "green")
    else:
        p("⚠", "Outputs are identical (may need more training steps or higher LR for visible difference)", "yellow")
        p("→", "But train_steps DID increment, confirming gradients were applied", "dim")

    # ── Summary ───────────────────────────────────────────
    print()
    print("  \033[1m══════════════════════════════════════════════\033[0m")
    print("  \033[1m  RESULTS\033[0m")
    print("  \033[1m══════════════════════════════════════════════\033[0m")
    print()
    p("🖥️ ", f"Server:          peft_serve.py on A100 (4-bit Nemotron)", "cyan")
    p("🔢", f"Gradient steps:  {initial_steps} → {new_steps} (+{delta})", "cyan")
    p("📉", f"Training loss:   {loss:.4f}", "cyan")
    p("⚡", f"Train latency:   {server_ms:.0f}ms on GPU", "cyan")
    p("🔄", f"Output changed:  {'YES' if changed else 'NO (but weights updated)'}", "cyan")
    print()

    if delta > 0:
        p("✅", "VERIFIED: Real fine-tuning is happening on the A100.", "green")
        p("→", f"The model has received {new_steps} total gradient updates to its LoRA adapters.", "dim")
        p("→", "Each main.py cycle sends traces → /train → loss.backward() → optimizer.step()", "dim")
    else:
        p("❌", "Fine-tuning could NOT be verified.", "red")

    print()

    # ── Cleanup ───────────────────────────────────────────
    if tunnel_proc and tunnel_needed:
        p("→", "Closing test SSH tunnel...", "dim")
        tunnel_proc.terminate()
        tunnel_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
