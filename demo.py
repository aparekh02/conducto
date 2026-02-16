#!/usr/bin/env python3
"""Conducto Grid Demo — visual simulation with a heuristic agent + A100 fine-tuning.

Runs the power grid simulation with a smart rule-based agent that
responds to congestion, voltage violations, and disturbances in real-time.
Agent decisions are sent to the VESSL A100 for online LoRA fine-tuning.
All actions are pushed to the web dashboard so you can watch the grid
being managed live.

    python demo.py              # start at http://localhost:8080
    python demo.py --port 3000  # custom port
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time

import numpy as np

from nemotron_agent.gym.grid_env import GridEnv
from nemotron_agent.gym.scenarios import SCENARIOS

logger = logging.getLogger(__name__)


# ── VESSL A100 initialization ──────────────────────────────────

def init_a100() -> dict:
    """Deploy peft_serve.py to VESSL A100, open SSH tunnel, wait for server.

    Returns a dict with 'runner', 'tunnel', 'ready', 'train_steps', 'status'.
    """
    from nemotron_agent.config import VESSLConfig
    from nemotron_agent.training.vessl_runner import VESSLRunner

    state = {"runner": None, "tunnel": None, "ready": False, "train_steps": 0, "status": "initializing"}

    try:
        cfg = VESSLConfig()
        runner = VESSLRunner(cfg)
        state["runner"] = runner
        state["status"] = "setting up workspace"
        print("  [A100] Setting up VESSL workspace...")
        runner.setup_workspace()

        state["status"] = "deploying model"
        print("  [A100] Deploying Nemotron-Mini-4B + fresh LoRA to A100...")
        runner.deploy_inference_server(port=8000)

        state["status"] = "opening SSH tunnel"
        print("  [A100] Opening SSH tunnel (localhost:9090 → A100:8000)...")
        tunnel = runner.start_ssh_tunnel(remote_port=8000, local_port=9090)
        state["tunnel"] = tunnel

        state["status"] = "waiting for model"
        print("  [A100] Waiting for model to load (this takes ~60-90s)...")
        if runner.wait_for_server(local_port=9090, timeout_sec=180):
            state["ready"] = True
            state["status"] = "LIVE — fine-tuning active"
            print("  [A100] Model loaded and ready for online training!")
        else:
            state["status"] = "server timeout — retrying next cycle"
            print("  [A100] WARNING: Server didn't start in time. Will retry.")

    except Exception as e:
        state["status"] = f"error: {e}"
        print(f"  [A100] Init failed: {e}")
        print("  [A100] Continuing without GPU fine-tuning...")

    return state


def do_online_train(a100_state: dict, traces: list[dict], chat_log: list[dict], cycle: int) -> None:
    """Send training traces to the A100 for LoRA gradient updates."""
    runner = a100_state.get("runner")
    if not runner or not a100_state.get("ready"):
        return

    try:
        result = runner.online_train(
            traces=traces,
            local_port=9090,
            learning_rate=1e-4,
            max_steps=len(traces),
        )
        if result.get("status") == "ok":
            steps = result.get("steps", 0)
            loss = result.get("loss", 0)
            total = result.get("total_steps", 0)
            a100_state["train_steps"] = total
            a100_state["status"] = f"LIVE — {total} steps, loss={loss:.4f}"
            chat_log.append({
                "role": "system",
                "kind": "TRAINING",
                "text": f"Online LoRA update: {steps} gradient steps, loss={loss:.4f}, total_steps={total}",
                "cycle": cycle,
            })
        else:
            err = result.get("error", "unknown")
            a100_state["status"] = f"train error: {err}"
    except Exception as e:
        a100_state["status"] = f"train error: {e}"
        logger.warning("Online train failed: %s", e)


# ── Heuristic agent — reads grid state, returns actions ──────────

def heuristic_decide(state: dict) -> tuple[list[dict], str, str]:
    """Analyse grid state and return (actions, assessment, risk_level).

    This mimics what the LLM agent does: looks at buses, lines, storage,
    and decides what to redispatch / charge / shed / switch.
    """
    actions: list[dict] = []
    used_targets: set[str] = set()   # avoid duplicate targets
    problems: list[str] = []
    buses = state.get("buses", [])
    lines = state.get("lines", [])
    storage = state.get("storage", [])

    bus_map = {b["bus_id"]: b for b in buses}
    total_load = sum(b.get("load_mw", 0) for b in buses)
    total_gen = sum(b.get("generation_mw", 0) for b in buses)

    # ── 1. Find problems ────────────────────────────────────
    congested = sorted(
        [l for l in lines if l["loading_pct"] > 85],
        key=lambda l: l["loading_pct"], reverse=True,
    )
    low_v = sorted(
        [b for b in buses if b["voltage_pu"] < 0.95],
        key=lambda b: b["voltage_pu"],
    )
    critical_v = [b for b in low_v if b["voltage_pu"] < 0.93]

    for line in congested:
        problems.append(f"{line['line_id']} at {line['loading_pct']:.0f}%")
    for bus in low_v:
        problems.append(f"{bus['bus_id']} V={bus['voltage_pu']:.3f}")

    # ── 2. CRITICAL VOLTAGE FIRST — shed load at the problem bus ──
    for vbus in critical_v:
        if len(actions) >= 6:
            break
        vid = vbus["bus_id"]
        if vid in used_targets:
            continue
        cur_load = vbus.get("load_mw", 0)
        if cur_load > 20:
            shed = round(min(cur_load * 0.3, 40), 1)
            actions.append({
                "type": "LOAD_SHAPE",
                "target": vid,
                "params": {"delta_mw": -shed},
                "reason": f"Emergency shed {shed} MW at {vid} (V={vbus['voltage_pu']:.3f} pu, load {cur_load:.0f}→{cur_load-shed:.0f} MW)",
            })
            used_targets.add(vid)

        # Also boost nearest generator to push power toward this bus
        # Find lines connected to this bus and boost gen at the other end
        for line in lines:
            if len(actions) >= 6:
                break
            neighbor_id = None
            if line["to_bus"] == vid:
                neighbor_id = line["from_bus"]
            elif line["from_bus"] == vid:
                neighbor_id = line["to_bus"]
            if not neighbor_id or neighbor_id in used_targets:
                continue
            nb = bus_map.get(neighbor_id, {})
            if nb.get("generation_mw", 0) > 10:
                cur_gen = nb["generation_mw"]
                bump = round(min(50, cur_gen * 0.3), 1)
                actions.append({
                    "type": "REDISPATCH",
                    "target": neighbor_id,
                    "params": {"target_mw": round(cur_gen + bump, 1)},
                    "reason": f"Boost {neighbor_id} gen {cur_gen:.0f}→{cur_gen+bump:.0f} MW to raise voltage at {vid} ({vbus['voltage_pu']:.3f} pu)",
                })
                used_targets.add(neighbor_id)
                break  # one gen boost per voltage bus

    # ── 3. STORAGE — fast response for congestion or voltage ──
    if storage and len(actions) < 6:
        s = storage[0]
        if (congested or low_v) and s["soc_pct"] > 15:
            discharge = round(min(40, s["soc_pct"] * 0.4), 1)
            actions.append({
                "type": "STORAGE",
                "target": s["unit_id"],
                "params": {"setpoint_mw": discharge, "mode": "discharge"},
                "reason": f"Discharge BESS {discharge} MW (SoC {s['soc_pct']:.0f}%) to support grid",
            })
            used_targets.add(s["unit_id"])
        elif not congested and not low_v and s["soc_pct"] < 70 and total_gen > total_load + 10:
            actions.append({
                "type": "STORAGE",
                "target": s["unit_id"],
                "params": {"setpoint_mw": -25, "mode": "charge"},
                "reason": f"Charge BESS (SoC {s['soc_pct']:.0f}%→{min(100,s['soc_pct']+5):.0f}%) during stable period",
            })
            used_targets.add(s["unit_id"])

    # ── 4. REDISPATCH for congestion ──────────────────────────
    if congested and len(actions) < 6:
        worst = congested[0]
        to_bus = bus_map.get(worst["to_bus"], {})
        from_bus = bus_map.get(worst["from_bus"], {})

        if to_bus.get("generation_mw", 0) > 0 and worst["to_bus"] not in used_targets:
            cur = to_bus["generation_mw"]
            bump = round(min(30, max(10, (worst["loading_pct"] - 80) * 0.5)), 1)
            actions.append({
                "type": "REDISPATCH",
                "target": worst["to_bus"],
                "params": {"target_mw": round(cur + bump, 1)},
                "reason": f"Ramp {worst['to_bus']} gen {cur:.0f}→{cur+bump:.0f} MW to relieve {worst['line_id']} ({worst['loading_pct']:.0f}%)",
            })
            used_targets.add(worst["to_bus"])

        if from_bus.get("generation_mw", 0) > 20 and worst["from_bus"] not in used_targets and len(actions) < 6:
            cur = from_bus["generation_mw"]
            cut = round(min(25, max(10, (worst["loading_pct"] - 80) * 0.4)), 1)
            new_mw = round(max(10, cur - cut), 1)
            actions.append({
                "type": "REDISPATCH",
                "target": worst["from_bus"],
                "params": {"target_mw": new_mw},
                "reason": f"Cut {worst['from_bus']} gen {cur:.0f}→{new_mw} MW to lower flow on {worst['line_id']}",
            })
            used_targets.add(worst["from_bus"])

    # ── 5. Moderate voltage support (0.93–0.95) ──────────────
    moderate_v = [b for b in low_v if b["voltage_pu"] >= 0.93 and b["bus_id"] not in used_targets]
    for vbus in moderate_v:
        if len(actions) >= 6:
            break
        gens = sorted(
            [b for b in buses if b["generation_mw"] > 10 and b["bus_id"] not in used_targets],
            key=lambda b: b["generation_mw"], reverse=True,
        )
        if gens:
            g = gens[0]
            cur = g["generation_mw"]
            bump = round(min(30, cur * 0.15), 1)
            actions.append({
                "type": "REDISPATCH",
                "target": g["bus_id"],
                "params": {"target_mw": round(cur + bump, 1)},
                "reason": f"Boost {g['bus_id']} gen {cur:.0f}→{cur+bump:.0f} MW to raise {vbus['bus_id']} voltage ({vbus['voltage_pu']:.3f} pu)",
            })
            used_targets.add(g["bus_id"])

    # ── 6. LOAD_SHAPE — shed flex load if heavily congested ───
    if len(congested) >= 2 and len(actions) < 6:
        flex_targets = [b for b in buses
                        if b.get("load_mw", 0) > 30
                        and b["bus_id"] not in used_targets]
        if flex_targets:
            fb = max(flex_targets, key=lambda b: b.get("load_mw", 0))
            cur_load = fb["load_mw"]
            shed = round(min(20, cur_load * 0.1), 1)
            actions.append({
                "type": "LOAD_SHAPE",
                "target": fb["bus_id"],
                "params": {"delta_mw": -shed},
                "reason": f"Shed {fb['bus_id']} load {cur_load:.0f}→{cur_load-shed:.0f} MW — {len(congested)} lines congested",
            })
            used_targets.add(fb["bus_id"])

    # ── Build assessment ────────────────────────────────────
    if not problems:
        assessment = "Grid nominal — all lines <85% loading, voltages within [0.95, 1.05] pu. Monitoring for disturbances."
        risk = "low"
    elif critical_v:
        assessment = f"CRITICAL VOLTAGE: {critical_v[0]['bus_id']} at {critical_v[0]['voltage_pu']:.3f} pu. Emergency load shed + gen boost. {', '.join(problems[:2])}"
        risk = "critical"
    elif len(congested) >= 3 or (congested and low_v):
        assessment = f"CRITICAL: {', '.join(problems[:3])}. Redispatch + storage + load shed."
        risk = "critical"
    elif congested:
        assessment = f"High stress: {', '.join(problems[:3])}. Redispatching generators and deploying storage."
        risk = "high"
    elif low_v:
        assessment = f"Moderate: {', '.join(problems[:2])}. Boosting generation for voltage support."
        risk = "medium"
    else:
        assessment = "Grid stable. Monitoring."
        risk = "low"

    return actions[:6], assessment, risk


def actions_to_gym(actions: list[dict], env: GridEnv) -> dict[str, np.ndarray]:
    """Convert heuristic action dicts to gym action arrays."""
    scenario = env._scenario
    n_gen = len(env._gen_indices)
    n_storage = max(len(scenario.storage), 1)
    n_flex = max(len(env._flex_indices), 1)
    n_branch = len(scenario.branches)

    redispatch = np.zeros(n_gen, dtype=np.float32)
    storage_act = np.zeros(n_storage, dtype=np.float32)
    load_shape = np.zeros(n_flex, dtype=np.float32)
    topology = np.ones(n_branch, dtype=np.int8)

    bus_map = {b.bus_id: b for b in scenario.buses}
    gen_idx_map = {scenario.buses[gi].bus_id: i for i, gi in enumerate(env._gen_indices)}
    flex_idx_map = {scenario.buses[fi].bus_id: i for i, fi in enumerate(env._flex_indices)}

    for a in actions:
        atype = a.get("type", "").upper()
        target = a.get("target", "")
        params = a.get("params", {})

        if atype == "REDISPATCH" and target in gen_idx_map:
            idx = gen_idx_map[target]
            bus = bus_map[target]
            target_mw = params.get("target_mw", bus.gen_mw)
            delta = target_mw - bus.gen_mw
            # Normalise to [-1, 1] range (max 20% swing)
            max_swing = bus.gen_max_mw * 0.2
            redispatch[idx] = np.clip(delta / max(max_swing, 1), -1, 1)

        elif atype == "STORAGE":
            setpoint = params.get("setpoint_mw", 0)
            if scenario.storage:
                max_p = scenario.storage[0].max_power_mw
                storage_act[0] = np.clip(setpoint / max(max_p, 1), -1, 1)

        elif atype == "LOAD_SHAPE" and target in flex_idx_map:
            idx = flex_idx_map[target]
            bus = bus_map[target]
            delta = params.get("delta_mw", 0)
            flex_range = bus.flexible_load_mw
            load_shape[idx] = np.clip(delta / max(flex_range, 1), -1, 1)

        elif atype == "TOPOLOGY":
            for i, br in enumerate(scenario.branches):
                if br.branch_id == target:
                    topology[i] = 0 if params.get("action") == "open" else 1

    return {
        "redispatch": redispatch,
        "storage": storage_act,
        "load_shape": load_shape,
        "topology": topology,
    }


# ── Main loop ────────────────────────────────────────────────

def run_demo(port: int = 8080, scenario_name: str = "6-bus", interval: float = 1.5):
    from frontend.server import dashboard_state, run_server

    # Start web server
    web_thread = threading.Thread(
        target=run_server, kwargs={"port": port}, daemon=True,
    )
    web_thread.start()
    print(f"\n  Conducto Grid Demo + A100 Fine-tuning")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    # Initialize A100 for online training
    a100 = init_a100()
    training_traces: list[dict] = []
    TRAIN_EVERY = 3  # send traces every N cycles

    # Create grid env
    scenario_fn = SCENARIOS.get(scenario_name)
    if not scenario_fn:
        print(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIOS.keys())}")
        return
    env = GridEnv(scenario_fn(), enable_growth=True)
    obs, info = env.reset()

    dashboard_state.update({"topology": {"name": scenario_name}})

    cycle = 0
    episode = 1
    step_in_ep = 0
    all_rewards: list[float] = []
    recent_decisions: list[dict] = []
    chat_log: list[dict] = []
    _prev_chat_count = 0  # track how many env chat_events we've already seen

    try:
        while True:
            cycle += 1
            step_in_ep += 1

            # 1. Read grid state
            state = env.get_grid_state()
            state_dict = state.to_dict()

            # 2. Heuristic agent decides
            t0 = time.time()
            actions, assessment, risk = heuristic_decide(state_dict)
            latency_ms = (time.time() - t0) * 1000

            # 3. Convert to gym action & step
            gym_action = actions_to_gym(actions, env)
            obs, reward, terminated, truncated, step_info = env.step(gym_action)

            all_rewards.append(reward)

            # 4. Read NEW state after actions applied
            new_state = env.get_grid_state()
            new_state_dict = new_state.to_dict()
            new_state_dict["active_events"] = getattr(env, "active_events", [])[-5:]

            # 5. Track decisions
            recent_decisions.append({
                "cycle": cycle,
                "risk_level": risk,
                "assessment": assessment[:120],
                "n_actions": len(actions),
                "reward": reward,
                "latency_ms": latency_ms,
            })
            recent_decisions = recent_decisions[-8:]

            # 5b. Build chat messages from system events + agent thinking + actions
            # First, pull any new system events from the env
            env_chats = getattr(env, "chat_events", [])
            new_system_msgs = env_chats[_prev_chat_count:]
            _prev_chat_count = len(env_chats)
            for ev in new_system_msgs:
                chat_log.append({
                    "role": "system",
                    "kind": ev.get("kind", "EVENT"),
                    "text": ev["message"],
                    "cycle": cycle,
                })

            # Agent thinking message
            chat_log.append({
                "role": "thinking",
                "text": assessment,
                "risk": risk,
                "cycle": cycle,
            })

            # Agent actions message
            if actions:
                action_lines = []
                for a in actions:
                    action_lines.append(
                        f"{a['type']} → {a['target']}: {a.get('reason', '')}"
                    )
                chat_log.append({
                    "role": "action",
                    "text": "\n".join(action_lines),
                    "actions": actions,
                    "cycle": cycle,
                })
            else:
                chat_log.append({
                    "role": "action",
                    "text": "No actions needed — grid nominal.",
                    "actions": [],
                    "cycle": cycle,
                })

            # Keep chat log bounded
            chat_log = chat_log[-80:]

            # 5c. Collect training trace from this cycle's decision
            actions_json = json.dumps(actions)
            messages = [
                {"role": "system", "content": "You are Conducto, an AI grid pressure autopilot. Analyze the grid state and return JSON actions to maintain stability."},
                {"role": "user", "content": state.to_context_block()},
            ]
            training_traces.append({
                "messages": messages,
                "raw_response": actions_json,
            })
            training_traces = training_traces[-12:]  # keep last 12

            # 5d. Send traces to A100 for online LoRA training every N cycles
            if cycle % TRAIN_EVERY == 0 and len(training_traces) >= TRAIN_EVERY:
                traces_to_send = training_traces[-TRAIN_EVERY:]
                do_online_train(a100, traces_to_send, chat_log, cycle)

            # 6. Compute metrics
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            recent_10 = all_rewards[-10:] if len(all_rewards) >= 10 else all_rewards
            recent_avg = sum(recent_10) / len(recent_10) if recent_10 else 0

            if len(all_rewards) >= 20:
                first_half = all_rewards[:len(all_rewards)//2]
                second_half = all_rewards[len(all_rewards)//2:]
                trend = "improving" if sum(second_half)/len(second_half) > sum(first_half)/len(first_half) else "declining"
            else:
                trend = "stable"

            # 7. Push to web dashboard
            dashboard_state.update({
                "cycle": cycle,
                "episode": episode,
                "step_in_episode": step_in_ep,
                "grid": new_state_dict,
                "chat": chat_log[-40:],
                "agent": {
                    "recent_decisions": recent_decisions,
                    "latest_actions": actions,
                    "latest_assessment": assessment[:200],
                    "latest_risk": risk,
                },
                "metrics": {
                    "rewards": all_rewards[-30:],
                    "avg_reward": avg_reward,
                    "recent_avg_reward": recent_avg,
                    "reward_trend": trend,
                    "online_train_steps": a100.get("train_steps", 0),
                    "using_finetuned": a100.get("ready", False),
                    "finetune_status": a100.get("status", "offline"),
                },
            })

            # 8. Log to terminal
            n_alerts = len(step_info.get("alerts", []))
            action_types = [a["type"] for a in actions]
            r_color = "\033[32m" if reward > 0 else "\033[31m" if reward < -5 else "\033[33m"
            print(
                f"  C{cycle:>3d}  ep{episode}  "
                f"reward={r_color}{reward:+6.1f}\033[0m  "
                f"alerts={n_alerts}  "
                f"max_load={step_info['max_loading_pct']:.0f}%  "
                f"actions=[{', '.join(action_types)}]  "
                f"risk={risk}"
            )

            # 9. Episode boundary
            if terminated or truncated:
                ep_rewards = all_rewards[-step_in_ep:] if step_in_ep > 0 else [0]
                avg = sum(ep_rewards) / len(ep_rewards)
                print(f"\n  --- Episode {episode} done ({step_in_ep} steps, avg reward {avg:+.1f}) ---\n")
                obs, _ = env.reset()
                episode += 1
                step_in_ep = 0
                _prev_chat_count = 0

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n  Demo stopped. {cycle} cycles, {episode} episodes.")
        if all_rewards:
            print(f"  Final avg reward: {sum(all_rewards)/len(all_rewards):+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conducto Grid Demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scenario", default="6-bus", choices=list(SCENARIOS.keys()))
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between cycles")
    args = parser.parse_args()
    run_demo(port=args.port, scenario_name=args.scenario, interval=args.interval)
