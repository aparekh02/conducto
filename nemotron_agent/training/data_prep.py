"""Data preparation — converts raw traces into SFT and RL training datasets.

SFT dataset:  prompt/completion pairs from high-reward or operator-approved traces.
RL dataset:   full trajectories with reward signals for PPO/DPO training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nemotron_agent.trace_buffer import TraceBuffer

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets"


class DataPrep:
    """Samples from the trace buffer and builds training-ready datasets."""

    def __init__(
        self,
        buffer: TraceBuffer,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.buffer = buffer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── SFT dataset ──────────────────────────────────────

    def build_sft_dataset(
        self,
        max_samples: int = 500,
        min_reward: float = 0.0,
        only_approved: bool = True,
    ) -> Path:
        """Build a JSONL of prompt/completion pairs from good decisions.

        Format per line (compatible with NeMo/TRL SFTTrainer):
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user",   "content": "<grid state>"},
                {"role": "assistant", "content": "<JSON actions>"}
            ]
        }
        """
        traces = self.buffer.sample_traces(
            max_samples * 2,  # fetch more, then rank and filter
            only_with_outcome=True,
            only_approved=only_approved,
            min_reward=min_reward,
        )

        # Fall back to reward-only filtering if no approved traces
        if not traces and only_approved:
            logger.info("No operator-approved traces, falling back to reward filter")
            traces = self.buffer.sample_traces(
                max_samples * 2,
                only_with_outcome=True,
                min_reward=min_reward,
            )

        # Rank by reward and keep the best half (relative ranking)
        if traces:
            traces = sorted(traces, key=lambda t: t.get("reward", -999), reverse=True)
            traces = traces[:max_samples]
            logger.info(
                "SFT: selected top %d traces (reward range: %.1f to %.1f)",
                len(traces),
                traces[0].get("reward", 0) if traces else 0,
                traces[-1].get("reward", 0) if traces else 0,
            )

        out_path = self.output_dir / "sft_train.jsonl"
        count = 0
        with open(out_path, "w") as f:
            for trace in traces:
                messages = trace["messages"]
                # Replace the last user message and append the model's response
                sft_row = {
                    "messages": messages + [
                        {"role": "assistant", "content": trace["raw_response"]}
                    ]
                }
                f.write(json.dumps(sft_row) + "\n")
                count += 1

        logger.info("SFT dataset: %d samples → %s", count, out_path)
        return out_path

    # ── RL dataset ───────────────────────────────────────

    def build_rl_dataset(self, max_samples: int = 1000) -> Path:
        """Build a JSONL of (prompt, response, reward) for RL fine-tuning.

        Format per line:
        {
            "prompt": [system + user messages],
            "response": "<model JSON output>",
            "reward": <float>
        }
        """
        traces = self.buffer.sample_traces(
            max_samples,
            only_with_outcome=True,
        )

        out_path = self.output_dir / "rl_train.jsonl"
        count = 0
        with open(out_path, "w") as f:
            for trace in traces:
                rl_row = {
                    "prompt": trace["messages"],
                    "response": trace["raw_response"],
                    "reward": trace["reward"],
                }
                f.write(json.dumps(rl_row) + "\n")
                count += 1

        logger.info("RL dataset: %d samples → %s", count, out_path)
        return out_path

    # ── Preference pairs for DPO ─────────────────────────

    def build_dpo_dataset(self, max_pairs: int = 250) -> Path:
        """Build preference pairs: (prompt, chosen, rejected) for DPO.

        Pairs a high-reward response with a low-reward response for the
        same or similar grid state.
        """
        all_traces = self.buffer.sample_traces(
            max_pairs * 4,
            only_with_outcome=True,
        )

        # Sort by reward, pair top half with bottom half
        sorted_traces = sorted(all_traces, key=lambda t: t["reward"], reverse=True)
        mid = len(sorted_traces) // 2
        good = sorted_traces[:mid]
        bad = sorted_traces[mid:]

        out_path = self.output_dir / "dpo_train.jsonl"
        count = 0
        with open(out_path, "w") as f:
            for g, b in zip(good, bad):
                pair = {
                    "prompt": g["messages"],
                    "chosen": g["raw_response"],
                    "rejected": b["raw_response"],
                }
                f.write(json.dumps(pair) + "\n")
                count += 1
                if count >= max_pairs:
                    break

        logger.info("DPO dataset: %d pairs → %s", count, out_path)
        return out_path
