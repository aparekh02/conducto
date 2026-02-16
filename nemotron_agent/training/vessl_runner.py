"""VESSL runner — launch fine-tuning via the VESSL API (or SSH fallback).

Primary path (VESSL API):
  1. Renders a YAML template with dataset paths and hyperparams
  2. Calls `vessl run create -f <yaml>` to launch a managed GPU run
  3. Polls `vessl run read <run_id>` until complete
  4. Downloads adapters from the VESSL artifact store

Fallback path (SSH):
  If VESSL_API_TOKEN is not set, falls back to direct SSH to the
  workspace (the old behaviour).

Monitor runs at:  https://app.vessl.ai
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from nemotron_agent.config import VESSLConfig

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent
SCRIPTS_DIR = Path(__file__).parent / "scripts"

# ── Run status constants ──────────────────────────────────
_TERMINAL_STATES = {"completed", "failed", "terminated", "error"}
_POLL_INTERVAL = 30  # seconds between status checks


# ═══════════════════════════════════════════════════════════
#  VESSL API path — uses `vessl` CLI under the hood
# ═══════════════════════════════════════════════════════════

def _vessl_cli(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a vessl CLI command."""
    cmd = ["vessl", *args]
    logger.info("vessl %s", " ".join(args))
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture,
        text=True,
    )


def _render_yaml(template_name: str, variables: dict) -> Path:
    """Fill in a YAML template and write to a temp file."""
    template_path = TEMPLATES_DIR / template_name
    content = template_path.read_text()
    for key, val in variables.items():
        content = content.replace(f"{{{key}}}", str(val))
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="vessl_run_", delete=False,
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _extract_run_id(output: str) -> str | None:
    """Parse the run ID from `vessl run create` output."""
    # VESSL CLI prints something like: "Run #12345 created"
    match = re.search(r"#(\d+)", output)
    if match:
        return match.group(1)
    # Also try "run_id: XXXX" or just a bare number on a line
    match = re.search(r"(?:run[_\s]?id[:\s]+)?(\d{4,})", output, re.IGNORECASE)
    return match.group(1) if match else None


def _poll_run(run_id: str, timeout_sec: int = 7200) -> str:
    """Poll `vessl run read` until the run reaches a terminal state.

    Returns the final status string.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        result = _vessl_cli("run", "read", run_id, capture=True)
        output = result.stdout.lower()
        for state in _TERMINAL_STATES:
            if state in output:
                logger.info("Run %s reached state: %s", run_id, state)
                return state
        logger.debug("Run %s still running... (%.0fs elapsed)", run_id, time.time() - start)
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"Run {run_id} did not complete within {timeout_sec}s")


def _download_artifact(artifact_name: str, local_dir: Path) -> None:
    """Download a VESSL artifact to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    _vessl_cli("artifact", "download", artifact_name, "-o", str(local_dir))
    logger.info("Artifact %s downloaded to %s", artifact_name, local_dir)


# ═══════════════════════════════════════════════════════════
#  SSH fallback — direct workspace access
# ═══════════════════════════════════════════════════════════

def _ssh_cmd(cfg: VESSLConfig) -> list[str]:
    key_path = str(Path(cfg.ssh_key_path).expanduser())
    return [
        "ssh", "-i", key_path,
        "-p", str(cfg.ssh_port),
        "-o", "StrictHostKeyChecking=no",
        cfg.ssh_host,
    ]


def _rsync_to(cfg: VESSLConfig, local_path: str, remote_path: str) -> None:
    key_path = str(Path(cfg.ssh_key_path).expanduser())
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh -i {key_path} -p {cfg.ssh_port} -o StrictHostKeyChecking=no",
        local_path, f"{cfg.ssh_host}:{remote_path}",
    ]
    logger.info("Uploading %s → %s:%s", local_path, cfg.ssh_host, remote_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _rsync_from(cfg: VESSLConfig, remote_path: str, local_path: str) -> None:
    key_path = str(Path(cfg.ssh_key_path).expanduser())
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh -i {key_path} -p {cfg.ssh_port} -o StrictHostKeyChecking=no",
        f"{cfg.ssh_host}:{remote_path}", local_path,
    ]
    logger.info("Downloading %s:%s → %s", cfg.ssh_host, remote_path, local_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _ssh_run(cfg: VESSLConfig, command: str) -> subprocess.CompletedProcess:
    full_cmd = _ssh_cmd(cfg) + [command]
    logger.info("Running on VESSL: %s", command)
    return subprocess.run(full_cmd, check=True, capture_output=True, text=True)


def _ssh_bg(cfg: VESSLConfig, command: str) -> None:
    """Run a background command via SSH without blocking.

    For commands like 'nohup ... & disown' that should detach from SSH.
    Uses Popen with DEVNULL to avoid the pipe-blocking hang that occurs
    when capture_output=True is used with background SSH commands.
    """
    full_cmd = _ssh_cmd(cfg) + [command]
    logger.info("Running background on VESSL: %s", command)
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    # Wait briefly for SSH to send the command
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # SSH didn't exit in time — the remote process is running via nohup
        proc.kill()
        proc.wait()


# ═══════════════════════════════════════════════════════════
#  VESSLRunner — unified interface
# ═══════════════════════════════════════════════════════════

class VESSLRunner:
    """Manages fine-tuning jobs via the VESSL API (with SSH fallback).

    If VESSL_API_TOKEN is set → uses `vessl run create` (shows in dashboard).
    Otherwise → falls back to direct SSH to the workspace.
    """

    def __init__(self, cfg: VESSLConfig | None = None) -> None:
        self.cfg = cfg or VESSLConfig()
        self._use_api = self.cfg.has_api
        self._remote_dir = self.cfg.workspace_dir
        self._last_run_id: str | None = None

        if self._use_api:
            # Configure vessl CLI with the token
            os.environ["VESSL_ACCESS_TOKEN"] = self.cfg.api_token
            if self.cfg.organization:
                os.environ["VESSL_ORGANIZATION"] = self.cfg.organization
            logger.info(
                "VESSLRunner ready (API mode)  org=%s  project=%s  cluster=%s",
                self.cfg.organization, self.cfg.project, self.cfg.cluster,
            )
        elif self.cfg.ssh_host:
            logger.info(
                "VESSLRunner ready (SSH fallback)  host=%s  dir=%s",
                self.cfg.ssh_host, self._remote_dir,
            )
        else:
            raise RuntimeError(
                "Neither VESSL_API_TOKEN nor VESSL_SSH_HOST is set in .env.\n"
                "Set up VESSL API:  pip install vessl && vessl configure\n"
                "Or set VESSL_SSH_HOST for direct workspace access."
            )

    @property
    def last_run_id(self) -> str | None:
        """The run ID from the most recent VESSL API launch."""
        return self._last_run_id

    @property
    def dashboard_url(self) -> str:
        """URL to monitor runs in the VESSL dashboard."""
        org = self.cfg.organization
        project = self.cfg.project
        if org and project:
            return f"https://app.vessl.ai/{org}/{project}/runs"
        return "https://app.vessl.ai"

    # ── Setup ──────────────────────────────────────────────

    def setup_workspace(self) -> None:
        """Ensure the remote environment is ready."""
        if self._use_api:
            # API mode: deps are installed in the run container, nothing to do
            logger.info("API mode — workspace setup handled by run container")
            return
        # SSH fallback
        _ssh_run(self.cfg, f"mkdir -p {self._remote_dir}/scripts {self._remote_dir}/data {self._remote_dir}/output")
        _ssh_run(self.cfg, "pip install -q 'transformers==4.45.1' peft 'trl==0.12.2' datasets bitsandbytes uvicorn fastapi")
        _rsync_to(self.cfg, str(SCRIPTS_DIR) + "/", f"{self._remote_dir}/scripts/")
        logger.info("Workspace setup complete")

    # ── SFT training ───────────────────────────────────────

    def launch_sft(
        self,
        dataset_path: Path | str,
        *,
        model_id: str = "nvidia/Nemotron-Mini-4B-Instruct",
        epochs: int = 1,
        learning_rate: float = 2e-5,
        lora_rank: int = 16,
        wait: bool = True,
    ) -> str | None:
        """Launch SFT training. Returns run_id if using API."""
        dataset_path = Path(dataset_path)

        if self._use_api:
            return self._launch_api_run(
                template="vessl_sft.yaml",
                variables={
                    "run_tag": f"v{int(time.time())}",
                    "cluster": self.cfg.cluster,
                    "gpu_preset": self.cfg.gpu_preset,
                    "model_id": model_id,
                    "local_data_dir": str(dataset_path.parent),
                    "sft_filename": dataset_path.name,
                    "sft_samples": sum(1 for _ in open(dataset_path)),
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "batch_size": 2,
                },
                wait=wait,
            )

        # SSH fallback
        _rsync_to(self.cfg, str(dataset_path), f"{self._remote_dir}/data/")
        _rsync_to(self.cfg, str(SCRIPTS_DIR / "sft_train.py"), f"{self._remote_dir}/scripts/")
        remote_data = f"{self._remote_dir}/data/{dataset_path.name}"
        remote_output = f"{self._remote_dir}/output/sft-adapters"
        _ssh_run(self.cfg, (
            f"cd {self._remote_dir} && python scripts/sft_train.py "
            f"--model_id {model_id} "
            f"--data_path {remote_data} "
            f"--output_dir {remote_output} "
            f"--epochs {epochs} "
            f"--learning_rate {learning_rate} "
            f"--lora_rank {lora_rank}"
        ))
        logger.info("SFT training complete. Adapters at %s:%s", self.cfg.ssh_host, remote_output)
        return None

    # ── DPO / RL training ──────────────────────────────────

    def launch_rl(
        self,
        dataset_path: Path | str,
        *,
        model_id: str = "nvidia/Nemotron-Mini-4B-Instruct",
        method: str = "dpo",
        epochs: int = 1,
        learning_rate: float = 5e-6,
        lora_rank: int = 16,
        wait: bool = True,
    ) -> str | None:
        """Launch DPO/RL training. Returns run_id if using API."""
        dataset_path = Path(dataset_path)

        if self._use_api:
            return self._launch_api_run(
                template="vessl_dpo.yaml",
                variables={
                    "run_tag": f"v{int(time.time())}",
                    "cluster": self.cfg.cluster,
                    "gpu_preset": self.cfg.gpu_preset,
                    "model_id": model_id,
                    "local_data_dir": str(dataset_path.parent),
                    "dpo_filename": dataset_path.name,
                    "dpo_pairs": sum(1 for _ in open(dataset_path)),
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "batch_size": 1,
                },
                wait=wait,
            )

        # SSH fallback
        _rsync_to(self.cfg, str(dataset_path), f"{self._remote_dir}/data/")
        _rsync_to(self.cfg, str(SCRIPTS_DIR / "rl_train.py"), f"{self._remote_dir}/scripts/")
        remote_data = f"{self._remote_dir}/data/{dataset_path.name}"
        remote_output = f"{self._remote_dir}/output/rl-adapters"
        _ssh_run(self.cfg, (
            f"cd {self._remote_dir} && python scripts/rl_train.py "
            f"--model_id {model_id} "
            f"--data_path {remote_data} "
            f"--output_dir {remote_output} "
            f"--method {method} "
            f"--epochs {epochs} "
            f"--learning_rate {learning_rate} "
            f"--lora_rank {lora_rank}"
        ))
        logger.info("RL training complete. Adapters at %s:%s", self.cfg.ssh_host, remote_output)
        return None

    # ── Deploy inference server ──────────────────────────────

    def deploy_inference_server(
        self,
        adapter_path: str | None = None,
        port: int = 8000,
        model_id: str = "nvidia/Nemotron-Mini-4B-Instruct",
    ) -> None:
        """Start a PEFT inference server on the VESSL workspace.

        After SFT training, this loads the base model + LoRA adapters and
        serves NIM-compatible chat completions on the given port.
        """
        if not self.cfg.ssh_host:
            raise RuntimeError("deploy_inference_server requires SSH access")

        # Upload the serve script
        _rsync_to(self.cfg, str(SCRIPTS_DIR / "peft_serve.py"), f"{self._remote_dir}/scripts/")

        # Kill any existing inference server
        try:
            _ssh_run(self.cfg, "pkill -f peft_serve.py || true")
            time.sleep(2)
        except subprocess.CalledProcessError:
            pass

        # Install serving deps
        _ssh_run(self.cfg, "pip install -q uvicorn fastapi")

        # Start the server in background (non-blocking)
        adapter_arg = f"--adapter_path {adapter_path}" if adapter_path else ""
        _ssh_bg(self.cfg, (
            f"cd {self._remote_dir} && "
            f"nohup python scripts/peft_serve.py "
            f"--model_id {model_id} "
            f"{adapter_arg} "
            f"--port {port} "
            f"< /dev/null > serve.log 2>&1 & disown"
        ))
        logger.info("Inference server starting on VESSL port %d", port)

    def online_train(
        self,
        traces: list[dict],
        local_port: int = 9090,
        learning_rate: float = 1e-5,
        max_steps: int = 0,
    ) -> dict:
        """Send recent traces to the running inference server for online LoRA updates.

        Returns the /train response dict with steps, loss, etc.
        """
        import urllib.request
        url = f"http://localhost:{local_port}/train"
        payload = json.dumps({
            "traces": traces,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
        }).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read().decode())
            logger.info(
                "Online train: %d steps, loss=%.4f, total_steps=%d",
                result.get("steps", 0),
                result.get("loss", 0),
                result.get("total_steps", 0),
            )
            return result
        except Exception as e:
            logger.error("Online train failed: %s", e)
            return {"status": "error", "error": str(e)}

    def save_checkpoint(self, local_port: int = 9090, path: str = "/root/conducto/output/online-adapters") -> bool:
        """Tell the running server to save current LoRA weights to disk."""
        import urllib.request
        url = f"http://localhost:{local_port}/save?path={path}"
        req = urllib.request.Request(url, data=b"", method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            logger.info("Checkpoint saved: %s", result)
            return result.get("status") == "ok"
        except Exception as e:
            logger.error("Save checkpoint failed: %s", e)
            return False

    def tail_server_log(self, lines: int = 3) -> str:
        """Read the last N lines of serve.log from the remote server."""
        try:
            result = _ssh_run(self.cfg, f"tail -{lines} {self._remote_dir}/serve.log 2>/dev/null || echo ''")
            return result.stdout.strip()
        except Exception:
            return ""

    def reload_adapters(self, adapter_path: str, port: int = 8000) -> bool:
        """Hot-reload new adapters on an already-running inference server."""
        if not self.cfg.ssh_host:
            return False
        try:
            _ssh_run(self.cfg, (
                f'curl -s -X POST "http://localhost:{port}/reload?adapter_path={adapter_path}"'
            ))
            logger.info("Adapters reloaded at %s", adapter_path)
            return True
        except subprocess.CalledProcessError:
            logger.warning("Adapter reload failed — will restart server")
            return False

    def start_ssh_tunnel(
        self,
        remote_port: int = 8000,
        local_port: int = 9090,
    ) -> subprocess.Popen:
        """Start SSH local port forwarding to access VESSL inference server.

        Returns the tunnel subprocess (keep alive for the session).
        """
        key_path = str(Path(self.cfg.ssh_key_path).expanduser())
        cmd = [
            "ssh", "-i", key_path,
            "-p", str(self.cfg.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-L", f"{local_port}:localhost:{remote_port}",
            "-N",
            self.cfg.ssh_host,
        ]
        logger.info("Starting SSH tunnel: localhost:%d → VESSL:%d", local_port, remote_port)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Give tunnel time to establish
        if proc.poll() is not None:
            raise RuntimeError(f"SSH tunnel died immediately (exit={proc.returncode})")
        return proc

    def wait_for_server(self, local_port: int = 9090, timeout_sec: int = 180) -> bool:
        """Poll the health endpoint until the inference server is ready."""
        import urllib.request
        url = f"http://localhost:{local_port}/health"
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                req = urllib.request.Request(url, method="GET")
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    logger.info("Inference server ready at localhost:%d", local_port)
                    return True
            except Exception:
                pass
            time.sleep(5)
        logger.error("Inference server did not become ready within %ds", timeout_sec)
        return False

    # ── Download adapters ──────────────────────────────────

    def download_adapters(self, local_dir: Path | str, adapter_type: str = "sft") -> None:
        """Download trained adapters to local machine."""
        local_dir = Path(local_dir)

        if self._use_api and self._last_run_id:
            artifact_name = f"conducto-{adapter_type}-v{self._last_run_id}"
            try:
                _download_artifact(artifact_name, local_dir)
                return
            except subprocess.CalledProcessError:
                logger.warning("Artifact download failed, trying SSH fallback...")

        # SSH fallback
        if self.cfg.ssh_host:
            remote = f"{self._remote_dir}/output/{adapter_type}-adapters/"
            _rsync_from(self.cfg, remote, str(local_dir))
            logger.info("Adapters downloaded to %s", local_dir)
        else:
            logger.error("No way to download adapters — neither artifact nor SSH available")

    # ── Run status ─────────────────────────────────────────

    def get_run_status(self, run_id: str | None = None) -> str:
        """Check the status of a VESSL run."""
        rid = run_id or self._last_run_id
        if not rid:
            return "no_run"
        try:
            result = _vessl_cli("run", "read", rid, capture=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def stream_logs(self, run_id: str | None = None) -> None:
        """Stream logs from a running VESSL job (blocking)."""
        rid = run_id or self._last_run_id
        if rid:
            _vessl_cli("run", "logs", rid, "--follow")

    # ── Internal ───────────────────────────────────────────

    def _launch_api_run(
        self,
        template: str,
        variables: dict,
        wait: bool = True,
    ) -> str | None:
        """Render YAML, call `vessl run create`, optionally poll to completion."""
        yaml_path = _render_yaml(template, variables)
        try:
            result = _vessl_cli(
                "run", "create",
                "-f", str(yaml_path),
                capture=True,
            )
            run_id = _extract_run_id(result.stdout)
            if run_id:
                self._last_run_id = run_id
                logger.info(
                    "VESSL run created: #%s — monitor at %s",
                    run_id, self.dashboard_url,
                )
                if wait:
                    status = _poll_run(run_id)
                    if status != "completed":
                        raise RuntimeError(
                            f"VESSL run #{run_id} ended with status: {status}. "
                            f"Check logs: vessl run logs {run_id}"
                        )
                return run_id
            else:
                logger.warning("Could not parse run ID from: %s", result.stdout)
                return None
        finally:
            Path(yaml_path).unlink(missing_ok=True)
