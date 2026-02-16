"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the nemotron_agent package directory
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)


def _require(var: str) -> str:
    val = os.getenv(var)
    if not val or val.startswith("XXXX"):
        raise RuntimeError(
            f"Environment variable {var} is not set. "
            f"Fill it in at {_ENV_PATH}"
        )
    return val


@dataclass(frozen=True)
class NVIDIAConfig:
    api_key: str = field(default_factory=lambda: _require("NVIDIA_API_KEY"))
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )
    model_id: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_MODEL_ID", "nvidia/llama-3.3-nemotron-super-49b-v1"
        )
    )


@dataclass(frozen=True)
class VESSLConfig:
    # API-based access (preferred — shows in dashboard)
    api_token: str = field(
        default_factory=lambda: os.getenv("VESSL_API_TOKEN", "")
    )
    organization: str = field(
        default_factory=lambda: os.getenv("VESSL_ORGANIZATION", "")
    )
    project: str = field(
        default_factory=lambda: os.getenv("VESSL_PROJECT", "conducto")
    )
    cluster: str = field(
        default_factory=lambda: os.getenv("VESSL_CLUSTER", "vessl-kr-a100-80g-sxm")
    )
    gpu_preset: str = field(
        default_factory=lambda: os.getenv("VESSL_GPU_PRESET", "gpu-a100-80g-small")
    )

    # SSH fallback (for workspace / manual debugging)
    ssh_host: str = field(
        default_factory=lambda: os.getenv("VESSL_SSH_HOST", "")
    )
    ssh_port: int = field(
        default_factory=lambda: int(os.getenv("VESSL_SSH_PORT", "22"))
    )
    ssh_key_path: str = field(
        default_factory=lambda: os.getenv("VESSL_SSH_KEY_PATH", "~/.ssh/vessl_key")
    )
    workspace_dir: str = field(
        default_factory=lambda: os.getenv("VESSL_WORKSPACE_DIR", "/root/conducto")
    )

    @property
    def has_api(self) -> bool:
        return bool(self.api_token and self.organization)


@dataclass(frozen=True)
class AgentConfig:
    loop_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("AGENT_LOOP_INTERVAL_SEC", "5"))
    )
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "131072"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.6"))
    )
    max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "4096"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "INFO")
    )


@dataclass(frozen=True)
class Config:
    nvidia: NVIDIAConfig = field(default_factory=NVIDIAConfig)
    vessl: VESSLConfig = field(default_factory=VESSLConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
