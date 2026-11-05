"""Central configuration for the V0 watermark experiments.

All tunable values live here so they are never hard-coded across modules.
The secret key is read from the WATERMARK_SECRET_KEY environment variable
(or a local .env file) and falls back to a placeholder for experimentation.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("WATERMARK_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# ---------------------------------------------------------------------------
# Watermark parameters (Section 57 starting values)
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("WATERMARK_SECRET_KEY", "CHANGE_THIS_KEY")
GREEN_FRACTION = 0.5
WATERMARK_DELTA = 1.0
CONTEXT_WIDTH = 4
Z_THRESHOLD = 4.0  # V0 threshold; must be justified by false-positive analysis

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 200
MIN_NEW_TOKENS = 0
TEMPERATURE = 0.8
TOP_P = 0.95

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
EXPERIMENT_SEED = 42
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "prompts.txt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WatermarkConfig:
    secret_key: str = SECRET_KEY
    green_fraction: float = GREEN_FRACTION
    delta: float = WATERMARK_DELTA
    context_width: int = CONTEXT_WIDTH
    z_threshold: float = Z_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 < self.green_fraction < 1.0:
            raise ValueError("green_fraction must be in (0, 1)")
        if self.context_width < 1:
            raise ValueError("context_width must be >= 1")

    def public_dict(self) -> dict:
        """Parameters safe to log (the secret key is never written to disk)."""
        d = asdict(self)
        d.pop("secret_key")
        return d


@dataclass
class GenerationConfig:
    max_new_tokens: int = MAX_NEW_TOKENS
    min_new_tokens: int = MIN_NEW_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use sampling, not argmax)")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
