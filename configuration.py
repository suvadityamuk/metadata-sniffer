from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SlackConfig:
    metadata_channel_name = "#hub-model-metadata-snippets-sprint"
    custom_code_channel_name = "#custom-code-repos"
    # debug_channel_name = "#exp-slack-alerts"


@dataclass
class DatasetConfig:
    # Top N trending models of the day
    trending_models_metadata_id: str = "model-metadata/trending_models_metadata"


@dataclass
class ModelCheckerConfig:
    avocado_team_members: List[str] = field(
        default_factory=lambda: [
            "ariG23498",
            "reach-vb",
            "pcuenq",
            "burtenshaw",
            "davanstrien",
            "merve",
            "sergiopaniego",
            "Steveeeeeeen",
            "nielsr",
            "dn6",
            "linoyts",
            "sayakpaul",
        ]
    )
    num_trending_models: int = 100


@dataclass
class ExecuteCodeConfig:
    docker_image: str = "ghcr.io/astral-sh/uv:debian"
    jobs_namespace: str = "suvadityamuk"
    dependencies: List[str] = field(
        default_factory=lambda: [
            "transformers",
            "diffusers",
            "sentence-transformers",
            "torch",
            "torchvision",
            "torchaudio",
            "accelerate",
            "pillow",
            "requests",
            "sentencepiece",
            "protobuf",
            "einops",
            "timm",
            "librosa",
            "soundfile",
            "av",
            "num2words",
            "kernels",
        ]
    )
    # memory needed = weights * memory_overhead + memory_headroom_gb
    memory_overhead: float = 1.2
    memory_headroom_gb: float = 2.0
    environment_disk_gb: float = 20.0  # python deps + CUDA wheels
    max_cpu_weights_gb: float = 1.0  # smaller models run on CPU flavors
    base_snippet_timeout_s: int = 300
    seconds_per_weight_gb: int = 20  # download + load
    job_overhead_s: int = 600  # image pull + dependency install
    max_job_minutes: int = 60


@dataclass
class AgentConfig:
    # False: proposed PRs are posted to Slack for approval first; True: opened directly
    auto_open_prs: bool = False
    num_trending_models_per_task: int = 5
    max_snippet_runs: int = 3  # original run + 2 repair attempts
    max_cost_per_run_usd: float = 10.0
    tasks: Optional[List[str]] = None  # None: every Hub task
    llm_model: str = "moonshotai/Kimi-K3.1-Pro"
    llm_max_tokens: int = 24000  # Kimi reasons at length on long READMEs
    llm_max_call_usd: float = 0.20  # reserved per call, settled with the reported cost
    llm_max_readme_chars: int = 60000
