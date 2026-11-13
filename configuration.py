from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    min_cpus_large_cpu_models: int = (
        8  # CPU-only (llama.cpp) models above max_cpu_weights_gb
    )
    base_snippet_timeout_s: int = 300
    seconds_per_weight_gb: int = 20  # download + load
    job_overhead_s: int = 600  # image pull + dependency install
    max_job_minutes: int = 60
    max_queue_minutes: int = 20  # waiting for hardware, on top of the job timeout


@dataclass
class AgentConfig:
    # False: proposed PRs are posted to Slack for approval first; True: opened directly
    auto_open_prs: bool = False
    num_trending_models_per_task: int = 5
    max_snippet_runs: int = 3  # original run + 2 repair attempts
    max_cost_per_run_usd: float = 10.0
    tasks: Optional[List[str]] = None  # None: every Hub task
    recheck_after_days: int = 7  # unchanged models checked more recently are skipped
    llm_model: str = "moonshotai/Kimi-K3.1-Pro"
    llm_max_tokens: int = 24000  # Kimi reasons at length on long READMEs
    llm_max_call_usd: float = 0.20  # reserved per call, settled with the reported cost
    llm_max_readme_chars: int = 60000
    # operators (HF usernames) who can approve PRs, change settings and start runs in the Space
    operators: List[str] = field(default_factory=lambda: ["suvadityamuk", "ariG23498"])
    state_repo: str = (
        "suvadityamuk/metadata-agent-state"  # private dataset, see state.py
    )
    approval_space: str = "suvadityamuk/metadata-agent"
    code_repo: str = "https://github.com/suvadityamuk/metadata-sniffer"
    code_revision: str = "main"
    schedule: str = (
        "0 7 * * 1-5"  # weekdays 07:00 UTC, like the trending-models workflow
    )
    run_timeout_h: int = 6
    radar_enabled: bool = False  # on hold: Radar's channel is not used yet
    radar_days_ahead: int = 2
    radar_days_back: int = 7  # released recently, support pending
    # Radar org name -> Hub accounts, where the names differ
    radar_org_authors: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Zhipu": ["zai-org"],
            "Kimi": ["moonshotai"],
            "Ant Group": ["inclusionAI"],
            "Meta": ["facebook", "meta-llama"],
            "Shanghai AI Lab": ["internlm", "OpenGVLab"],
            "Meituan": ["meituan-longcat"],
            "IBM": ["ibm-granite"],
            "SparkLLM": ["iflytek"],
        }
    )
