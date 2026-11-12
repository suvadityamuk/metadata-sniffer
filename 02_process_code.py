"""
Selects the top N trending models per task and collects their "Use this model"
snippets for the libraries we can run (only when the model is tagged with that
library), then estimates the hardware each model needs. Models without a
runnable snippet are still listed, for the metadata checks.
Writes `execution/snippets.json`, which is consumed by `03_run_jobs.py`.
"""

import fnmatch
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

import radar
from configuration import AgentConfig
from jobs import prepare_snippet
from state import load_agent_config, read
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = Path("execution/snippets.json")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
DIFFUSERS_VARIANT = re.compile(r"\.(fp16|bf16|fp32|non_ema|ema)\.\w+$")
# model tag -> module its snippets import; snippets are only run when the model has the tag
RUNNABLE_LIBRARIES = {
    "transformers": "transformers",
    "diffusers": "diffusers",
    "sentence-transformers": "sentence_transformers",
    "gguf": "llama_cpp",
}
# not every job needs these, so they are installed per model inside the job
LIBRARY_DEPENDENCIES = {
    "gguf": ["llama-cpp-python"],
}


@dataclass
class ModelSnippets:
    id: str
    task: str
    sha: str
    snippets: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    weights_gb: float = 0.0
    needs_fp8: bool = False
    cpu_only: bool = False  # llama.cpp snippets run on CPU
    skip_reason: str | None = None


def list_tasks(config: AgentConfig) -> list[str]:
    if config.tasks:
        return config.tasks
    return list(requests.get("https://huggingface.co/api/tasks", timeout=30).json())


def has_discussions(huggingface_api: HfApi, model_id: str) -> bool:
    try:
        next(iter(huggingface_api.get_repo_discussions(model_id)), None)
        return True
    except Exception:
        return False


def recently_checked(checked: dict, model_id: str, sha: str, days: int) -> bool:
    """Checked within `days` days and unchanged since (same commit)."""
    entry = checked.get(model_id)
    if not entry or entry["sha"] != sha:
        return False
    return datetime.now(timezone.utc) - datetime.fromisoformat(
        entry["date"]
    ) < timedelta(days=days)


def select_models(
    huggingface_api: HfApi, task: str, limit: int, checked: dict, recheck_days: int
) -> list:
    """
    Top `limit` trending, non-gated models with a discussion tab, as (id, tags, sha).
    Recently checked, unchanged models keep their place in the top `limit` but are not checked again.
    """
    selected, taken = [], 0
    for model in huggingface_api.list_models(
        pipeline_tag=task,
        sort="trendingScore",
        limit=limit * 4,  # some candidates are filtered out below
        expand=["gated", "tags", "sha"],
    ):
        if model.gated:
            continue
        if not has_discussions(huggingface_api, model.id):
            logger.info(f"Skipped {model.id}: no discussion tab")
            continue
        taken += 1
        if recently_checked(checked, model.id, model.sha, recheck_days):
            logger.info(f"Skipped {model.id}: checked recently, unchanged")
        else:
            selected.append((model.id, model.tags or [], model.sha))
        if taken == limit:
            break
    return selected


def snippet_library(code: str) -> str | None:
    for tag, module in RUNNABLE_LIBRARIES.items():
        if re.search(rf"^\s*(from|import) {module}\b", code, re.MULTILINE):
            return tag
    return None


def fetch_snippets(model_id: str, tags: list[str]) -> dict[str, list[str]]:
    """
    Runnable snippets from the model page's auto-generated notebook (the same
    snippets as "Use this model"), keyed by library. A cell without imports
    continues the previous snippet (e.g. llama.cpp's load cell, then its call cell).
    """
    response = requests.get(f"https://huggingface.co/{model_id}.ipynb", timeout=30)
    response.raise_for_status()
    snippets, current = {}, None
    for cell in response.json().get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or source.lstrip().startswith("!"):
            continue
        library = snippet_library(source)
        if library:
            current = [source] if library in tags else None
            if current:
                snippets.setdefault(library, []).append(current)
        elif current and "import" not in source:
            current.append(source)
        else:
            current = None  # e.g. Inference Providers cells
    return {
        lib: [prepare_snippet("\n".join(parts)) for parts in group]
        for lib, group in snippets.items()
    }


def estimate_weights(
    huggingface_api: HfApi, model_id: str, snippets: list[str]
) -> tuple[float, bool]:
    """
    Weight size from file sizes (dtype agnostic) and whether the weights are FP8.
    For GGUF repos only the files the snippets reference are counted.
    """
    info = huggingface_api.model_info(model_id, files_metadata=True)
    gguf_files = re.findall(r"filename=['\"]([^'\"]+)['\"]", "\n".join(snippets))
    if gguf_files:
        files = [
            s
            for s in info.siblings
            if any(fnmatch.fnmatch(s.rfilename, g) for g in gguf_files)
        ]
    else:
        files = [s for s in info.siblings if s.rfilename.endswith(WEIGHT_SUFFIXES)]
        if any(f.rfilename.endswith(".safetensors") for f in files):
            files = [f for f in files if f.rfilename.endswith(".safetensors")]
        if any(s.rfilename == "model_index.json" for s in info.siblings):
            # diffusers loads one copy of each component folder, not variants or single-file checkpoints
            files = [
                f
                for f in files
                if "/" in f.rfilename and not DIFFUSERS_VARIANT.search(f.rfilename)
            ]
    weights_gb = sum(f.size or 0 for f in files) / 1024**3
    dtypes = info.safetensors.parameters if info.safetensors else {}
    return round(weights_gb, 2), any(d.startswith("F8") for d in dtypes)


def collect(
    huggingface_api: HfApi, model_id: str, task: str, tags: list[str], sha: str
) -> ModelSnippets:
    result = ModelSnippets(id=model_id, task=task, sha=sha)
    try:
        by_library = fetch_snippets(model_id, tags)
        result.snippets = [code for group in by_library.values() for code in group]
        if not result.snippets:
            result.skip_reason = "no runnable snippet for the model's libraries"
            return result
        result.dependencies = [
            d for lib in by_library for d in LIBRARY_DEPENDENCIES.get(lib, [])
        ]
        result.cpu_only = list(by_library) == ["gguf"]
        result.weights_gb, result.needs_fp8 = estimate_weights(
            huggingface_api, model_id, result.snippets
        )
    except Exception as e:
        result.skip_reason = f"could not collect: {e}"
    return result


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    agent_config = load_agent_config(huggingface_api)
    checked = read(huggingface_api, "checked.json", {})

    results = []
    for task in list_tasks(agent_config):
        for model_id, tags, sha in select_models(
            huggingface_api,
            task,
            agent_config.num_trending_models_per_task,
            checked,
            agent_config.recheck_after_days,
        ):
            results.append(asdict(collect(huggingface_api, model_id, task, tags, sha)))
            logger.info(f"Collected {model_id} ({task})")

    # new repos of releases on Radar's calendar
    calendar = read(huggingface_api, "radar.json", {}).get("calendar", "")
    releases = radar.releases_to_watch(radar.parse_calendar(calendar), agent_config)
    seen = {r["id"] for r in results}
    for model_id in radar.new_models(huggingface_api, releases, agent_config):
        info = huggingface_api.model_info(model_id)
        if (
            model_id in seen
            or info.gated
            or recently_checked(
                checked, model_id, info.sha, agent_config.recheck_after_days
            )
            or not has_discussions(huggingface_api, model_id)
        ):
            continue
        results.append(
            asdict(
                collect(huggingface_api, model_id, "radar", info.tags or [], info.sha)
            )
        )
        logger.info(f"Collected {model_id} (radar)")

    SNIPPETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNIPPETS_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {len(results)} models to {SNIPPETS_PATH}")
