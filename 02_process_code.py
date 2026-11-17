"""
Selects the models of a run and collects their "Use this model" snippets for the
libraries we can run (only when the model is tagged with that library), then
sizes the hardware each model needs, in one pass over all models.

- sweep: top `TOP_N` of the Hub's main trending list,
- task sweep (approval Space only): top `PER_TASK` of each task in `TASKS`,
- run: the single `MODEL_ID`.

Models checked before at the same commit are skipped (cached) unless `FORCE=1`;
hardware sizes are cached per commit too. Models without a runnable snippet are
still listed, for the metadata checks; models with an open PR by the team are
flagged so no stage proposes anything for them.
Writes `execution/snippets.json` (consumed by `03_run_jobs.py`) and `execution/cached.json`.
"""

import fnmatch
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig
from jobs import prepare_snippet
from state import load_agent_config, read, write
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = Path("execution/snippets.json")
CACHED_PATH = Path("execution/cached.json")  # reported by 08_report.py
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
    team_pr: dict | None = (
        None  # an open PR by the team: nothing is proposed for this model
    )


def list_tasks(tasks: list[str]) -> list[str]:
    """`all` means every Hub task."""
    if tasks != ["all"]:
        return tasks
    return list(requests.get("https://huggingface.co/api/tasks", timeout=30).json())


def inspect_discussions(
    huggingface_api: HfApi, model_id: str, team: list[str]
) -> tuple[bool, dict | None]:
    """Whether the repo has a discussion tab, and the team's open PR on it, if any."""
    try:
        discussions = list(huggingface_api.get_repo_discussions(model_id))
    except Exception:
        return False, None
    for d in discussions:
        if d.is_pull_request and d.status == "open" and d.author in team:
            return True, {
                "author": d.author,
                "num": d.num,
                "url": f"https://huggingface.co/{model_id}/discussions/{d.num}",
            }
    return True, None


def select_models(
    huggingface_api, limit, checked, force, team, task=None
) -> tuple[list, list[str]]:
    """
    Top `limit` trending (overall, or within `task`), non-gated models with a discussion tab.
    Returns (models to check as (model, team_pr), cached model ids). Cached models, checked before at
    the same commit, keep their place in the top `limit` but are not checked again unless `force`.
    """
    selected, cached = [], []
    for model in huggingface_api.list_models(
        pipeline_tag=task,
        sort="trendingScore",
        limit=limit * 4,  # some candidates are filtered out below
        expand=["gated", "tags", "sha", "pipeline_tag"],
    ):
        if model.gated:
            continue
        has_tab, team_pr = inspect_discussions(huggingface_api, model.id, team)
        if not has_tab:
            logger.info(f"Skipped {model.id}: no discussion tab")
            continue
        if not force and checked.get(model.id, {}).get("sha") == model.sha:
            cached.append(model.id)
        else:
            selected.append((model, team_pr))
        if len(selected) + len(cached) == limit:
            break
    return selected, cached


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
    huggingface_api, model_id, task, tags, sha, hardware: dict, team_pr=None
) -> ModelSnippets:
    result = ModelSnippets(id=model_id, task=task, sha=sha, team_pr=team_pr)
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
        cached = hardware.get(model_id)
        if cached and cached["sha"] == sha:
            result.weights_gb, result.needs_fp8 = (
                cached["weights_gb"],
                cached["needs_fp8"],
            )
        else:
            result.weights_gb, result.needs_fp8 = estimate_weights(
                huggingface_api, model_id, result.snippets
            )
            hardware[model_id] = {
                "sha": sha,
                "weights_gb": result.weights_gb,
                "needs_fp8": result.needs_fp8,
            }
    except Exception as e:
        result.skip_reason = f"could not collect: {e}"
    return result


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    agent_config = load_agent_config(huggingface_api)
    checked = read(huggingface_api, "checked.json", {})
    hardware = read(huggingface_api, "hardware.json", {})
    team = AgentConfig().team_pr_authors
    force = (
        os.environ.get("FORCE") == "1"
    )  # manual sweeps after a breaking change on the Hub; never the schedule

    to_collect, cached = [], []  # (model id, task, tags, sha, team PR)
    if model_id := os.environ.get(
        "MODEL_ID"
    ):  # `/metadata-agent run <model-id>` always checks
        info = huggingface_api.model_info(model_id)
        has_tab, team_pr = inspect_discussions(huggingface_api, model_id, team)
        to_collect.append(
            (
                model_id,
                info.pipeline_tag or "unknown",
                info.tags or [],
                info.sha,
                team_pr,
            )
        )
        gated_or_closed = info.gated or not has_tab
    else:
        gated_or_closed = False
        if tasks := [
            t for t in os.environ.get("TASKS", "").split(",") if t
        ]:  # task sweep, from the approval Space
            per_task = int(
                os.environ.get("PER_TASK") or agent_config.num_trending_models_per_task
            )
            selections = [
                (
                    task,
                    *select_models(
                        huggingface_api, per_task, checked, force, team, task
                    ),
                )
                for task in list_tasks(tasks)
            ]
        else:
            top_n = int(os.environ.get("TOP_N") or agent_config.top_n)
            selections = [
                (None, *select_models(huggingface_api, top_n, checked, force, team))
            ]
        for task, selected, task_cached in selections:
            cached += task_cached
            to_collect += [
                (
                    m.id,
                    task or m.pipeline_tag or "unknown",
                    m.tags or [],
                    m.sha,
                    team_pr,
                )
                for m, team_pr in selected
            ]

    # one pass sizes every model (in parallel, cached per commit) before any job is planned
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda args: collect(huggingface_api, *args[:4], hardware, args[4]),
                to_collect,
            )
        )
    if gated_or_closed:
        results[0].skip_reason = "gated or no discussion tab"
    write(huggingface_api, "hardware.json", hardware, "Cache hardware sizes")

    SNIPPETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNIPPETS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2))
    CACHED_PATH.write_text(json.dumps(cached))
    logger.info(
        f"Collected {len(results)} models, {len(cached)} cached (checked before, unchanged)"
    )
