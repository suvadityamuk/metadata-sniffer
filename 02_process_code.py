"""
Collects the "Use this model -> Transformers" snippets of the top N trending
transformers models per task and estimates the hardware each model needs.
Writes `execution/snippets.json`, which is consumed by `03_run_jobs.py`.
"""

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = Path("execution/snippets.json")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


@dataclass
class ModelSnippets:
    id: str
    task: str
    snippets: list[str] = field(default_factory=list)
    weights_gb: float = 0.0
    needs_fp8: bool = False
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


def select_models(huggingface_api: HfApi, task: str, limit: int) -> list[str]:
    """Top `limit` trending, public, non-gated transformers models with a discussion tab."""
    selected = []
    for model in huggingface_api.list_models(
        pipeline_tag=task,
        filter="transformers",
        sort="trendingScore",
        limit=limit * 4,  # some candidates are filtered out below
        expand=["gated", "library_name", "tags"],
    ):
        if model.gated or model.library_name != "transformers":
            continue
        if not has_discussions(huggingface_api, model.id):
            logger.info(f"Skipped {model.id}: no discussion tab")
            continue
        selected.append(model.id)
        if len(selected) == limit:
            break
    return selected


def prepare_snippet(code: str) -> str:
    """Drops install lines and binds the trailing expression to `_result` so the runner can inspect it."""
    lines = [
        line
        for line in code.splitlines()
        if not line.lstrip().startswith(("!", "# pip", "%pip"))
    ]
    tree = ast.parse("\n".join(lines))
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())], value=last.value
        )
        ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def fetch_transformers_snippets(model_id: str) -> list[str]:
    """The model page's auto-generated notebook holds the same snippets as "Use this model"."""
    response = requests.get(f"https://huggingface.co/{model_id}.ipynb", timeout=30)
    response.raise_for_status()
    snippets = []
    for cell in response.json().get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "from transformers" in source:
            snippets.append(prepare_snippet(source))
    return snippets


def estimate_weights(huggingface_api: HfApi, model_id: str) -> tuple[float, bool]:
    """Weight size from file sizes (dtype agnostic) and whether the weights are FP8."""
    info = huggingface_api.model_info(model_id, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith(WEIGHT_SUFFIXES)]
    if any(f.rfilename.endswith(".safetensors") for f in files):
        files = [f for f in files if f.rfilename.endswith(".safetensors")]
    weights_gb = sum(f.size or 0 for f in files) / 1024**3
    dtypes = info.safetensors.parameters if info.safetensors else {}
    return round(weights_gb, 2), any(d.startswith("F8") for d in dtypes)


def collect(huggingface_api: HfApi, model_id: str, task: str) -> ModelSnippets:
    result = ModelSnippets(id=model_id, task=task)
    try:
        result.snippets = fetch_transformers_snippets(model_id)
        result.weights_gb, result.needs_fp8 = estimate_weights(
            huggingface_api, model_id
        )
    except Exception as e:
        result.skip_reason = f"could not collect: {e}"
        return result
    if not result.snippets:
        result.skip_reason = "no transformers snippet"
    return result


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    agent_config = AgentConfig()

    results = []
    for task in list_tasks(agent_config):
        for model_id in select_models(
            huggingface_api, task, agent_config.num_trending_models_per_task
        ):
            results.append(asdict(collect(huggingface_api, model_id, task)))
            logger.info(f"Collected {model_id} ({task})")

    SNIPPETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNIPPETS_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {len(results)} models to {SNIPPETS_PATH}")
