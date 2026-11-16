"""
Proposed Hub PRs. Stages only append proposals; `open_pr` is the single place
that writes to a model repo, and is called directly in auto mode or after
approval in manual mode.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, metadata_update

PROPOSALS_PATH = Path("execution/proposals.json")


@dataclass
class Proposal:
    model_id: str
    kind: str  # "metadata" (fill missing fields) or "snippet" (metadata fix that repairs the snippet)
    metadata: dict
    title: str
    description: str
    overwrite: bool = False  # only snippet fixes may change existing values


def load_proposals() -> list[Proposal]:
    if not PROPOSALS_PATH.exists():
        return []
    return [Proposal(**p) for p in json.loads(PROPOSALS_PATH.read_text())]


def add_proposals(new: list[Proposal]):
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    proposals = load_proposals() + new
    PROPOSALS_PATH.write_text(json.dumps([asdict(p) for p in proposals], indent=2))


def load_vocabulary() -> dict[str, set[str]]:
    """Valid Hub values, so proposals can be checked instead of trusted."""
    tags = requests.get(
        "https://huggingface.co/api/models-tags-by-type", timeout=30
    ).json()
    metrics = requests.get("https://huggingface.co/api/metrics", timeout=30).json()
    return {
        "license": {t["label"] for t in tags["license"]},
        "language": {t["id"] for t in tags["language"]},
        "library_name": {t["id"] for t in tags["library"]},
        "pipeline_tag": {t["id"] for t in tags["pipeline_tag"]},
        "metrics": {m["id"] for m in metrics},
    }


def open_pr(huggingface_api: HfApi, proposal: Proposal) -> str:
    """Opens the PR on the model repo and returns its URL."""
    url = metadata_update(
        proposal.model_id,
        proposal.metadata,
        overwrite=proposal.overwrite,
        create_pr=True,
        commit_message=proposal.title,
        commit_description=proposal.description,
        token=huggingface_api.token,
    )
    return url
