"""
Proposed changes to model repos. Stages `submit` proposals: in auto mode they
are opened right away, otherwise they wait in the agent state for approval in
the Space. `open_pr` is the single place that writes to a model repo, and
records every PR it opens for `07_track_prs.py`.
"""

import json
from dataclasses import asdict, dataclass

import requests
from huggingface_hub import HfApi, metadata_update

from configuration import ModelCheckerConfig
from state import read, write


@dataclass
class Proposal:
    model_id: str
    kind: str  # "metadata", "snippet", or "pr_update" (follow-up on an open PR)
    metadata: dict
    title: str
    description: str
    overwrite: bool = (
        False  # only snippet fixes and PR follow-ups may change existing values
    )
    pr_num: int | None = None  # set for "pr_update"
    reply: str | None = None  # comment posted with a "pr_update"


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


def has_open_team_pr(huggingface_api: HfApi, model_id: str) -> bool:
    team = {
        *ModelCheckerConfig().avocado_team_members,
        huggingface_api.whoami()["name"],
    }
    return any(
        d.is_pull_request and d.status == "open" and d.author in team
        for d in huggingface_api.get_repo_discussions(model_id)
    )


def open_pr(huggingface_api: HfApi, proposal: Proposal) -> str:
    """Opens the PR (or pushes the follow-up to an open one) and returns its URL."""
    if proposal.pr_num is not None:
        if proposal.metadata:
            metadata_update(
                proposal.model_id,
                proposal.metadata,
                overwrite=True,
                revision=f"refs/pr/{proposal.pr_num}",
                commit_message=proposal.title,
                token=huggingface_api.token,
            )
        if proposal.reply:
            huggingface_api.comment_discussion(
                proposal.model_id, proposal.pr_num, proposal.reply
            )
        return (
            f"https://huggingface.co/{proposal.model_id}/discussions/{proposal.pr_num}"
        )

    url = metadata_update(
        proposal.model_id,
        proposal.metadata,
        overwrite=proposal.overwrite,
        create_pr=True,
        commit_message=proposal.title,
        commit_description=proposal.description,
        token=huggingface_api.token,
    )
    prs = read(huggingface_api, "prs.json", [])
    prs.append(
        {
            "model_id": proposal.model_id,
            "num": int(url.rstrip("/").split("/")[-1]),
            "url": url,
            "kind": proposal.kind,
            "title": proposal.title,
            "metadata": proposal.metadata,
            "status": "open",
            "handled_comments": [],
        }
    )
    write(huggingface_api, "prs.json", prs, f"Track {url}")
    return url


def submit(huggingface_api: HfApi, new: list[Proposal], auto_open: bool):
    """Opens (auto mode) or queues the proposals, skipping ones already proposed or covered by an open PR."""
    proposals = read(huggingface_api, "proposals.json", [])

    def key(p: dict) -> tuple:
        return (
            p["model_id"],
            p["kind"],
            p["pr_num"],
            json.dumps(p["metadata"], sort_keys=True),
            p["reply"],
        )

    seen = {key(p) for p in proposals}
    for proposal in new:
        entry = asdict(proposal)
        if key(entry) in seen or (
            proposal.pr_num is None
            and has_open_team_pr(huggingface_api, proposal.model_id)
        ):
            continue
        seen.add(key(entry))
        entry["status"] = "pending"
        if auto_open:
            entry["url"], entry["status"] = open_pr(huggingface_api, proposal), "opened"
        proposals.append(entry)
    write(huggingface_api, "proposals.json", proposals, f"{len(new)} new proposals")
