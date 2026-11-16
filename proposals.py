"""
Proposed changes to model repos. Stages `submit` proposals: in auto mode they
are opened right away, otherwise they wait in the agent state for approval in
the Space. `open_pr` is the single place that writes to a model repo, and
records every PR it opens for `07_track_prs.py`.
"""

import json
import os
from dataclasses import asdict, dataclass

import requests
from huggingface_hub import HfApi, metadata_update

from configuration import AgentConfig, ModelCheckerConfig
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
    # what backs the proposal, shown in the approval Space: snippet check/fix jobs and snippets
    evidence: dict | None = None


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


def covered_by_open_pr(huggingface_api: HfApi, proposal: Proposal) -> bool:
    """An open PR by the team or the agent already changes one of the proposal's fields."""
    team = {
        *ModelCheckerConfig().avocado_team_members,
        huggingface_api.whoami()["name"],
    }
    for discussion in huggingface_api.get_repo_discussions(proposal.model_id):
        if not (
            discussion.is_pull_request
            and discussion.status == "open"
            and discussion.author in team
        ):
            continue
        diff = (
            huggingface_api.get_discussion_details(
                proposal.model_id, discussion.num
            ).diff
            or ""
        )
        changed = "\n".join(line for line in diff.splitlines() if line[:1] in "+-")
        if any(f"{field}:" in changed for field in proposal.metadata):
            return True
    return False


class DryRunBlocked(Exception):
    pass


def open_pr(huggingface_api: HfApi, proposal: Proposal) -> str:
    """Opens the PR (or pushes the follow-up to an open one) and returns its URL."""
    config = AgentConfig()
    if (
        config.dry_run
        and proposal.model_id.split("/")[0] not in config.dry_run_namespaces
    ):
        raise DryRunBlocked(
            f"dry run: PRs are only allowed on {config.dry_run_namespaces}, not {proposal.model_id}"
        )
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

    commit = metadata_update(
        proposal.model_id,
        proposal.metadata,
        overwrite=proposal.overwrite,
        create_pr=True,
        commit_message=proposal.title,
        commit_description=proposal.description,
        token=huggingface_api.token,
    )
    track_pr(huggingface_api, proposal, commit.pr_num, commit.pr_url)
    return commit.pr_url


def track_pr(huggingface_api: HfApi, proposal: Proposal, num: int, url: str):
    """Records an opened PR for `07_track_prs.py`."""
    prs = read(huggingface_api, "prs.json", [])
    prs.append(
        {
            "model_id": proposal.model_id,
            "num": num,
            "url": url,
            "kind": proposal.kind,
            "title": proposal.title,
            "metadata": proposal.metadata,
            "status": "open",
            "handled_comments": [],
        }
    )
    write(huggingface_api, "prs.json", prs, f"Track {url}")


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
            proposal.pr_num is None and covered_by_open_pr(huggingface_api, proposal)
        ):
            continue
        seen.add(key(entry))
        entry["status"], entry["run_id"] = "pending", os.environ.get("RUN_ID")
        if auto_open:
            try:
                entry["url"], entry["status"] = (
                    open_pr(huggingface_api, proposal),
                    "opened",
                )
            except DryRunBlocked:
                pass  # stays pending
        proposals.append(entry)
    write(huggingface_api, "proposals.json", proposals, f"{len(new)} new proposals")


def decide(huggingface_api: HfApi, index: int, approve: bool, decided_by: str) -> str:
    """Approves (opens the PR) or rejects a pending proposal; returns a short outcome message."""
    proposals = read(huggingface_api, "proposals.json", [])
    proposal = proposals[index]
    if proposal["status"] != "pending":
        return f"Already {proposal['status']}."
    if approve:
        fields = {
            k: v for k, v in proposal.items() if k in Proposal.__dataclass_fields__
        }
        try:
            proposal["url"] = open_pr(huggingface_api, Proposal(**fields))
        except DryRunBlocked as e:
            return str(e)
    proposal["status"] = "opened" if approve else "rejected"
    proposal["decided_by"] = decided_by
    write(
        huggingface_api,
        "proposals.json",
        proposals,
        f"{proposal['status']} by {decided_by}",
    )
    return f"Opened {proposal['url']}" if approve else "Rejected"
