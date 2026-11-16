"""
Agent state in a private Hub dataset, shared by the scheduled jobs and the
approval Space:
- settings.json: toggles that override `AgentConfig` defaults,
- proposals.json: proposed PRs and their approval status,
- prs.json: PRs the agent opened, tracked until they are merged or closed.
"""

import json
from dataclasses import fields

from huggingface_hub import HfApi

from configuration import AgentConfig

# toggles that can be changed from the Space; everything else stays in code
EDITABLE_SETTINGS = (
    "auto_open_prs",
    "num_trending_models_per_task",
    "max_cost_per_run_usd",
    "tasks",
)


def read(huggingface_api: HfApi, name: str, default):
    if not huggingface_api.file_exists(
        AgentConfig.state_repo, name, repo_type="dataset"
    ):
        return default
    path = huggingface_api.hf_hub_download(
        AgentConfig.state_repo, name, repo_type="dataset", force_download=True
    )
    with open(path) as f:
        return json.load(f)


def write(huggingface_api: HfApi, name: str, data, message: str):
    huggingface_api.create_repo(
        AgentConfig.state_repo, repo_type="dataset", private=True, exist_ok=True
    )
    huggingface_api.upload_file(
        path_or_fileobj=json.dumps(data, indent=2).encode(),
        path_in_repo=name,
        repo_id=AgentConfig.state_repo,
        repo_type="dataset",
        commit_message=message,
    )


def load_agent_config(huggingface_api: HfApi) -> AgentConfig:
    settings = read(huggingface_api, "settings.json", {})
    known = {f.name for f in fields(AgentConfig)}
    return AgentConfig(
        **{k: v for k, v in settings.items() if k in EDITABLE_SETTINGS and k in known}
    )
