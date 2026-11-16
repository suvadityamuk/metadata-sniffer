"""
Deploys the agent's private Spaces with the secrets they need:

    python space/deploy.py approval   # approval page (space/)
    python space/deploy.py slack      # /metadata-agent Slack app (slack_app/)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SHARED_MODULES = ["configuration.py", "orchestrator.py", "proposals.py", "state.py"]
TARGETS = {
    # space id, sdk, folder uploaded to the Space root, extra files kept at their repo path, secrets
    "approval": (
        AgentConfig.approval_space,
        "gradio",
        "space",
        [],
        ["HF_TOKEN", "EXP_SLACK_WEBHOOK_URL"],
    ),
    "slack": (
        AgentConfig.slack_space,
        "docker",
        "slack_app",
        ["slack_app/app.py"],
        ["HF_TOKEN", "EXP_SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
    ),
}

if __name__ == "__main__":
    space, sdk, folder, extra_files, secrets = TARGETS[sys.argv[1]]
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    huggingface_api.create_repo(
        space, repo_type="space", space_sdk=sdk, private=True, exist_ok=True
    )
    huggingface_api.update_repo_settings(
        space, private=True, repo_type="space"
    )  # stays private on redeploys
    for name in secrets:
        huggingface_api.add_space_secret(space, name, os.environ[name])
    huggingface_api.upload_folder(
        folder_path=ROOT / folder,
        repo_id=space,
        repo_type="space",
        # the Slack app's entry point keeps its package path (see extra files)
        ignore_patterns=["deploy.py", "__pycache__/*", "manifest.json"]
        + (["app.py"] if extra_files else []),
    )
    for path in SHARED_MODULES + extra_files:
        huggingface_api.upload_file(
            path_or_fileobj=ROOT / path,
            path_in_repo=path,
            repo_id=space,
            repo_type="space",
        )
    print(f"https://huggingface.co/spaces/{space}")
