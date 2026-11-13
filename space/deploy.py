"""
Deploys the approval Space: `space/` plus the agent modules it imports, and
the secrets it needs to open PRs and start runs.

    python space/deploy.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SHARED_MODULES = ["configuration.py", "orchestrator.py", "proposals.py", "state.py"]

if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    space = AgentConfig.approval_space
    huggingface_api.create_repo(
        space, repo_type="space", space_sdk="gradio", exist_ok=True
    )
    for name in ("HF_TOKEN", "EXP_SLACK_WEBHOOK_URL"):
        huggingface_api.add_space_secret(space, name, os.environ[name])
    huggingface_api.upload_folder(
        folder_path=ROOT / "space",
        repo_id=space,
        repo_type="space",
        ignore_patterns=["deploy.py", "__pycache__/*"],
    )
    for module in SHARED_MODULES:
        huggingface_api.upload_file(
            path_or_fileobj=ROOT / module,
            path_in_repo=module,
            repo_id=space,
            repo_type="space",
        )
    print(f"https://huggingface.co/spaces/{space}")
