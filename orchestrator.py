"""
Runs the whole agent as an HF Job: snippet checks and repairs, metadata
fills, PR tracking and the Slack report, in that order. The orchestrator job
holds the secrets but runs no model code; model snippets run in separate,
secret-free jobs (see `jobs.py`).

Every run has an ID (`sweep-…` for all trending models, `run-…` for a single
model) that tags its proposals and its stats in the agent state (`runs/<id>.json`).

    python orchestrator.py sweep             # sweep now
    python orchestrator.py run <model-id>    # check one model now
    python orchestrator.py schedule          # scheduled sweeps (AgentConfig.schedule)
"""

import os
import secrets
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig, ExecuteCodeConfig
from state import write

load_dotenv()

STAGES = [
    "02_process_code.py",
    "03_run_jobs.py",
    "06_repair_snippets.py",
    "05_fill_metadata.py",
    "07_track_prs.py",
    "08_report.py",
]


def new_run_id(kind: str) -> str:
    return f"{kind}-{datetime.now(timezone.utc):%Y%m%d-%H%M}-{secrets.token_hex(2)}"


def _job_kwargs(config: AgentConfig, env: dict) -> dict:
    # a failing stage must not stop PR tracking and the report, so stages run with `;`
    stages = "; ".join(f"python {stage}" for stage in STAGES)
    # scheduled runs get their ID when they start
    run_id = 'export RUN_ID="${RUN_ID:-sweep-$(date -u +%Y%m%d-%H%M)-$(printf %04x $RANDOM)}"'
    return dict(
        image="python:3.12",
        command=[
            "/bin/bash",
            "-c",
            f"{run_id} && git clone --depth 1 -b {config.code_revision} {config.code_repo} agent && cd agent && "
            f"pip install -q -r requirements.txt && {stages}",
        ],
        env=env,
        secrets={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "EXP_SLACK_WEBHOOK_URL": os.environ["EXP_SLACK_WEBHOOK_URL"],
        },
        flavor="cpu-basic",
        timeout=config.run_timeout_h * 3600,
        namespace=ExecuteCodeConfig().jobs_namespace,
    )


def launch(
    huggingface_api: HfApi, config: AgentConfig, model_id: str | None = None
) -> tuple[str, str]:
    """Starts a sweep (or a single-model run) now; returns its run ID and job URL."""
    run_id = new_run_id("run" if model_id else "sweep")
    env = {"RUN_ID": run_id, **({"MODEL_ID": model_id} if model_id else {})}
    job = huggingface_api.run_job(**_job_kwargs(config, env))
    record = {
        "id": run_id,
        "model_id": model_id,
        "job_url": job.url,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write(huggingface_api, f"runs/{run_id}.json", record, f"Start {run_id}")
    return run_id, job.url


def schedule_sweeps(huggingface_api: HfApi, config: AgentConfig):
    return huggingface_api.create_scheduled_job(
        schedule=config.schedule, **_job_kwargs(config, {})
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    config = AgentConfig()
    action = sys.argv[1]
    if action == "schedule":
        print(schedule_sweeps(huggingface_api, config))
    else:
        print(launch(huggingface_api, config, sys.argv[2] if action == "run" else None))
