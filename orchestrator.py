"""
Runs the whole agent as an HF Job: snippet checks and repairs, metadata
fills, PR tracking and the Slack report, in that order. The orchestrator job
holds the secrets but runs no model code; model snippets run in separate,
secret-free jobs (see `jobs.py`).

Every run has an ID (`sweep-…` for all trending models, `run-…` for a single
model) that tags its proposals and its stats in the agent state (`runs/<id>.json`).

    python orchestrator.py sweep [n]         # sweep the top n now
    python orchestrator.py run <model-id>    # check one model now
    python orchestrator.py revise            # address review comments from the approval Space
    python orchestrator.py schedule          # scheduled top-10 sweeps, created suspended (AgentConfig.schedule)
"""

import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig, ExecuteCodeConfig
from state import read, write

load_dotenv()

# answers review comments left in the approval Space
REVISE_STAGES = ["06_repair_snippets.py --revise", "05_fill_metadata.py --revise"]
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


def _job_kwargs(config: AgentConfig, env: dict, stages: list[str] = STAGES) -> dict:
    # a failing stage must not stop PR tracking and the report, so stages run with `;`
    stages = "; ".join(f"python {stage}" for stage in stages)
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


def running_sweep(huggingface_api: HfApi, config: AgentConfig) -> str | None:
    """ID of a sweep started from Slack or the Space that has not finished yet (within the run timeout)."""
    files = huggingface_api.list_repo_files(config.state_repo, repo_type="dataset")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.run_timeout_h)
    for path in sorted((f for f in files if f.startswith("runs/sweep-")), reverse=True)[
        :5
    ]:
        record = read(huggingface_api, path, {})
        if (
            record.get("status") == "running"
            and datetime.fromisoformat(record["started_at"]) > cutoff
        ):
            return record["id"]
    return None


def launch(
    huggingface_api: HfApi,
    config: AgentConfig,
    model_id: str | None = None,
    top_n: int | None = None,
    tasks: list[str] | None = None,
    per_task: int | None = None,
    force: bool = False,
    reason: str | None = None,
    started_by: str | None = None,
) -> tuple[str, str]:
    """
    Starts a run now: a single model (`model_id`), a task sweep (`tasks`) or a sweep of the top `top_n`.
    `force` re-checks cached models (after a breaking change on the Hub); `reason` says why.
    Returns its run ID and job URL.
    """
    run_id = new_run_id("run" if model_id else "sweep")
    params = {
        "MODEL_ID": model_id,
        "TOP_N": top_n,
        "TASKS": ",".join(tasks) if tasks else None,
        "PER_TASK": per_task,
        "FORCE": "1" if force else None,
    }
    env = {"RUN_ID": run_id, **{k: str(v) for k, v in params.items() if v is not None}}
    job = huggingface_api.run_job(**_job_kwargs(config, env))
    record = {
        "id": run_id,
        "job_url": job.url,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_by": started_by,
        "params": {k.lower(): v for k, v in params.items() if v is not None},
        "reason": reason,
    }
    write(huggingface_api, f"runs/{run_id}.json", record, f"Start {run_id}")
    return run_id, job.url


def launch_revision(huggingface_api: HfApi, config: AgentConfig) -> str:
    """Starts a job that addresses all pending review comments; returns its URL."""
    return huggingface_api.run_job(
        **_job_kwargs(config, {"RUN_ID": new_run_id("revise")}, REVISE_STAGES)
    ).url


def schedule_sweeps(huggingface_api: HfApi, config: AgentConfig):
    """
    The scheduled sweep: the top `scheduled_top_n` of the main trending list, cached models skipped, never forced.
    Created suspended: it only starts running once someone resumes it on purpose.
    """
    return huggingface_api.create_scheduled_job(
        schedule=config.schedule,
        suspend=True,
        concurrency=False,  # never two scheduled sweeps at once
        **_job_kwargs(config, {"TOP_N": str(config.scheduled_top_n)}),
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    config = AgentConfig()
    action = sys.argv[1]
    if action == "schedule":
        print(schedule_sweeps(huggingface_api, config))
    elif action == "revise":
        print(launch_revision(huggingface_api, config))
    elif action == "run":
        print(launch(huggingface_api, config, model_id=sys.argv[2]))
    else:
        print(
            launch(
                huggingface_api,
                config,
                top_n=int(sys.argv[2]) if len(sys.argv) > 2 else None,
            )
        )
