"""
Runs the whole agent as an HF Job: snippet checks and repairs, metadata
fills, PR tracking and the Slack report, in that order. The orchestrator job
holds the secrets but runs no model code; model snippets run in separate,
secret-free jobs (see `jobs.py`).

    python orchestrator.py run       # one run now
    python orchestrator.py schedule  # weekday schedule (AgentConfig.schedule)
"""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, JobInfo

from configuration import AgentConfig, ExecuteCodeConfig

load_dotenv()

STAGES = [
    "02_process_code.py",
    "03_run_jobs.py",
    "06_repair_snippets.py",
    "05_fill_metadata.py",
    "07_track_prs.py",
    "08_report.py",
]


def _job_kwargs(config: AgentConfig) -> dict:
    # a failing stage must not stop PR tracking and the report, so stages run with `;`
    stages = "; ".join(f"python {stage}" for stage in STAGES)
    return dict(
        image="python:3.12",
        command=[
            "/bin/bash",
            "-c",
            f"git clone --depth 1 -b {config.code_revision} {config.code_repo} agent && cd agent && "
            f"pip install -q -r requirements.txt && {stages}",
        ],
        secrets={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "EXP_SLACK_WEBHOOK_URL": os.environ["EXP_SLACK_WEBHOOK_URL"],
        },
        flavor="cpu-basic",
        timeout=config.run_timeout_h * 3600,
        namespace=ExecuteCodeConfig().jobs_namespace,
    )


def launch_run(huggingface_api: HfApi, config: AgentConfig) -> JobInfo:
    return huggingface_api.run_job(**_job_kwargs(config))


def schedule_runs(huggingface_api: HfApi, config: AgentConfig):
    return huggingface_api.create_scheduled_job(
        schedule=config.schedule, **_job_kwargs(config)
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    action = {"run": launch_run, "schedule": schedule_runs}[sys.argv[1]]
    print(action(huggingface_api, AgentConfig()))
