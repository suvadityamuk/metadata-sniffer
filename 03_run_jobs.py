"""
Runs the snippets collected by `02_process_code.py` on HF Jobs.
Models are grouped by the cheapest flavor that fits them and each group runs in
a single job (split when it would exceed `max_job_minutes`). Jobs are only
launched while the per-run cost cap allows it. Writes `execution/results.json`,
which `08_report.py` reports.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import ExecuteCodeConfig
from jobs import Budget, run_snippets
from state import load_agent_config
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = Path("execution/snippets.json")
RESULTS_PATH = Path("execution/results.json")


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    execute_config = ExecuteCodeConfig()
    budget = Budget.for_run(load_agent_config(huggingface_api).max_cost_per_run_usd)

    collected = json.loads(SNIPPETS_PATH.read_text())
    report = {
        m["id"]: {"task": m["task"], "status": "skipped", "reason": m["skip_reason"]}
        for m in collected
        if m["skip_reason"]
    }
    runnable = [m for m in collected if not m["skip_reason"]]
    for model_id, entry in run_snippets(
        huggingface_api, runnable, budget, execute_config
    ).items():
        report[model_id] = {
            "task": next(m["task"] for m in runnable if m["id"] == model_id),
            **entry,
        }

    RESULTS_PATH.write_text(json.dumps(report, indent=2))
