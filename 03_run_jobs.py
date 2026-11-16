"""
Runs the snippets collected by `02_process_code.py` on HF Jobs.
Models are grouped by the cheapest flavor that fits them and each group runs in
a single job (split when it would exceed `max_job_minutes`). Jobs are only
launched while the per-run cost cap allows it. Writes `execution/results.json`.
"""

import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig, ExecuteCodeConfig
from jobs import Budget, run_snippets
from utilities import (
    SlackMessage,
    SlackMessageType,
    send_webhook_message,
    setup_logging,
)

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = Path("execution/snippets.json")
RESULTS_PATH = Path("execution/results.json")


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    execute_config = ExecuteCodeConfig()
    budget = Budget(cap_usd=AgentConfig().max_cost_per_run_usd)

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

    counts = Counter(entry["status"] for entry in report.values())
    failed = [
        f"• <https://huggingface.co/{mid}|{mid}>: {e['problems'][0]}"
        for mid, e in report.items()
        if e["status"] == "failed"
    ]
    send_webhook_message(
        os.environ["EXP_SLACK_WEBHOOK_URL"],
        [
            SlackMessage(text="Snippet checks", msg_type=SlackMessageType.HEADER),
            SlackMessage(
                text=f"{dict(counts)} · reserved ${budget.reserved_usd:.2f} of ${budget.cap_usd:.2f}"
            ),
            *[
                SlackMessage(text="\n".join(failed[i : i + 20]))
                for i in range(0, len(failed), 20)
            ],
        ],
    )
