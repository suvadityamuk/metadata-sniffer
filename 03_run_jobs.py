"""
Runs the snippets collected by `02_process_code.py` on HF Jobs.
Models are grouped by the cheapest flavor that fits them and each group runs in
a single job (split when it would exceed `max_job_minutes`). Jobs are only
launched while the per-run cost cap allows it. Writes `execution/results.json`.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig, ExecuteCodeConfig
from jobs import (
    Budget,
    collect_results,
    launch_batch,
    load_flavors,
    pick_flavor,
    snippet_timeout_s,
)
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
EXCEPTION_LINE = re.compile(r"^[\w.]+(Error|Exception|Interrupt)\b")


def output_problem(output: dict | None) -> str | None:
    """Correctness only: a produced output must be non-empty and numerically sane."""
    if output is None:
        return None  # snippet only loads the model
    if output["type"] == "tensor" and not output["finite"]:
        return "output tensor has NaN/inf"
    if output["type"] == "list":
        return "empty output" if output["len"] == 0 else output_problem(output["first"])
    if output["type"] == "dict":
        return next(filter(None, map(output_problem, output["items"].values())), None)
    if output["type"] == "str" and not output["value"].strip():
        return "empty output"
    return None


def plan_batches(models, flavors, budget, config):
    """Groups models into (flavor, models, timeout_s) batches; cheapest batches get the budget first."""
    by_flavor, skipped = {}, {}
    for model in models:
        flavor = pick_flavor(model["weights_gb"], model["needs_fp8"], flavors, config)
        if flavor is None:
            skipped[model["id"]] = "too large for available hardware"
            continue
        model["timeout_s"] = snippet_timeout_s(model["weights_gb"], config)
        by_flavor.setdefault(flavor.name, (flavor, []))[1].append(model)

    candidates = []
    for flavor, flavor_models in by_flavor.values():
        current, seconds = [], config.job_overhead_s
        for model in flavor_models:
            model_s = model["timeout_s"] * len(model["snippets"])
            if current and seconds + model_s > config.max_job_minutes * 60:
                candidates.append((flavor, current, seconds))
                current, seconds = [], config.job_overhead_s
            current.append(model)
            seconds += model_s
        candidates.append((flavor, current, seconds))

    batches = []
    for flavor, batch_models, seconds in sorted(
        candidates, key=lambda c: c[0].usd_per_minute * c[2]
    ):
        if budget.reserve(flavor.usd_per_minute * seconds / 60):
            batches.append((flavor, batch_models, seconds))
        else:
            skipped.update({m["id"]: "cost cap reached" for m in batch_models})
    return batches, skipped


def exception_line(stderr: str | None) -> str:
    lines = (stderr or "unknown error").strip().splitlines()
    return next(
        (line for line in reversed(lines) if EXCEPTION_LINE.match(line)), lines[-1]
    )


def summarise(model: dict, results: list[dict]) -> dict:
    problems = []
    for r in sorted(results, key=lambda r: r["index"]):
        if r["exit_code"] != 0:
            problems.append(f"snippet {r['index']}: {exception_line(r['error'])}")
        elif problem := output_problem(r.get("output")):
            problems.append(f"snippet {r['index']}: {problem}")
    if len(results) < len(model["snippets"]):
        problems.append("job ended before all snippets ran")
    return {
        "status": "failed" if problems else "ok",
        "problems": problems,
        "results": results,
    }


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

    batches, skipped = plan_batches(runnable, load_flavors(), budget, execute_config)
    report.update(
        {
            mid: {"status": "skipped", "reason": reason}
            for mid, reason in skipped.items()
        }
    )

    job_models = {}
    for flavor, models, timeout_s in batches:
        job = launch_batch(huggingface_api, flavor, models, timeout_s, execute_config)
        job_models[job.id] = (flavor, models)
        logger.info(
            f"Launched {job.url} on {flavor.name} for {[m['id'] for m in models]}"
        )

    for job_id, results in collect_results(
        huggingface_api, list(job_models), execute_config
    ).items():
        flavor, models = job_models[job_id]
        for model in models:
            model_results = [r for r in results if r["id"] == model["id"]]
            report[model["id"]] = {
                "task": model["task"],
                "flavor": flavor.name,
                "job_id": job_id,
                **summarise(model, model_results),
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
