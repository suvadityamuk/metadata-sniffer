"""
Repairs snippets that failed in `03_run_jobs.py`. The LLM diagnoses each failure
and proposes a fixed snippet that does everything the original does; fixes are
re-run on HF Jobs, `max_snippet_runs` runs in total including the original.
A fix that runs is turned into:
- a PR proposal, when the cause is the model's metadata (the Hub generates the
  snippet from it, e.g. a wrong pipeline_tag gives a wrong pipeline snippet), or
- a huggingface.js alert in Slack, when the snippet template itself is wrong.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import ExecuteCodeConfig
from jobs import Budget, prepare_snippet, run_snippets
from llm import ask_json
from proposals import Proposal, load_vocabulary, submit
from state import load_agent_config
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
REPAIRS_PATH = Path("execution/repairs.json")
FIXABLE_METADATA = ("pipeline_tag", "library_name")

SYSTEM_PROMPT = """You fix Hugging Face "Use this model" code snippets that fail to run.
The Hub generates these snippets from the model's metadata (pipeline_tag, library_name) and config
using the templates in huggingface.js, so a failure is caused by one of:
- "metadata": the model's pipeline_tag or library_name is wrong, so the Hub generates the wrong snippet,
- "snippet_template": the huggingface.js template is wrong for this kind of model,
- "model_repo": the model repository itself is broken (missing files, broken custom code).
Return JSON: {"cause": "...", "metadata_fix": {"pipeline_tag": "..."} or null, "fixed_snippet": "..." or null, "explanation": "..."}
Rules for fixed_snippet: it must do everything the original does (same model id, same inputs, same outputs),
end with the expression that produces the output, use only pip-installable packages and public URLs from the
original, and never use tokens. When the cause is "metadata", write the snippet the Hub would generate after
metadata_fix is applied. Set fixed_snippet to null when you cannot fix it."""


def diagnose(huggingface_api, model, failure, attempts, budget, config) -> dict | None:
    info = huggingface_api.model_info(model["id"])
    context = {
        "model_id": model["id"],
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "tags": info.tags,
        "transformers_info": info.transformers_info.__dict__
        if info.transformers_info
        else None,
        "architectures": (info.config or {}).get("architectures"),
        "original_snippet": model["snippets"][failure["index"]],
        "error": failure["error"],
        "previous_attempts": attempts,
    }
    return ask_json(
        SYSTEM_PROMPT, json.dumps(context, indent=2, default=str), budget, config
    )


def repair(
    huggingface_api, failed_models, failures, budget, agent_config, execute_config
):
    """Diagnose → re-run, until each failure is fixed or the run limit is reached."""
    attempts = {
        key: [] for key in failures
    }  # (model_id, index) -> [{"answer", "error"}]
    fixed = {}
    for _ in range(agent_config.max_snippet_runs - 1):
        open_failures = [key for key in failures if key not in fixed]
        candidates = {}
        for model_id, index in open_failures:
            last_error = (
                attempts[(model_id, index)][-1]["error"]
                if attempts[(model_id, index)]
                else failures[(model_id, index)]["error"]
            )
            answer = diagnose(
                huggingface_api,
                failed_models[model_id],
                {"index": index, "error": last_error},
                attempts[(model_id, index)],
                budget,
                agent_config,
            )
            if answer and answer.get("fixed_snippet"):
                candidates[(model_id, index)] = answer
        if not candidates:
            break
        # every candidate is a one-snippet "model" so it batches like the original runs
        to_run = [
            {
                **failed_models[m],
                "id": f"{m}#{i}",
                "snippets": [prepare_snippet(a["fixed_snippet"])],
            }
            for (m, i), a in candidates.items()
        ]
        for run_id, entry in run_snippets(
            huggingface_api, to_run, budget, execute_config
        ).items():
            model_id, index = run_id.rsplit("#", 1)
            key = (model_id, int(index))
            if entry["status"] == "ok":
                fixed[key] = {**candidates[key], "job_id": entry["job_id"]}
            else:
                result = (entry.get("results") or [{}])[0]
                attempts[key].append(
                    {
                        "answer": candidates[key],
                        "error": result.get("error") or entry.get("reason"),
                    }
                )
    return fixed


def to_proposal(model_id, fix, vocabulary) -> Proposal | None:
    metadata = {
        k: v
        for k, v in (fix.get("metadata_fix") or {}).items()
        if k in FIXABLE_METADATA and v in vocabulary[k]
    }
    if fix["cause"] != "metadata" or not metadata:
        return None
    changes = ", ".join(f"`{k}` → `{v}`" for k, v in metadata.items())
    return Proposal(
        model_id=model_id,
        kind="snippet",
        metadata=metadata,
        overwrite=True,
        title=f"Fix the 'Use this model' snippet: set {', '.join(metadata)}",
        description=f"The code snippet the Hub shows under *Use this model* currently fails. {fix['explanation']}\n\n"
        f"This PR sets {changes}, which makes the Hub generate the snippet below. "
        f"We ran it on [HF Jobs](https://huggingface.co/jobs/{ExecuteCodeConfig().jobs_namespace}/{fix['job_id']}) and it works:\n\n"
        f"```python\n{fix['fixed_snippet']}\n```\n\nOpened by the Hugging Face metadata agent, feedback welcome!",
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    agent_config, execute_config = (
        load_agent_config(huggingface_api),
        ExecuteCodeConfig(),
    )
    budget = Budget.for_run(agent_config.max_cost_per_run_usd)

    collected = {m["id"]: m for m in json.loads(SNIPPETS_PATH.read_text())}
    report = json.loads(RESULTS_PATH.read_text())
    failures = {
        (model_id, r["index"]): r
        for model_id, entry in report.items()
        if entry["status"] == "failed"
        for r in entry["results"]
        if r["exit_code"] not in (0, None)
    }
    vocabulary = load_vocabulary()
    fixed = repair(
        huggingface_api, collected, failures, budget, agent_config, execute_config
    )
    REPAIRS_PATH.write_text(
        json.dumps({f"{m}#{i}": fix for (m, i), fix in fixed.items()}, indent=2)
    )

    new_proposals, template_alerts = [], []
    for (model_id, index), fix in fixed.items():
        if proposal := to_proposal(model_id, fix, vocabulary):
            new_proposals.append(proposal)
        elif fix["cause"] == "snippet_template":
            template_alerts.append(
                f"• <https://huggingface.co/{model_id}|{model_id}> snippet {index}: {fix['explanation']}\n```{fix['fixed_snippet']}```"
            )
    submit(huggingface_api, new_proposals, agent_config.auto_open_prs)

    if template_alerts:
        send_webhook_message(
            os.environ["EXP_SLACK_WEBHOOK_URL"],
            [
                SlackMessage(
                    text="huggingface.js snippet fixes needed",
                    msg_type=SlackMessageType.HEADER,
                ),
                *[SlackMessage(text=alert[:2900]) for alert in template_alerts],
            ],
        )
