"""
Posts the run's report to Slack as one message: a table with a row per model
and check (snippet / metadata) with its PR, if any, and the PR's status, plus
any huggingface.js snippet template fixes. Then records the fully checked models.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig
from state import read, write
from utilities import SlackMessage, SlackMessageType, send_webhook_message

load_dotenv()

RESULTS_PATH = Path("execution/results.json")
SNIPPETS_PATH = Path("execution/snippets.json")
METADATA_DONE_PATH = Path("execution/metadata_done.json")
TEMPLATE_ALERTS_PATH = Path("execution/template_alerts.json")
SNIPPET_STATUS = {
    "ok": "snippet works",
    "failed": "snippet broken",
    "inconclusive": "inconclusive (timeout)",
    "skipped": "skipped",
}
PR_STATUS = {
    "merged": "accepted ✅",
    "closed": "rejected ❌",
    "draft": "draft",
    "open": "awaiting merge ⏳",
}
MAX_TABLE_ROWS = 99  # Slack tables hold 100 rows, including the header


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _link(text: str, url: str) -> dict:
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [{"type": "link", "text": text, "url": url}],
            }
        ],
    }


def _text(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def pr_status(pr: dict) -> str:
    if pr["status"] == "open" and pr.get("needs_human"):
        return "comment needs a human 🙋"
    return PR_STATUS.get(pr["status"], pr["status"])


def build_rows(
    results: dict, proposals: list[dict], prs: list[dict], space_url: str
) -> list[list[dict]]:
    rows, week_ago = [], datetime.now(timezone.utc) - timedelta(days=7)
    for pr in prs:
        if pr.get("settled_at") and datetime.fromisoformat(pr["settled_at"]) < week_ago:
            continue  # merged/closed PRs are reported for a week
        rows.append(
            [
                _link(pr["model_id"], f"https://huggingface.co/{pr['model_id']}"),
                _text(pr["kind"]),
                _link(f"#{pr['num']}", pr["url"]),
                _text(pr_status(pr)),
            ]
        )
    for proposal in proposals:
        if proposal["status"] == "pending":
            rows.append(
                [
                    _link(
                        proposal["model_id"],
                        f"https://huggingface.co/{proposal['model_id']}",
                    ),
                    _text(proposal["kind"]),
                    _link("review", space_url),
                    _text("awaiting approval 👀"),
                ]
            )
    for model_id, entry in results.items():
        status = SNIPPET_STATUS.get(entry["status"], entry["status"])
        detail = entry.get("reason") or "; ".join(entry.get("problems", [])[:1])
        rows.append(
            [
                _link(model_id, f"https://huggingface.co/{model_id}"),
                _text("snippet"),
                _text("–"),
                _text(f"{status}: {detail}" if detail else status),
            ]
        )
    return rows


def mark_checked(
    huggingface_api: HfApi,
    results: dict,
    snippets: list[dict],
    metadata_done: list[str],
):
    """Models whose snippet and metadata checks both ran are skipped until they change (see 02)."""
    checked = read(huggingface_api, "checked.json", {})
    now = datetime.now(timezone.utc).isoformat()
    for model in snippets:
        if (
            model["id"] in metadata_done
            and results.get(model["id"], {}).get("reason") != "cost cap reached"
        ):
            checked[model["id"]] = {"sha": model["sha"], "date": now}
    write(huggingface_api, "checked.json", checked, "Record checked models")


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    space_url = f"https://huggingface.co/spaces/{AgentConfig.approval_space}"
    results = _load(RESULTS_PATH, {})
    rows = build_rows(
        results,
        read(huggingface_api, "proposals.json", []),
        read(huggingface_api, "prs.json", []),
        space_url,
    )
    alerts = _load(TEMPLATE_ALERTS_PATH, [])

    # one message: header, table, huggingface.js alerts; only very long tables spill over
    header = [[_text("Model"), _text("Check"), _text("PR"), _text("Status")]]
    tables = [
        header + rows[i : i + MAX_TABLE_ROWS]
        for i in range(0, len(rows), MAX_TABLE_ROWS)
    ] or [header]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    messages = [
        SlackMessage(
            text=f"Metadata agent report {today}", msg_type=SlackMessageType.HEADER
        ),
        SlackMessage(msg_type=SlackMessageType.TABLE, text=tables[0]),
    ]
    if alerts:
        messages.append(
            SlackMessage(text="*Snippet template fixes needed in huggingface.js*")
        )
        messages += [SlackMessage(text=alert[:2900]) for alert in alerts]
    webhook_url = os.environ["EXP_SLACK_WEBHOOK_URL"]
    send_webhook_message(webhook_url, messages)
    for table in tables[1:]:  # Slack allows one table per message
        send_webhook_message(
            webhook_url, [SlackMessage(msg_type=SlackMessageType.TABLE, text=table)]
        )

    mark_checked(
        huggingface_api,
        results,
        _load(SNIPPETS_PATH, []),
        _load(METADATA_DONE_PATH, []),
    )
