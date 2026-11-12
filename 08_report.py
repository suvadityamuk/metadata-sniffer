"""
Posts the run's report to Slack as a table: one row per model and check
(snippet / metadata) with its PR, if any, and the PR's status.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from configuration import AgentConfig
from state import read
from utilities import SlackMessage, SlackMessageType, send_webhook_message

load_dotenv()

RESULTS_PATH = Path("execution/results.json")
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


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    space_url = f"https://huggingface.co/spaces/{AgentConfig.approval_space}"
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    prs = read(huggingface_api, "prs.json", [])
    rows = build_rows(
        results, read(huggingface_api, "proposals.json", []), prs, space_url
    )

    header = [[_text("Model"), _text("Check"), _text("PR"), _text("Status")]]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    webhook_url = os.environ["EXP_SLACK_WEBHOOK_URL"]
    send_webhook_message(
        webhook_url,
        [
            SlackMessage(
                text=f"Metadata agent report {today}", msg_type=SlackMessageType.HEADER
            )
        ],
    )
    for i in range(0, len(rows), MAX_TABLE_ROWS):  # one table per message
        send_webhook_message(
            webhook_url,
            [
                SlackMessage(
                    msg_type=SlackMessageType.TABLE,
                    text=header + rows[i : i + MAX_TABLE_ROWS],
                )
            ],
        )
