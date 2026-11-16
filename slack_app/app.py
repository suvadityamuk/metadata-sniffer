"""
Slack app for the metadata agent, over Socket Mode (no public URL needed).
Only `AgentConfig.slack_operators` can use it.

    /metadata-agent help               what each command does
    /metadata-agent sweep              sweep the trending models now
    /metadata-agent run <model-id>     check one model now
    /metadata-agent status [run-id]    recent runs, or one run's stats
    /metadata-agent pending            proposals awaiting approval, with Approve / Reject buttons
    /metadata-agent settings           current settings
    /metadata-agent set <key> <value>  auto on|off · per-task N · cap USD · tasks a,b|all
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from huggingface_hub import HfApi
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from configuration import AgentConfig
from orchestrator import launch
from proposals import decide
from state import load_agent_config, read, update_settings

huggingface_api = HfApi()  # HF_TOKEN secret
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# `set` keys -> (setting, parser)
SETTINGS = {
    "auto": ("auto_open_prs", lambda v: {"on": True, "off": False}[v]),
    "per-task": ("num_trending_models_per_task", int),
    "cap": ("max_cost_per_run_usd", float),
    "tasks": (
        "tasks",
        lambda v: None if v == "all" else [t for t in v.split(",") if t],
    ),
}


def sweep(args, user):
    run_id, job_url = launch(huggingface_api, load_agent_config(huggingface_api))
    return {
        "text": f"Sweep `{run_id}` started (<{job_url}|job>). The report will be posted when it finishes."
    }


def run(args, user):
    if len(args) != 1:
        return {"text": "Usage: `/metadata-agent run <model-id>`"}
    model_id = huggingface_api.model_info(
        args[0]
    ).id  # fails clearly on unknown or private repos
    run_id, job_url = launch(
        huggingface_api, load_agent_config(huggingface_api), model_id
    )
    return {"text": f"Run `{run_id}` for `{model_id}` started (<{job_url}|job>)."}


def status(args, user):
    if args:
        record = read(huggingface_api, f"runs/{args[0]}.json", None)
        return {
            "text": f"```{json.dumps(record, indent=2)}```"
            if record
            else f"No run `{args[0]}`."
        }
    files = huggingface_api.list_repo_files(AgentConfig.state_repo, repo_type="dataset")
    runs = sorted(
        (f for f in files if f.startswith("runs/")),
        key=lambda f: f.split("-", 1)[1],
        reverse=True,
    )
    lines = [f"• `{f[len('runs/') : -len('.json')]}`" for f in runs[:10]]
    return {
        "text": "Recent runs (`/metadata-agent status <run-id>` for stats):\n"
        + "\n".join(lines or ["none yet"])
    }


def pending(args, user):
    proposals = read(huggingface_api, "proposals.json", [])
    blocks = []
    for index, p in enumerate(proposals):
        if p["status"] != "pending":
            continue
        blocks += [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*<https://huggingface.co/{p['model_id']}|{p['model_id']}>* · {p['kind']} · `{p.get('run_id')}`\n"
                        f"{p['title']}\n```{json.dumps(p['metadata'], indent=1)[:2500]}```"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve_proposal",
                        "value": str(index),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "action_id": "reject_proposal",
                        "value": str(index),
                    },
                ],
            },
        ]
    return (
        {"text": "Pending proposals", "blocks": blocks[:50]}
        if blocks
        else {"text": "Nothing pending."}
    )


def settings(args, user):
    config = load_agent_config(huggingface_api)
    return {
        "text": (
            f"auto `{'on' if config.auto_open_prs else 'off'}` · per-task `{config.num_trending_models_per_task}` · "
            f"cap `${config.max_cost_per_run_usd}` · tasks `{','.join(config.tasks or ['all'])}` · "
            f"dry run `{config.dry_run}` (PRs only on {config.dry_run_namespaces})"
        )
    }


def set_setting(args, user):
    if len(args) != 2 or args[0] not in SETTINGS:
        return {"text": f"Usage: `/metadata-agent set <{'|'.join(SETTINGS)}> <value>`"}
    key, parse = SETTINGS[args[0]]
    update_settings(huggingface_api, {key: parse(args[1])}, f"slack:{user}")
    return settings([], user)


# (usage, what it does, example); shown by `/metadata-agent help`
HELP = [
    (
        "sweep",
        "Check the top trending models of every task now. Replies with a sweep ID.",
        "sweep",
    ),
    (
        "run <model-id>",
        "Check one model now. Replies with a run ID.",
        "run google/vit-base-patch16-224",
    ),
    (
        "status [run-id]",
        "Recent sweeps and runs, or one run's stats: models, snippet results, proposals, cost.",
        "status sweep-20261117-0700-a1b2",
    ),
    (
        "pending",
        "Proposed PRs waiting for approval, with Approve / Reject buttons.",
        "pending",
    ),
    ("settings", "Current settings, including dry run.", "settings"),
    (
        "set <key> <value>",
        "Change a setting: `auto on|off` (open PRs without approval), `per-task N`, "
        "`cap USD` (cost cap per run), `tasks a,b|all`.",
        "set per-task 3",
    ),
]


def help_message(args, user):
    lines = [
        f"*`/metadata-agent {usage}`*\n{what}  e.g. `/metadata-agent {example}`"
        for usage, what, example in HELP
    ]
    config = AgentConfig()
    intro = (
        "*Metadata agent*: checks the *Use this model* snippets and README metadata of trending Hub models, "
        f"and proposes fixes as PRs. Sweeps also run on their own, {config.schedule_text}."
    )
    if config.dry_run:
        intro += f"\n_Dry run: PRs can only be opened on repos under {', '.join(config.dry_run_namespaces)}._"
    return {
        "text": "Metadata agent commands",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}}
            for text in [intro, *lines]
        ],
    }


COMMANDS = {
    "help": help_message,
    "sweep": sweep,
    "run": run,
    "status": status,
    "pending": pending,
    "settings": settings,
    "set": set_setting,
}


@app.command("/metadata-agent")
def handle_command(ack, command, respond):
    ack()
    if command["user_id"] not in AgentConfig().slack_operators:
        respond("Only the metadata agent's operators can use this command.")
        return
    action, *args = command.get("text", "").split() or ["help"]
    if action not in COMMANDS:
        unknown = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Unknown command `{action}`. Here is what I can do:",
            },
        }
        respond(
            text=f"Unknown command {action}",
            blocks=[unknown, *help_message([], None)["blocks"]],
        )
        return
    try:
        respond(**COMMANDS[action](args, command["user_id"]))
    except Exception as e:  # tell the operator instead of failing silently
        respond(f"`{action}` failed: {e}")


def _decide(ack, body, respond, approve: bool):
    ack()
    user = body["user"]["id"]
    if user not in AgentConfig().slack_operators:
        return
    message = decide(
        huggingface_api, int(body["actions"][0]["value"]), approve, f"slack:{user}"
    )
    respond(text=message, replace_original=False)


app.action("approve_proposal")(
    lambda ack, body, respond: _decide(ack, body, respond, True)
)
app.action("reject_proposal")(
    lambda ack, body, respond: _decide(ack, body, respond, False)
)


class Health(BaseHTTPRequestHandler):
    """Spaces expect an open port; Slack traffic goes over the socket."""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )  # shows the Socket Mode connection in the Space logs
    threading.Thread(
        target=HTTPServer(
            ("0.0.0.0", int(os.environ.get("PORT", 7860))), Health
        ).serve_forever,
        daemon=True,
    ).start()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
