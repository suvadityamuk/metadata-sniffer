"""
Slack app for the metadata agent, over Socket Mode (no public URL needed).
Only `AgentConfig.slack_operators` can use it.

    /metadata-agent help               what each command does
    /metadata-agent sweep [n] [force]  sweep the top n trending models (passcode protected)
    /metadata-agent run <model-id>     check one model now
    /metadata-agent status [run-id]    recent runs, or one run's stats
    /metadata-agent pending            proposals awaiting approval, with Approve / Reject buttons
    /metadata-agent settings           current settings
    /metadata-agent set <key> <value>  auto on|off · top-n N · cap USD
"""

import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from huggingface_hub import HfApi
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from configuration import AgentConfig
from orchestrator import launch, running_sweep
from proposals import decide
from state import load_agent_config, read, update_settings

huggingface_api = HfApi()  # HF_TOKEN secret
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# `set` keys -> (setting, parser)
SETTINGS = {
    "auto": ("auto_open_prs", lambda v: {"on": True, "off": False}[v]),
    "top-n": ("top_n", int),
    "cap": ("max_cost_per_run_usd", float),
}
MAX_PASSCODE_FAILURES = 5  # per user, then sweeps from Slack are locked for an hour
passcode_failures: dict[str, list[float]] = {}


def sweep(args, user):
    """`sweep [n] [force]` previews the sweep; its button asks for the passcode before anything starts."""
    config = load_agent_config(huggingface_api)
    force = "force" in args
    numbers = [a for a in args if a.isdigit()]
    if len(numbers) + force != len(args) or len(numbers) > 1:
        return {
            "text": "Usage: `/metadata-agent sweep [n] [force]`, e.g. `/metadata-agent sweep 50`"
        }
    top_n = int(numbers[0]) if numbers else config.top_n
    if not 1 <= top_n <= config.max_top_n:
        return {"text": f"n must be between 1 and {config.max_top_n}."}
    if not os.environ.get("SWEEP_PASSCODE"):
        return {
            "text": "Sweeps from Slack are disabled until a passcode is configured (SWEEP_PASSCODE)."
        }
    if running := running_sweep(huggingface_api, config):
        return {
            "text": f"Sweep `{running}` is still running; start a new one after its report."
        }
    cache = "*re-checked* (forced)" if force else "skipped"
    preview = (
        f"*Sweep the top {top_n} trending models*{' · forced' if force else ''}\n"
        f"Models already checked at their current commit are {cache}. "
        f"Cost cap ${config.max_cost_per_run_usd} (HF Jobs + LLM). "
        f"{'Dry run: PRs only on ' + ', '.join(config.dry_run_namespaces) + '. ' if config.dry_run else ''}"
        f"PRs {'open without approval' if config.auto_open_prs else 'wait for approval'}."
    )
    button = {
        "type": "button",
        "text": {"type": "plain_text", "text": "Start sweep…"},
        "style": "danger" if force else "primary",
        "action_id": "confirm_sweep",
        "value": json.dumps({"top_n": top_n, "force": force}),
    }
    return {
        "text": f"Sweep the top {top_n}",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": preview}},
            {"type": "actions", "elements": [button]},
        ],
    }


@app.action("confirm_sweep")
def ask_passcode(ack, body, client):
    ack()
    if body["user"]["id"] not in AgentConfig().slack_operators:
        return
    params = json.loads(body["actions"][0]["value"])
    blocks = [
        {
            "type": "input",
            "block_id": "passcode",
            "label": {"type": "plain_text", "text": "Passcode"},
            "element": {"type": "plain_text_input", "action_id": "value"},
        }
    ]
    if params[
        "force"
    ]:  # forced sweeps re-check everything: say why, it is kept with the run
        blocks.append(
            {
                "type": "input",
                "block_id": "reason",
                "label": {
                    "type": "plain_text",
                    "text": "Why force? (breaking change, who asked)",
                },
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "multiline": True,
                },
            }
        )
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "start_sweep",
            "title": {"type": "plain_text", "text": "Start sweep"},
            "submit": {"type": "plain_text", "text": "Start"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "private_metadata": json.dumps(
                {**params, "response_url": body["response_url"]}
            ),
            "blocks": blocks,
        },
    )


@app.view("start_sweep")
def start_sweep(ack, body, view):
    user, now = body["user"]["id"], time.time()
    failures = [t for t in passcode_failures.get(user, []) if now - t < 3600]
    values = view["state"]["values"]
    passcode = values["passcode"]["value"]["value"] or ""
    if (
        user not in AgentConfig().slack_operators
        or len(failures) >= MAX_PASSCODE_FAILURES
    ):
        ack(
            response_action="errors",
            errors={"passcode": "Too many wrong passcodes; try again in an hour."},
        )
        return
    if not hmac.compare_digest(
        passcode.encode(), os.environ["SWEEP_PASSCODE"].encode()
    ):
        passcode_failures[user] = [*failures, now]
        ack(response_action="errors", errors={"passcode": "Wrong passcode."})
        return
    ack()  # closes the dialog; the sweep is announced through the original command's response URL
    params = json.loads(view["private_metadata"])
    config = load_agent_config(huggingface_api)
    if running := running_sweep(huggingface_api, config):
        message = {
            "response_type": "ephemeral",
            "text": f"Sweep `{running}` is still running; not starting another.",
        }
    else:
        reason = (values.get("reason") or {}).get("value", {}).get("value")
        run_id, job_url = launch(
            huggingface_api,
            config,
            top_n=params["top_n"],
            force=params["force"],
            reason=reason,
            started_by=f"slack:{user}",
        )
        kind = "a forced sweep" if params["force"] else "a sweep"
        message = {
            "response_type": "in_channel",  # sweeps and runs are announced to the channel
            "text": f"<@{user}> started {kind} of the top {params['top_n']}: `{run_id}` (<{job_url}|job>). "
            "The report will be posted when it finishes.",
        }
    requests.post(params["response_url"], json=message, timeout=10)


def run(args, user):
    if len(args) != 1:
        return {"text": "Usage: `/metadata-agent run <model-id>`"}
    model_id = huggingface_api.model_info(
        args[0]
    ).id  # fails clearly on unknown or private repos
    run_id, job_url = launch(
        huggingface_api, load_agent_config(huggingface_api), model_id
    )
    return {
        "response_type": "in_channel",
        "text": f"<@{user}> started run `{run_id}` for `{model_id}` (<{job_url}|job>).",
    }


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
            f"auto `{'on' if config.auto_open_prs else 'off'}` · top-n `{config.top_n}` · "
            f"cap `${config.max_cost_per_run_usd}` · "
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
        "sweep [n] [force]",
        "Check the top n of the Hub's trending models now (default: the `top-n` setting). Shows what will run, "
        "then asks for the passcode. Models checked before at the same commit are skipped; `force` re-checks "
        "them (for breaking changes on the Hub) and asks why.",
        "sweep 50",
    ),
    (
        "run <model-id>",
        "Check one model now, cached or not. Replies with a run ID.",
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
        "Change a setting: `auto on|off` (open PRs without approval), `top-n N` (default sweep size), "
        "`cap USD` (cost cap per run).",
        "set top-n 30",
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
        respond(
            text="Only the metadata agent's operators can use this command.",
            response_type="ephemeral",
        )
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
            response_type="ephemeral",
        )
        return
    try:
        # only the person who typed the command sees the reply, unless a sweep/run announces itself
        respond(
            **{
                "response_type": "ephemeral",
                **COMMANDS[action](args, command["user_id"]),
            }
        )
    except Exception as e:  # tell the operator instead of failing silently
        respond(text=f"`{action}` failed: {e}", response_type="ephemeral")


def _decide(ack, body, respond, approve: bool):
    ack()
    user = body["user"]["id"]
    if user not in AgentConfig().slack_operators:
        return
    message = decide(
        huggingface_api, int(body["actions"][0]["value"]), approve, f"slack:{user}"
    )
    respond(text=message, replace_original=False, response_type="ephemeral")


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
