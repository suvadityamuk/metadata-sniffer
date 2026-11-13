"""
Approval Space for the metadata agent (sign in with Hugging Face; only
`AgentConfig.operators` may act): approve or reject proposed PRs, change the
agent's toggles, and start a run now.
"""

import json

import gradio as gr
from huggingface_hub import HfApi

from configuration import AgentConfig
from orchestrator import launch_run
from proposals import Proposal, open_pr
from state import load_agent_config, read, write

huggingface_api = HfApi()  # HF_TOKEN Space secret


def _check(profile: gr.OAuthProfile | None):
    if profile is None or profile.username not in AgentConfig().operators:
        raise gr.Error("Sign in with an operator account to do this.")


def pending_choices():
    proposals = read(huggingface_api, "proposals.json", [])
    return [
        f"{i}: {p['model_id']} · {p['title']}"
        for i, p in enumerate(proposals)
        if p["status"] == "pending"
    ]


def show(choice):
    if not choice:
        return ""
    p = read(huggingface_api, "proposals.json", [])[int(choice.split(":")[0])]
    return f"### [{p['model_id']}](https://huggingface.co/{p['model_id']})\n**{p['title']}**\n\n```yaml\n{json.dumps(p['metadata'], indent=2)}\n```\n\n{p['description']}"


def decide(choice, approve, profile: gr.OAuthProfile | None):
    _check(profile)
    proposals = read(huggingface_api, "proposals.json", [])
    index = int(choice.split(":")[0])
    proposal = proposals[index]
    if proposal["status"] != "pending":
        raise gr.Error("Already decided.")
    if approve:
        fields = {
            k: v for k, v in proposal.items() if k in Proposal.__dataclass_fields__
        }
        proposal["url"] = open_pr(huggingface_api, Proposal(**fields))
    proposal["status"] = "opened" if approve else "rejected"
    proposal["decided_by"] = profile.username
    write(
        huggingface_api,
        "proposals.json",
        proposals,
        f"{proposal['status']} by {profile.username}",
    )
    message = f"Opened {proposal['url']}" if approve else "Rejected"
    return gr.update(choices=pending_choices(), value=None), message


def approve_proposal(choice, profile: gr.OAuthProfile | None):
    return decide(choice, True, profile)


def reject_proposal(choice, profile: gr.OAuthProfile | None):
    return decide(choice, False, profile)


def save_settings(auto_open, per_task, cap, tasks, profile: gr.OAuthProfile | None):
    _check(profile)
    settings = {
        "auto_open_prs": auto_open,
        "num_trending_models_per_task": int(per_task),
        "max_cost_per_run_usd": float(cap),
        "tasks": [t.strip() for t in tasks.split(",") if t.strip()] or None,
    }
    write(
        huggingface_api,
        "settings.json",
        settings,
        f"Settings changed by {profile.username}",
    )
    return "Saved"


def run_now(profile: gr.OAuthProfile | None):
    _check(profile)
    job = launch_run(huggingface_api, load_agent_config(huggingface_api))
    return f"Started [{job.id}]({job.url})"


config = load_agent_config(huggingface_api)
with gr.Blocks(title="Metadata agent") as demo:
    gr.Markdown(
        "# Metadata agent\nProposed PRs wait here for approval when auto mode is off."
    )
    gr.LoginButton()
    with gr.Tab("Proposals"):
        choice = gr.Dropdown(pending_choices(), label="Pending proposals")
        details = gr.Markdown()
        with gr.Row():
            approve, reject = (
                gr.Button("Approve and open PR", variant="primary"),
                gr.Button("Reject"),
            )
        outcome = gr.Markdown()
        choice.change(show, choice, details)
        approve.click(approve_proposal, choice, [choice, outcome])
        reject.click(reject_proposal, choice, [choice, outcome])
    with gr.Tab("Settings"):
        auto_open = gr.Checkbox(config.auto_open_prs, label="Open PRs without approval")
        per_task = gr.Number(
            config.num_trending_models_per_task,
            label="Trending models per task",
            precision=0,
        )
        cap = gr.Number(config.max_cost_per_run_usd, label="Cost cap per run (USD)")
        tasks = gr.Textbox(
            ", ".join(config.tasks or []), label="Tasks (comma separated, empty: all)"
        )
        saved = gr.Markdown()
        gr.Button("Save").click(save_settings, [auto_open, per_task, cap, tasks], saved)
    with gr.Tab("Run"):
        started = gr.Markdown()
        gr.Button("Run the agent now", variant="primary").click(run_now, None, started)

demo.launch()
