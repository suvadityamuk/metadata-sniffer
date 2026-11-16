"""
Approval Space for the metadata agent (sign in with Hugging Face; only
`AgentConfig.operators` may act): approve or reject proposed PRs, change the
agent's toggles, and start a run now.
"""

import json

import gradio as gr
from huggingface_hub import HfApi

from configuration import AgentConfig
from orchestrator import launch
from proposals import decide
from state import load_agent_config, read, update_settings

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


def _decide(choice, approve, profile):
    _check(profile)
    message = decide(
        huggingface_api, int(choice.split(":")[0]), approve, profile.username
    )
    return gr.update(choices=pending_choices(), value=None), message


def approve_proposal(choice, profile: gr.OAuthProfile | None):
    return _decide(choice, True, profile)


def reject_proposal(choice, profile: gr.OAuthProfile | None):
    return _decide(choice, False, profile)


def save_settings(auto_open, per_task, cap, tasks, profile: gr.OAuthProfile | None):
    _check(profile)
    changes = {
        "auto_open_prs": auto_open,
        "num_trending_models_per_task": int(per_task),
        "max_cost_per_run_usd": float(cap),
        "tasks": [t.strip() for t in tasks.split(",") if t.strip()] or None,
    }
    update_settings(huggingface_api, changes, profile.username)
    return "Saved"


def run_now(profile: gr.OAuthProfile | None):
    _check(profile)
    run_id, job_url = launch(huggingface_api, load_agent_config(huggingface_api))
    return f"Started sweep `{run_id}` ([job]({job_url}))"


def refresh():
    """Reads the current proposals and settings; runs on every page load and on Refresh."""
    config = load_agent_config(huggingface_api)
    return (
        gr.update(choices=pending_choices(), value=None),
        "",
        config.auto_open_prs,
        config.num_trending_models_per_task,
        config.max_cost_per_run_usd,
        ", ".join(config.tasks or []),
    )


with gr.Blocks(title="Metadata agent") as demo:
    gr.Markdown(
        "# Metadata agent\nProposed PRs wait here for approval when auto mode is off."
    )
    gr.LoginButton()
    with gr.Tab("Proposals"):
        with gr.Row():
            choice = gr.Dropdown([], label="Pending proposals", scale=5)
            refresh_button = gr.Button("↻ Refresh", scale=1)
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
        auto_open = gr.Checkbox(label="Open PRs without approval")
        per_task = gr.Number(label="Trending models per task", precision=0)
        cap = gr.Number(label="Cost cap per run (USD)")
        tasks = gr.Textbox(label="Tasks (comma separated, empty: all)")
        saved = gr.Markdown()
        gr.Button("Save").click(save_settings, [auto_open, per_task, cap, tasks], saved)
    with gr.Tab("Run"):
        started = gr.Markdown()
        gr.Button("Run the agent now", variant="primary").click(run_now, None, started)

    # proposals and settings change outside this page (scheduled sweeps, Slack), so never show stale ones
    current = [choice, details, auto_open, per_task, cap, tasks]
    demo.load(refresh, None, current)
    refresh_button.click(refresh, None, current)

demo.launch()
