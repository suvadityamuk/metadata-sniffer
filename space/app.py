"""
Approval Space for the metadata agent (sign in with Hugging Face; only
`AgentConfig.operators` may act): approve or reject proposed PRs, change the
agent's toggles, and start a run now.
"""

import difflib
import json

import gradio as gr
import yaml
from huggingface_hub import HfApi, ModelCard

from configuration import AgentConfig, ExecuteCodeConfig
from orchestrator import launch
from proposals import decide
from snippet_runner import LOG_BEGIN, LOG_END, RESULT_MARKER
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


def _job_link(job_id: str) -> str:
    return f"[{job_id[:8]}](https://huggingface.co/jobs/{ExecuteCodeConfig().jobs_namespace}/{job_id})"


def summary_section(index: int, p: dict) -> str:
    changes = "\n".join(f"- `{k}`: `{json.dumps(v)}`" for k, v in p["metadata"].items())
    return (
        f"### [{p['model_id']}](https://huggingface.co/{p['model_id']})\n"
        f"**{p['title']}**  \n#{index} · {p['kind']} · run `{p.get('run_id') or 'n/a'}` · {p['status']}\n\n{changes}"
    )


def checks_section(evidence: dict) -> str:
    checks = evidence.get("checks") or []
    if not checks:
        return f"No snippet check recorded for this proposal ({evidence.get('snippet_status') or 'it predates check recording'})."
    rows = [
        f"| {c['label']} | {'✅ passed' if c['passed'] else '❌ failed'} | {c['detail']} | {_job_link(c['job_id'])} |"
        for c in checks
    ]
    return "| Snippet | Result | Detail | Job |\n|---|---|---|---|\n" + "\n".join(rows)


def _diff(before: str, after: str, before_name: str, after_name: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(), after.splitlines(), before_name, after_name, lineterm=""
    )
    return "```diff\n" + ("\n".join(lines) or "(no change)") + "\n```"


def snippet_diff_section(evidence: dict) -> str:
    if "snippet_before" in evidence:
        return _diff(
            evidence["snippet_before"],
            evidence["snippet_after"],
            "snippet (now)",
            "snippet (with this PR)",
        )
    snippets = evidence.get("snippets") or []
    if not snippets:
        return "This proposal does not change a snippet."
    return (
        "This proposal does not change the snippets. The snippets that were checked:\n\n"
        + "\n\n".join(f"```python\n{code}\n```" for code in snippets)
    )


def metadata_diff_section(p: dict) -> str:
    """Front matter now vs. with this PR, computed against the current model card."""
    current = ModelCard.load(p["model_id"], token=huggingface_api.token).data.to_dict()
    proposed = dict(current)
    for key, value in p["metadata"].items():
        if p["overwrite"] or not proposed.get(key):
            proposed[key] = value
    dump = lambda data: yaml.safe_dump(data, sort_keys=False, allow_unicode=True)  # noqa: E731
    return _diff(
        dump(current), dump(proposed), "README.md (now)", "README.md (with this PR)"
    )


def snippet_logs(job_id: str, log_id: str) -> str:
    """The snippet's own stdout/stderr from its job, plus its result line."""
    lines = huggingface_api.fetch_job_logs(
        job_id=job_id, namespace=ExecuteCodeConfig().jobs_namespace
    )
    out, block = [], None
    for line in (line.rstrip("\n") for line in lines):
        if line.startswith(LOG_BEGIN):
            model_id, index = line[len(LOG_BEGIN) :].rsplit(" ", 1)
            block = [f"----- snippet {index} -----"] if model_id == log_id else None
        elif line.startswith(LOG_END):
            out += block or []
            block = None
        elif block is not None:
            block.append(line)
        elif (
            line.startswith(RESULT_MARKER)
            and json.loads(line[len(RESULT_MARKER) :])["id"] == log_id
        ):
            result = json.loads(line[len(RESULT_MARKER) :])
            out.append(
                f"result: exit code {result['exit_code']}, {result.get('seconds')}s, output {json.dumps(result.get('output'))[:300]}"
            )
            if result.get("error"):
                out.append(result["error"])
    return "\n".join(out)


def logs_section(evidence: dict) -> str:
    parts = []
    for check in evidence.get("checks") or []:
        try:
            log = snippet_logs(check["job_id"], check["model_id"])
        except Exception as e:  # the job may be gone; the rest of the page still works
            log = f"could not fetch the logs: {e}"
        parts.append(
            f"===== {check['label']} · job {check['job_id']} =====\n{log or '(no output)'}"
        )
    return "\n\n".join(parts) or "No job logs recorded for this proposal."


def show(choice):
    if not choice:
        return [""] * 6
    index = int(choice.split(":")[0])
    p = read(huggingface_api, "proposals.json", [])[index]
    evidence = p.get("evidence") or {}
    return (
        summary_section(index, p),
        checks_section(evidence),
        snippet_diff_section(evidence),
        metadata_diff_section(p),
        logs_section(evidence),
        p["description"],
    )


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
        *[""] * 6,
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
        summary = gr.Markdown()
        with gr.Accordion("Snippet check", open=True):
            checks = gr.Markdown()
        with gr.Accordion("Snippet diff", open=True):
            snippet_diff = gr.Markdown()
        with gr.Accordion("Metadata diff (README front matter)", open=True):
            metadata_diff = gr.Markdown()
        with gr.Accordion("Job logs", open=False):
            logs = gr.Code(language=None, lines=20, interactive=False)
        with gr.Accordion("PR description", open=False):
            description = gr.Markdown()
        details = [summary, checks, snippet_diff, metadata_diff, logs, description]
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
    current = [choice, *details, auto_open, per_task, cap, tasks]
    demo.load(refresh, None, current)
    refresh_button.click(refresh, None, current)

demo.launch()
