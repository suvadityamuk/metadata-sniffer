# Trending Respositories Investigation

This repository will analyse the top N (N=100 at the moment) trending models on the Hugging Face Hub.
The purpose of this is to check for `library name`, and the `pipeline` tag, which help with code snippet
population on the model page. We want to alert the 🥑 team of any metadata issues.
This runs daily through GitHub Actions (`01_parse_trending_models.py`, `04_upstream_transformers.py`).

## Metadata agent

The agent checks the top `num_trending_models_per_task` trending models of every task (plus new repos of
releases on Radar's calendar), fixes what it can and proposes PRs to the model repos.

| Stage | What it does |
|---|---|
| `02_process_code.py` | Selects models (non-gated, with a discussion tab) and collects their *Use this model* snippets for the libraries they are tagged with (transformers, diffusers, sentence-transformers, GGUF/llama.cpp). |
| `03_run_jobs.py` | Runs the snippets on [HF Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs): cheapest flavor that fits, one job per flavor batch, **no secrets in the job**, hard cost cap per run. |
| `06_repair_snippets.py` | An LLM diagnoses failed snippets; fixes are re-run on Jobs. Metadata causes become PR proposals, huggingface.js template causes become Slack alerts. |
| `05_fill_metadata.py` | Proposes metadata that is missing from the front matter but stated in the README. Every value needs a verbatim quote and is checked against the Hub; existing values are never changed. |
| `07_track_prs.py` | Follows the agent's PRs; answers comments from the repo's owner/org/team (as proposals), flags others for a human. |
| `08_report.py` | Posts a Slack table: model, check, PR, status. |

Proposals are opened right away when `auto_open_prs` is on, otherwise they wait for approval in the
[approval Space](space/) (sign in with Hugging Face; only `AgentConfig.operators` can act). The Space also edits
the toggles, takes Radar's calendar and starts runs. State (settings, proposals, tracked PRs) lives in the private
dataset `AgentConfig.state_repo` (`state.py`).

```bash
python orchestrator.py run        # one run as an HF Job
python orchestrator.py schedule   # weekday schedule as a scheduled HF Job
python space/deploy.py            # (re)deploy the approval Space
```

Environment: `HF_TOKEN` (Jobs, Inference Providers, PRs, state) and `EXP_SLACK_WEBHOOK_URL`.

Some edge cases to note:
1. The repository might have no discussion tab (no need to engage)
2. The repository may be gated (skipped)
