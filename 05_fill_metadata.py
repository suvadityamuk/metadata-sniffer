"""
Fills metadata fields that are missing from a model's README front matter,
using only what the README itself states. The LLM proposes each value with a
verbatim quote; a value is kept only if the quote is in the README and the
value checks out against the Hub (repos exist, license/language/library/task/
metric are valid Hub values, numbers appear in the README). Existing metadata
is never changed. Proposals go to `execution/proposals.json`.
"""

import json
import os
import re

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard

from configuration import AgentConfig
from jobs import Budget
from llm import ask_json
from proposals import Proposal, add_proposals, load_vocabulary, open_pr
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)

SNIPPETS_PATH = "execution/snippets.json"
FIELDS = (
    "base_model",
    "datasets",
    "library_name",
    "pipeline_tag",
    "license",
    "language",
    "tags",
    "metrics",
    "model-index",
)

SYSTEM_PROMPT = """You extract Hugging Face model card metadata from a README.
Only use facts the README states explicitly. Never infer, guess or use outside knowledge.
For every field you fill, give the exact sentence or table row from the README that states it.
Prefer Hugging Face Hub ids (huggingface.co/... links) over other links, and Hub datasets over anything else.
Leave a field out when the README does not state it unambiguously.
datasets means only the data the model was trained or fine-tuned on, never evaluation benchmarks.
base_model means only the model this one was fine-tuned, quantized, merged or adapted from, never models it is compared to.
Answer with JSON: {"fields": {"<field>": {"value": <value>, "evidence": "<verbatim quote>"}}}
Value formats: base_model, datasets: list of Hub repo ids "owner/name"; license, library_name, pipeline_tag: string;
language: list of ISO 639-1 codes; tags, metrics: list of strings; model-index: list in the Hub model-index format."""


# the quote must state the value's role, e.g. a benchmark the model was evaluated on is not training data
ROLE_EVIDENCE = {
    "datasets": re.compile(
        r"train|fine-?tun|sft|distill|pre-?train|corpus|instruction[- ]tun|dpo|rlhf|grpo",
        re.I,
    ),
    "base_model": re.compile(
        r"fine-?tun|based on|built (on|upon)|quantiz|merge|adapter|lora|derived|distill|"
        r"version of|initiali[sz]ed from|continued|post-?train|trained (on top|from)",
        re.I,
    ),
}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[`*_|]", " ", text).split()).lower()


def resolve_repo(
    huggingface_api: HfApi, repo_id: str, readme: str, repo_type: str
) -> bool:
    """The repo must exist and be named in the README, either fully or by its unique name."""
    if not huggingface_api.repo_exists(repo_id, repo_type=repo_type):
        return False
    if repo_id.lower() in readme.lower():
        return True
    name = repo_id.split("/")[-1]
    if name.lower() not in readme.lower():
        return False
    # only the name is mentioned: accept the most downloaded repo with exactly that name
    search = (
        huggingface_api.list_models
        if repo_type == "model"
        else huggingface_api.list_datasets
    )
    exact = [
        r.id
        for r in search(search=name, sort="downloads", limit=20)
        if r.id.split("/")[-1] == name
    ]
    return bool(exact) and exact[0] == repo_id


def verify(field, value, readme, huggingface_api, vocabulary):
    """Returns the verified value (possibly narrowed), or None."""
    as_list = value if isinstance(value, list) else [value]
    if field in ("base_model", "datasets"):
        repo_type = "model" if field == "base_model" else "dataset"
        kept = [
            v
            for v in as_list
            if isinstance(v, str)
            and resolve_repo(huggingface_api, v, readme, repo_type)
        ]
    elif field in ("license", "library_name", "pipeline_tag"):
        return value if isinstance(value, str) and value in vocabulary[field] else None
    elif field == "language":
        kept = [v for v in as_list if v in vocabulary["language"]]
    elif field == "metrics":
        kept = [
            v
            for v in as_list
            if v in vocabulary["metrics"] and _norm(v) in _norm(readme)
        ]
    elif field == "tags":
        kept = [v for v in as_list if isinstance(v, str) and _norm(v) in _norm(readme)]
    elif field == "model-index":
        # every reported number must appear in the README
        numbers = [
            str(m.get("value"))
            for entry in as_list
            for r in entry.get("results", [])
            for m in r.get("metrics", [])
        ]
        return value if numbers and all(n in readme for n in numbers) else None
    return kept or None


def propose(model_id, huggingface_api, vocabulary, budget, config) -> Proposal | None:
    card = ModelCard.load(model_id, token=huggingface_api.token)
    existing = card.data.to_dict()
    missing = [f for f in FIELDS if not existing.get(f)]
    if not missing:
        return None
    answer = ask_json(
        SYSTEM_PROMPT,
        f"Missing fields: {missing}\n\nREADME:\n{card.text[: config.llm_max_readme_chars]}",
        budget,
        config,
    )
    if not answer:
        return None

    metadata, evidence = {}, {}
    for field, proposal in answer.get("fields", {}).items():
        if field not in missing or not isinstance(proposal, dict):
            continue
        quote = str(proposal.get("evidence", ""))
        if len(quote) < 10 or _norm(quote) not in _norm(card.text):
            logger.info(f"{model_id}: dropped {field}, evidence not found in README")
            continue
        if field in ROLE_EVIDENCE and not ROLE_EVIDENCE[field].search(quote):
            logger.info(
                f"{model_id}: dropped {field}, evidence does not state its role"
            )
            continue
        if field in ROLE_EVIDENCE:
            # each repo must be named in the quote itself, so the quote ties it to the role
            values = proposal.get("value")
            values = [values] if isinstance(values, str) else values or []
            proposal["value"] = [
                v
                for v in values
                if isinstance(v, str) and v.split("/")[-1].lower() in quote.lower()
            ]
        value = verify(
            field, proposal.get("value"), card.text, huggingface_api, vocabulary
        )
        if value is None:
            logger.info(f"{model_id}: dropped {field}, value failed verification")
            continue
        metadata[field], evidence[field] = value, quote
    if not metadata:
        return None

    lines = [
        f"- **{field}**: `{json.dumps(value)}`\n  > {evidence[field]}"
        for field, value in metadata.items()
    ]
    return Proposal(
        model_id=model_id,
        kind="metadata",
        metadata=metadata,
        title=f"Add missing metadata: {', '.join(metadata)}",
        description="This PR adds metadata that the model card already states, so the Hub can index and link "
        "it (e.g. base model tree, dataset pages, license filter).\n\nEach value is backed by a quote from the README:\n\n"
        + "\n".join(lines)
        + "\n\nNo existing metadata is changed. Opened by the Hugging Face metadata agent, feedback welcome!",
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    agent_config = AgentConfig()
    budget = Budget.for_run(agent_config.max_cost_per_run_usd)
    vocabulary = load_vocabulary()

    model_ids = [m["id"] for m in json.load(open(SNIPPETS_PATH))]
    new_proposals = []
    for model_id in model_ids:
        try:
            proposal = propose(
                model_id, huggingface_api, vocabulary, budget, agent_config
            )
        except Exception as e:
            logger.error(f"{model_id}: {e}")
            continue
        if proposal is None:
            continue
        logger.info(f"{model_id}: proposing {list(proposal.metadata)}")
        if agent_config.auto_open_prs:
            logger.info(f"Opened {open_pr(huggingface_api, proposal)}")
        new_proposals.append(proposal)
    add_proposals(new_proposals)
