"""
JSON-only chat client for an open model on HF Inference Providers.
Every call is charged to the run's `Budget`, alongside the HF Jobs spend.
"""

import json
import os

from huggingface_hub import InferenceClient

from configuration import AgentConfig
from jobs import Budget
from utilities import setup_logging

logger = setup_logging(__name__)


def ask_json(
    system: str, user: str, budget: Budget, config: AgentConfig
) -> dict | None:
    """Returns the model's JSON answer, or None when the budget is spent or the answer is not JSON."""
    if not budget.reserve(config.llm_max_call_usd):
        logger.warning("Budget exhausted, skipping LLM call")
        return None
    response = InferenceClient(token=os.environ["HF_TOKEN"]).chat_completion(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        max_tokens=config.llm_max_tokens,
        temperature=0,
    )
    # settle the reservation with the actual cost, when the provider reports it
    cost = getattr(response.usage, "cost", None)
    if cost is not None:
        budget.charge(cost - config.llm_max_call_usd)
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM answer was not valid JSON")
        return None
