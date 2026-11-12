"""
Reads Radar's model release calendar (posted in #transformers-new-models) and
finds the Hub repos of releases that are due within `radar_days_ahead` days or
were released within `radar_days_back` days, so they get the same checks as
trending models. Release orgs are mapped to Hub accounts via
`AgentConfig.radar_org_authors`, falling back to a Hub search on the org name.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from huggingface_hub import HfApi

from configuration import AgentConfig
from utilities import setup_logging

logger = setup_logging(__name__)

ROW = re.compile(
    r"(?P<date>~?(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?P<day>\d+)(?:[–-]\d+)?/(?P<month>\d+)|TBD)\s+"
    r"(?:today|in \d+d|\d+d ago)?\s*"
    r":(?P<pr_status>[a-z_]+): (?P<name>[^·:]+?)\s*(?:\((?P<org>[^()]+)\))?\s*·"
)


@dataclass
class Release:
    name: str
    org: str | None
    days_from_today: int | None  # None: no date yet


def _days_from_today(day: int, month: int, today: date) -> int:
    """Rows carry no year: pick the year that puts the date closest to today."""
    candidates = [date(today.year + offset, month, day) for offset in (-1, 0, 1)]
    return min(((d - today).days for d in candidates), key=abs)


def parse_calendar(text: str, today: date | None = None) -> list[Release]:
    """Uses each row's absolute date (start of a range), so a stored calendar never goes stale."""
    today = today or datetime.now(timezone.utc).date()
    releases = []
    for match in ROW.finditer(text):
        days = None
        if match["day"]:
            days = _days_from_today(int(match["day"]), int(match["month"]), today)
        releases.append(Release(match["name"].strip(), match["org"], days))
    return releases


def releases_to_watch(releases: list[Release], config: AgentConfig) -> list[Release]:
    return [
        r
        for r in releases
        if r.days_from_today is not None
        and -config.radar_days_back <= r.days_from_today <= config.radar_days_ahead
    ]


def hub_authors(huggingface_api: HfApi, org: str, config: AgentConfig) -> list[str]:
    if org in config.radar_org_authors:
        return config.radar_org_authors[org]
    # an account whose name starts with the org, e.g. "Liquid AI" -> "LiquidAI", "MiniMax" -> "MiniMaxAI"
    compact = re.sub(r"\W", "", org).lower()
    authors = [
        m.id.split("/")[0]
        for m in huggingface_api.list_models(
            search=org.split()[0], sort="downloads", limit=50
        )
    ]
    return [
        a
        for a in dict.fromkeys(authors)
        if re.sub(r"\W", "", a).lower().startswith(compact)
    ][:1]


def new_models(
    huggingface_api: HfApi, releases: list[Release], config: AgentConfig
) -> list[str]:
    """Public repos created by the releases' orgs within the watch window."""
    since = datetime.now(timezone.utc) - timedelta(days=config.radar_days_back)
    model_ids = []
    for org in {r.org for r in releases if r.org}:
        for author in hub_authors(huggingface_api, org, config):
            for model in huggingface_api.list_models(
                author=author, sort="createdAt", limit=20
            ):
                if model.created_at and model.created_at >= since:
                    model_ids.append(model.id)
    return model_ids
