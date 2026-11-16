"""
HF Jobs helpers shared by the snippet checks and the snippet repair loop:
flavor selection from the live hardware catalogue, a per-run cost budget, and
batched, secret-free snippet execution with results parsed from the job logs.
"""

import ast
import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, JobInfo

from configuration import ExecuteCodeConfig
from snippet_runner import RESULT_MARKER
from utilities import setup_logging

logger = setup_logging(__name__)

HARDWARE_URL = "https://huggingface.co/api/jobs/hardware"
FP8_GPUS = ("L4", "L40S", "H100", "H200", "B200")  # compute capability >= 8.9
RUNNER_SOURCE = Path(__file__).with_name("snippet_runner.py").read_text()
TERMINAL_STAGES = ("COMPLETED", "ERROR", "CANCELED", "DELETED")
EXCEPTION_LINE = re.compile(r"^[\w.]+(Error|Exception|Interrupt)\b")
BUDGET_PATH = Path("execution/budget.json")


@dataclass
class Flavor:
    name: str
    gpu: str | None
    memory_gb: float  # total VRAM for GPU flavors, RAM for CPU flavors
    storage_gb: float
    usd_per_minute: float


@dataclass
class Budget:
    """Per-run spend cap, persisted so every stage of a run draws from the same cap."""

    cap_usd: float
    reserved_usd: float = 0.0

    @classmethod
    def for_run(cls, cap_usd: float) -> "Budget":
        reserved = (
            json.loads(BUDGET_PATH.read_text())["reserved_usd"]
            if BUDGET_PATH.exists()
            else 0.0
        )
        return cls(cap_usd=cap_usd, reserved_usd=reserved)

    def reserve(self, usd: float) -> bool:
        if self.reserved_usd + usd > self.cap_usd:
            return False
        self.charge(usd)
        return True

    def charge(self, usd: float):
        self.reserved_usd += usd
        BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
        BUDGET_PATH.write_text(json.dumps({"reserved_usd": self.reserved_usd}))


def _gb(text: str) -> float:
    return float(text.split()[0])


def load_flavors() -> list[Flavor]:
    """Live Jobs hardware catalogue, cheapest first."""
    flavors = []
    for hw in requests.get(HARDWARE_URL, timeout=30).json():
        accelerator = hw.get("accelerator")
        if accelerator and accelerator.get("type") != "gpu":
            continue
        flavors.append(
            Flavor(
                name=hw["name"],
                gpu=accelerator["model"] if accelerator else None,
                memory_gb=_gb(accelerator["vram"]) * int(accelerator["quantity"])
                if accelerator
                else _gb(hw["ram"]),
                storage_gb=_gb(hw["ephemeralStorage"]),
                usd_per_minute=hw["unitCostUSD"],
            )
        )
    return sorted(flavors, key=lambda f: f.usd_per_minute)


def pick_flavor(
    weights_gb: float,
    needs_fp8: bool,
    cpu_only: bool,
    flavors: list[Flavor],
    config: ExecuteCodeConfig,
) -> Flavor | None:
    """Cheapest flavor whose memory and disk fit the model."""
    memory_needed = weights_gb * config.memory_overhead + config.memory_headroom_gb
    disk_needed = weights_gb * 1.5 + config.environment_disk_gb
    for flavor in flavors:
        if cpu_only and flavor.gpu is not None:
            continue
        if (
            flavor.gpu is None
            and not cpu_only
            and weights_gb > config.max_cpu_weights_gb
        ):
            continue
        if needs_fp8 and flavor.gpu not in FP8_GPUS:
            continue
        if flavor.memory_gb >= memory_needed and flavor.storage_gb >= disk_needed:
            return flavor
    return None


def prepare_snippet(code: str) -> str:
    """Drops install lines and binds the snippet's final value to `_result` so the runner can inspect it."""
    lines = [
        line
        for line in code.splitlines()
        if not line.lstrip().startswith(("!", "# pip", "%pip"))
    ]
    tree = ast.parse("\n".join(lines))
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "print"
            and value.args
        ):
            value = value.args[0]  # `print(x)` -> inspect x
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())], value=value
        )
        ast.fix_missing_locations(tree)
    elif (
        tree.body
        and isinstance(tree.body[-1], ast.Assign)
        and isinstance(tree.body[-1].targets[0], ast.Name)
    ):
        return f"{ast.unparse(tree)}\n_result = {tree.body[-1].targets[0].id}"
    return ast.unparse(tree)


def snippet_timeout_s(weights_gb: float, config: ExecuteCodeConfig) -> int:
    """Covers download + load + one forward/generate call."""
    return int(
        config.base_snippet_timeout_s + weights_gb * config.seconds_per_weight_gb
    )


def launch_batch(
    huggingface_api: HfApi,
    flavor: Flavor,
    models: list[dict],
    timeout_s: int,
    config: ExecuteCodeConfig,
) -> JobInfo:
    """One job per batch. Only the snippets travel to the job: no token, no secrets."""
    batch = base64.b64encode(json.dumps(models).encode()).decode()
    runner = base64.b64encode(RUNNER_SOURCE.encode()).decode()
    dependencies = " ".join(f"--with {d}" for d in config.dependencies)
    return huggingface_api.run_job(
        image=config.docker_image,
        command=[
            "/bin/bash",
            "-c",
            f"echo $SNIPPET_RUNNER | base64 -d > /tmp/runner.py && "
            f"uv run --python 3.12 {dependencies} python /tmp/runner.py",
        ],
        env={
            "SNIPPET_BATCH": batch,
            "SNIPPET_RUNNER": runner,
            "HOME": "/tmp",
            "HF_HOME": "/tmp/hf",
        },
        flavor=flavor.name,
        timeout=timeout_s,
        namespace=config.jobs_namespace,
    )


def collect_results(
    huggingface_api: HfApi, job_ids: list[str], config: ExecuteCodeConfig
) -> dict[str, list[dict]]:
    """Waits for all jobs and returns the per-snippet results, keyed by job id."""
    pending, results = set(job_ids), {}
    while pending:
        for job_id in list(pending):
            job = huggingface_api.inspect_job(
                job_id=job_id, namespace=config.jobs_namespace
            )
            if job.status.stage not in TERMINAL_STAGES:
                continue
            pending.discard(job_id)
            logs = huggingface_api.fetch_job_logs(
                job_id=job_id, namespace=config.jobs_namespace
            )
            results[job_id] = [
                json.loads(line[len(RESULT_MARKER) :])
                for line in logs
                if line.startswith(RESULT_MARKER)
            ]
            logger.info(
                f"Job {job_id} {job.status.stage}: {len(results[job_id])} results"
            )
        if pending:
            time.sleep(30)
    return results


def output_problem(output: dict | None) -> str | None:
    """Correctness only: a produced output must be non-empty and numerically sane."""
    if output is None:
        return None  # snippet only loads the model
    if output["type"] == "tensor" and not output["finite"]:
        return "output tensor has NaN/inf"
    if output["type"] == "list":
        return "empty output" if output["len"] == 0 else output_problem(output["first"])
    if output["type"] == "dict":
        return next(filter(None, map(output_problem, output["items"].values())), None)
    if output["type"] == "str" and not output["value"].strip():
        return "empty output"
    return None


def plan_batches(models, flavors, budget, config):
    """Groups models into (flavor, models, timeout_s) batches; cheapest batches get the budget first."""
    by_flavor, skipped = {}, {}
    for model in models:
        flavor = pick_flavor(
            model["weights_gb"], model["needs_fp8"], model["cpu_only"], flavors, config
        )
        if flavor is None:
            skipped[model["id"]] = "too large for available hardware"
            continue
        model["timeout_s"] = snippet_timeout_s(model["weights_gb"], config)
        by_flavor.setdefault(flavor.name, (flavor, []))[1].append(model)

    candidates = []
    for flavor, flavor_models in by_flavor.values():
        current, seconds = [], config.job_overhead_s
        for model in flavor_models:
            model_s = model["timeout_s"] * len(model["snippets"])
            if current and seconds + model_s > config.max_job_minutes * 60:
                candidates.append((flavor, current, seconds))
                current, seconds = [], config.job_overhead_s
            current.append(model)
            seconds += model_s
        candidates.append((flavor, current, seconds))

    batches = []
    for flavor, batch_models, seconds in sorted(
        candidates, key=lambda c: c[0].usd_per_minute * c[2]
    ):
        if budget.reserve(flavor.usd_per_minute * seconds / 60):
            batches.append((flavor, batch_models, seconds))
        else:
            skipped.update({m["id"]: "cost cap reached" for m in batch_models})
    return batches, skipped


def exception_line(stderr: str | None) -> str:
    lines = (stderr or "unknown error").strip().splitlines()
    return next(
        (line for line in reversed(lines) if EXCEPTION_LINE.match(line)), lines[-1]
    )


def summarise(model: dict, results: list[dict]) -> dict:
    """Timeouts and unfinished jobs are inconclusive, never a broken snippet."""
    problems, inconclusive = [], len(results) < len(model["snippets"])
    for r in sorted(results, key=lambda r: r["index"]):
        if r["exit_code"] is None:
            inconclusive = True
            problems.append(f"snippet {r['index']}: {r['error']}")
        elif r["exit_code"] != 0:
            problems.append(f"snippet {r['index']}: {exception_line(r['error'])}")
        elif problem := output_problem(r.get("output")):
            problems.append(f"snippet {r['index']}: {problem}")
    if len(results) < len(model["snippets"]):
        problems.append("job ended before all snippets ran")
    status = "failed" if problems else "ok"
    if inconclusive and not any(
        "timeout" not in p and "job ended" not in p for p in problems
    ):
        status = "inconclusive"
    return {
        "status": status,
        "problems": problems,
        "results": results,
    }


def run_snippets(
    huggingface_api: HfApi,
    models: list[dict],
    budget: Budget,
    config: ExecuteCodeConfig,
) -> dict[str, dict]:
    """Plans, launches and waits for the batches, and returns a summary per model id."""
    batches, skipped = plan_batches(models, load_flavors(), budget, config)
    report = {
        mid: {"status": "skipped", "reason": reason} for mid, reason in skipped.items()
    }

    job_models = {}
    for flavor, batch_models, timeout_s in batches:
        job = launch_batch(huggingface_api, flavor, batch_models, timeout_s, config)
        job_models[job.id] = (flavor, batch_models)
        logger.info(
            f"Launched {job.url} on {flavor.name} for {[m['id'] for m in batch_models]}"
        )

    for job_id, results in collect_results(
        huggingface_api, list(job_models), config
    ).items():
        flavor, batch_models = job_models[job_id]
        for model in batch_models:
            model_results = [r for r in results if r["id"] == model["id"]]
            report[model["id"]] = {
                "flavor": flavor.name,
                "job_id": job_id,
                **summarise(model, model_results),
            }
    return report
