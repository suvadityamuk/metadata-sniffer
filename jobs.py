"""
HF Jobs helpers shared by the snippet checks and the snippet repair loop:
flavor selection from the live hardware catalogue, a per-run cost budget, and
batched, secret-free snippet execution with results parsed from the job logs.
"""

import base64
import json
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


@dataclass
class Flavor:
    name: str
    gpu: str | None
    memory_gb: float  # total VRAM for GPU flavors, RAM for CPU flavors
    storage_gb: float
    usd_per_minute: float


@dataclass
class Budget:
    cap_usd: float
    reserved_usd: float = 0.0

    def reserve(self, usd: float) -> bool:
        if self.reserved_usd + usd > self.cap_usd:
            return False
        self.reserved_usd += usd
        return True


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
    weights_gb: float, needs_fp8: bool, flavors: list[Flavor], config: ExecuteCodeConfig
) -> Flavor | None:
    """Cheapest flavor whose memory and disk fit the model."""
    memory_needed = weights_gb * config.memory_overhead + config.memory_headroom_gb
    disk_needed = weights_gb * 1.5 + config.environment_disk_gb
    for flavor in flavors:
        if flavor.gpu is None and weights_gb > config.max_cpu_weights_gb:
            continue
        if needs_fp8 and flavor.gpu not in FP8_GPUS:
            continue
        if flavor.memory_gb >= memory_needed and flavor.storage_gb >= disk_needed:
            return flavor
    return None


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
