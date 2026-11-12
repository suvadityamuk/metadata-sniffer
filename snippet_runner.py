"""
Runs inside an HF Job (no secrets are passed to it). Executes every snippet of
every model in the `SNIPPET_BATCH` env var in its own subprocess, so one crash
or OOM does not take down the batch, and prints one result line per snippet.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time

RESULT_MARKER = "SNIPPET_RESULT "
# prebuilt CPU wheels, so llama-cpp-python does not have to be compiled in the job
EXTRA_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
OUTPUT_MARKER = "SNIPPET_OUTPUT "

# Appended to each snippet: summarises `_result` (the snippet's trailing expression).
DESCRIBE_RESULT = f"""
def _describe(value):
    import json
    if value is None:
        return None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        import torch
        tensor = torch.as_tensor(value).float()
        return {{"type": "tensor", "shape": list(tensor.shape), "finite": bool(tensor.isfinite().all())}}
    if type(value).__module__.startswith("PIL"):
        return {{"type": "image", "size": list(value.size)}}
    if isinstance(value, (list, tuple)):
        return {{"type": "list", "len": len(value), "first": _describe(value[0]) if value else None}}
    if isinstance(value, dict):
        return {{"type": "dict", "items": {{str(k): _describe(v) for k, v in list(value.items())[:10]}}}}
    if isinstance(value, (str, int, float, bool)):
        return {{"type": type(value).__name__, "value": value if not isinstance(value, str) else value[:500]}}
    return {{"type": type(value).__name__, "repr": repr(value)[:500]}}

print({OUTPUT_MARKER!r} + __import__("json").dumps(_describe(globals().get("_result"))), flush=True)
"""


def run_snippet(code: str, timeout_s: int) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code + DESCRIBE_RESULT],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "error": f"timeout after {timeout_s}s"}
    output = None
    for line in proc.stdout.splitlines():
        if line.startswith(OUTPUT_MARKER):
            output = json.loads(line[len(OUTPUT_MARKER) :])
    return {
        "exit_code": proc.returncode,
        "output": output,
        "error": proc.stderr[-3000:] if proc.returncode else None,
        "seconds": round(time.time() - start),
    }


if __name__ == "__main__":
    batch = json.loads(base64.b64decode(os.environ["SNIPPET_BATCH"]))
    for model in batch:
        if model.get("dependencies"):
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--extra-index-url",
                    EXTRA_INDEX_URL,
                    *model["dependencies"],
                ],
                check=False,
            )
        for index, code in enumerate(model["snippets"]):
            result = {"id": model["id"], "index": index}
            result.update(run_snippet(code, model["timeout_s"]))
            print(RESULT_MARKER + json.dumps(result), flush=True)
        # keep disk usage bounded: models in a batch are downloaded one at a time
        shutil.rmtree(os.environ.get("HF_HOME", "/tmp/hf"), ignore_errors=True)
