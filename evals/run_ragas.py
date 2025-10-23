"""RAGAS evaluation runner for the RAG API.

This script:
- Waits for the API to be healthy
- Loads a JSONL dataset of {question, ground_truth}
- Calls the /ask endpoint for each question and collects contexts (citations)
- Computes RAGAS metrics (faithfulness, answer relevancy, context precision/recall)
- Gates on configured thresholds and exits with a non-zero code if failing

Configuration is read from app.config.settings.* (EVAL_ variables).
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from datasets import Dataset as HFDataset

# RAGAS v0.1.x API
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

from app.config import settings


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    Ignores empty lines and lines that fail to parse.

    Args:
        path: Path to the .jsonl dataset.

    Returns:
        List[Dict[str, Any]]: Parsed rows.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def wait_for_health(base_url: str, timeout_s: int = 120) -> None:
    """Block until the API health endpoint returns OK or timeout occurs.

    Args:
        base_url: Base URL of the API (without trailing slash).
        timeout_s: Maximum seconds to wait.

    Raises:
        RuntimeError: If health does not pass within the timeout.
    """
    t0 = time.time()
    url = base_url.rstrip("/") + "/health"
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Health check failed for {url} after {timeout_s}s")


def query_api(question: str, base_url: str) -> Dict[str, Any]:
    """Query the /ask endpoint and return the parsed JSON response.

    Args:
        question: The question to send to the API.
        base_url: API base URL.

    Returns:
        Dict[str, Any]: Parsed JSON response.

    Raises:
        RuntimeError: If the request fails.
    """
    url = base_url.rstrip("/") + "/ask"
    r = requests.post(url, json={"question": question}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"/ask failed {r.status_code}: {r.text}")
    return r.json()


def main() -> int:
    """Run the evaluation workflow end-to-end.

    Returns:
        int: Exit code (0=success, 1=gate failed, 2=error).
    """
    # Ensure API is up
    base = settings.EVAL_API_BASE_URL
    print(f"[EVAL] Using API base: {base}", flush=True)
    try:
        wait_for_health(base)
    except Exception as e:
        print(f"[EVAL][ERROR] {e}", flush=True)
        return 2

    # Load dataset
    ds_path = Path(settings.EVAL_DATASET_PATH)
    if not ds_path.exists():
        print(f"[EVAL][ERROR] dataset not found: {ds_path}", flush=True)
        return 2
    rows = load_jsonl(ds_path)
    if not rows:
        print(f"[EVAL][ERROR] empty dataset: {ds_path}", flush=True)
        return 2

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    print(f"[EVAL] Running {len(rows)} queries...", flush=True)
    for i, row in enumerate(rows, start=1):
        q = str(row.get("question", "")).strip()
        gt = str(row.get("ground_truth", "")).strip()
        if not q:
            continue

        try:
            resp = query_api(q, base)
        except Exception as e:
            print(f"[EVAL][WARN] API error for item {i}: {e}", flush=True)
            continue

        ans = str(resp.get("answer", "")).strip()
        cits = resp.get("citations", []) or []
        ctx_snippets = []
        for c in cits:
            snip = str(c.get("snippet") or "").strip()
            if snip:
                ctx_snippets.append(snip)

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_snippets)
        ground_truths.append(gt)

        if i % 5 == 0 or i == len(rows):
            print(f"[EVAL] Progress: {i}/{len(rows)}", flush=True)

    # Build HF dataset for RAGAS
    hf = HFDataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Evaluate
    print("[EVAL] Computing RAGAS metrics...", flush=True)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    try:
        result = evaluate(hf, metrics=metrics)
    except Exception as e:
        print(f"[EVAL][ERROR] ragas.evaluate failed: {e}", flush=True)
        return 2

    # Extract metric means
    scores = {m.name: float(result[m.name]) for m in metrics if m.name in result}
    print("[EVAL] Scores:")
    for k, v in scores.items():
        print(f"  - {k}: {v:.3f}")

    # Gate thresholds
    min_faith = settings.EVAL_MIN_FAITHFULNESS
    min_ans_rel = settings.EVAL_MIN_ANSWER_RELEVANCY
    min_ctx_prec = settings.EVAL_MIN_CONTEXT_PRECISION
    min_ctx_rec = settings.EVAL_MIN_CONTEXT_RECALL

    ok = True
    if scores.get("faithfulness", 1.0) < min_faith:
        ok = False
    if scores.get("answer_relevancy", 1.0) < min_ans_rel:
        ok = False
    if scores.get("context_precision", 1.0) < min_ctx_prec:
        ok = False
    if scores.get("context_recall", 1.0) < min_ctx_rec:
        ok = False

    if not ok:
        print(
            "[EVAL][GATE] FAILED thresholds: "
            f"faithfulness>={min_faith}, answer_relevancy>={min_ans_rel}, "
            f"context_precision>={min_ctx_prec}, context_recall>={min_ctx_rec}",
            flush=True,
        )
        return 1

    print("[EVAL][GATE] PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
