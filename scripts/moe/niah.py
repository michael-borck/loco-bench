"""Needle-in-a-haystack long-context retrieval probe.

Inserts a known fact into a long filler document at a controlled depth, asks
the model to retrieve it, grades whether the response contains the expected
value. Run at multiple target context lengths to test KV-cache fidelity.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


NEEDLE_TEMPLATE = "The magic number for project {project} is {value}."

QUESTION_TEMPLATE = (
    "What is the magic number for project {project}? "
    "Answer with just the number."
)


@dataclass
class NiahResult:
    target_tokens: int
    depth_fraction: float
    needle_value: str
    response: str
    score: float


def build_haystack(
    *,
    needle: str,
    target_tokens: int,
    depth_fraction: float,
    filler_text: str,
    approx_chars_per_token: int = 4,
) -> tuple[str, float]:
    """Construct a (haystack, needle_position) pair.

    `depth_fraction` is the target relative position (0=start, 1=end). The actual
    returned position is the fraction of `len(haystack)` where the needle sits.
    """
    target_chars = target_tokens * approx_chars_per_token
    repeat = max(1, target_chars // len(filler_text) + 1)
    body = filler_text * repeat
    body = body[:target_chars]

    insert_at = int(len(body) * depth_fraction)
    haystack = body[:insert_at] + needle + " " + body[insert_at:]
    actual_position = insert_at / len(haystack)
    return haystack, actual_position


def build_filler(seed: int = 0) -> str:
    """Deterministic filler text (no semantic content that could leak the needle)."""
    rng = random.Random(seed)
    sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "Pack my box with five dozen liquor jugs. ",
        "Sphinx of black quartz, judge my vow. ",
        "How vexingly quick daft zebras jump. ",
    ]
    rng.shuffle(sentences)
    return "".join(sentences)


def grade_response(response: str, needle: str, expected_value: str) -> float:
    """Score 1.0 if expected_value appears in response, 0.0 otherwise."""
    return 1.0 if expected_value in response else 0.0


def query_server(
    server_url: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0
) -> str:
    """Send a completion request to a llama.cpp OpenAI-compatible server."""
    r = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": "local",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def run_one_probe(
    *,
    server_url: str,
    target_tokens: int,
    depth_fraction: float,
    project: str,
    value: str,
    filler_text: str,
) -> NiahResult:
    needle = NEEDLE_TEMPLATE.format(project=project, value=value)
    haystack, _ = build_haystack(
        needle=needle,
        target_tokens=target_tokens,
        depth_fraction=depth_fraction,
        filler_text=filler_text,
    )
    question = QUESTION_TEMPLATE.format(project=project)
    prompt = haystack + "\n\n" + question + "\n"
    response = query_server(server_url, prompt)
    score = grade_response(response, needle, expected_value=value)
    return NiahResult(
        target_tokens=target_tokens,
        depth_fraction=depth_fraction,
        needle_value=value,
        response=response.strip(),
        score=score,
    )


def score_at_context_lengths(
    server_url: str,
    target_token_lengths: list[int] = [1000, 8000, 64000, 256000],
    depths: list[float] = [0.1, 0.5, 0.9],
) -> dict[str, Any]:
    """Run NIAH across target lengths × depths grid. Returns summary dict."""
    filler = build_filler()
    rng = random.Random(42)
    runs: list[NiahResult] = []
    for tok in target_token_lengths:
        for depth in depths:
            value = str(rng.randint(1000, 9999))
            project = f"alpha{tok}d{int(depth * 100)}"
            runs.append(
                run_one_probe(
                    server_url=server_url,
                    target_tokens=tok,
                    depth_fraction=depth,
                    project=project,
                    value=value,
                    filler_text=filler,
                )
            )

    by_length: dict[int, list[float]] = {}
    for r in runs:
        by_length.setdefault(r.target_tokens, []).append(r.score)

    return {
        "scores_by_context": {
            tok: sum(scores) / len(scores) for tok, scores in by_length.items()
        },
        "raw_runs": [asdict(r) for r in runs],
    }


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.write_text(json.dumps(results, indent=2))
