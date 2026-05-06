"""Tests for needle-in-a-haystack probe."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.moe.niah import (
    NEEDLE_TEMPLATE,
    build_haystack,
    grade_response,
    score_at_context_lengths,
)


def test_build_haystack_inserts_needle_at_depth() -> None:
    needle = "The magic number is 7421."
    haystack, needle_position = build_haystack(
        needle=needle,
        target_tokens=1000,
        depth_fraction=0.5,
        filler_text="The quick brown fox jumps over the lazy dog. ",
        approx_chars_per_token=4,
    )

    # Needle present
    assert needle in haystack
    # Inserted near the middle
    assert 0.3 < needle_position < 0.7
    # Roughly the right size (within 30% of target)
    estimated_tokens = len(haystack) / 4
    assert 700 < estimated_tokens < 1300


def test_grade_response_correct_extraction() -> None:
    needle = "The magic number is 7421."
    response = "Based on the text, the magic number is 7421."

    assert grade_response(response, needle, expected_value="7421") == 1.0


def test_grade_response_wrong_value() -> None:
    needle = "The magic number is 7421."
    response = "The magic number is 9999."

    assert grade_response(response, needle, expected_value="7421") == 0.0


def test_grade_response_partial_credit() -> None:
    # Response references the needle indirectly
    response = "It looks like 7421 is mentioned somewhere."

    assert grade_response(response, needle="x", expected_value="7421") == 1.0
