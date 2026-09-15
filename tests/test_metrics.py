"""precision@k / recall@k tests: pure math, no OpenAI key, no network."""

import pytest

from ragpkt.metrics import average_scores, precision_recall_at_k


def test_perfect_retrieval():
    score = precision_recall_at_k(["a", "b", "c"], ["a"])
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == 1.0
    assert score.hit is True


def test_complete_miss():
    score = precision_recall_at_k(["x", "y", "z"], ["a"])
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.hit is False


def test_multiple_expected_ids_partial_recall():
    score = precision_recall_at_k(["a", "b"], ["a", "c"])
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.hit is True


def test_empty_expected_ids_raises():
    with pytest.raises(ValueError):
        precision_recall_at_k(["a"], [])


def test_empty_retrieved_ids_is_zero_not_a_crash():
    score = precision_recall_at_k([], ["a"])
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.hit is False


def test_average_scores_over_empty_list():
    avg = average_scores([])
    assert avg == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0}


def test_average_scores_mixes_hit_and_miss():
    scores = [
        precision_recall_at_k(["a"], ["a"]),   # hit, p=1, r=1
        precision_recall_at_k(["x"], ["a"]),   # miss, p=0, r=0
    ]
    avg = average_scores(scores)
    assert avg["hit_rate"] == pytest.approx(0.5)
    assert avg["precision"] == pytest.approx(0.5)
    assert avg["recall"] == pytest.approx(0.5)
