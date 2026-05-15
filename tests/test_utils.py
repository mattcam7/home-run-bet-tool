import pytest
from agents.utils import american_to_decimal


def test_positive_odds():
    assert american_to_decimal(450) == 5.5


def test_negative_odds():
    assert american_to_decimal(-110) == pytest.approx(1 + 100/110, abs=1e-6)


def test_plus_100():
    assert american_to_decimal(100) == 2.0


def test_minus_100():
    assert american_to_decimal(-100) == 2.0
