# tests/test_simulation.py
"""Unit tests for agents.simulation — name normalization and matching."""
import pytest
from agents.simulation import _normalize_name, _match_player


class TestNormalizeName:
    def test_lowercases_then_titlecases(self):
        assert _normalize_name("AARON JUDGE") == "Aaron Judge"

    def test_strips_jr_suffix(self):
        assert _normalize_name("Vladimir Guerrero Jr.") == "Vladimir Guerrero"

    def test_strips_iii_suffix(self):
        assert _normalize_name("Cal Ripken III") == "Cal Ripken"

    def test_strips_accents(self):
        assert _normalize_name("Yandy Díaz") == "Yandy Diaz"

    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_name("  Aaron Judge  ") == "Aaron Judge"

    def test_already_normal(self):
        assert _normalize_name("Aaron Judge") == "Aaron Judge"


class TestMatchPlayer:
    CANDIDATES = ["Aaron Judge", "Rafael Devers", "Shohei Ohtani"]

    def test_exact_match(self):
        assert _match_player("Aaron Judge", self.CANDIDATES) == "Aaron Judge"

    def test_fuzzy_match_typo(self):
        # "Aron Judge" is close enough
        result = _match_player("Aron Judge", self.CANDIDATES)
        assert result == "Aaron Judge"

    def test_no_match_returns_none(self):
        assert _match_player("Totally Unknown Player", self.CANDIDATES) is None

    def test_case_insensitive_exact(self):
        assert _match_player("aaron judge", self.CANDIDATES) == "Aaron Judge"
