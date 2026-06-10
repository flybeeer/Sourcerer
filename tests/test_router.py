"""Tests for the heuristic router (pure logic, no I/O)."""

from sourcerer.routing.router import HeuristicRouter

_R = HeuristicRouter(local_model="local-llm", api_model="api-llm", hard_threshold=0.7)


def test_simple_lookup_goes_local():
    d = _R.decide("What is the company's PTO policy?")
    assert d.route == "local"
    assert d.model == "local-llm"
    assert 0.0 <= d.difficulty <= 1.0


def test_complex_reasoning_goes_api():
    d = _R.decide(
        "Compare the trade-offs between the on-prem and cloud deployment options "
        "and explain which one better fits our compliance requirements and why."
    )
    assert d.route == "api"
    assert d.model == "api-llm"
    assert d.difficulty >= 0.7


def test_sensitive_query_stays_local():
    d = _R.decide("What is the employee salary band for a senior engineer?")
    assert d.route == "local"
    assert "salary" in d.reason


def test_privacy_overrides_difficulty():
    # Hard *and* sensitive: privacy must win and keep it local.
    d = _R.decide(
        "Analyze and compare the salary and compensation implications of the "
        "confidential layoff plan and explain the trade-offs in detail."
    )
    assert d.route == "local"
    assert d.signals["sensitive_hits"]  # tripped a sensitivity rule
    assert d.difficulty >= 0.7  # would otherwise have routed to API


def test_reason_and_signals_are_populated():
    d = _R.decide("Why does the cache invalidate, and how do I fix it?")
    assert d.reason
    assert "reasoning_hits" in d.signals
    assert isinstance(d.signals["n_words"], int)


def test_threshold_is_respected():
    strict = HeuristicRouter("local-llm", "api-llm", hard_threshold=0.99)
    # Same query that routes to API at 0.70 should fall back to local at 0.99.
    d = strict.decide("Why did latency drop and how does the new flow compare?")
    assert d.route == "local"


def test_difficulty_always_in_range():
    for q in ["", "hi", "Why " * 50, "What is X? " * 10]:
        d = _R.decide(q)
        assert 0.0 <= d.difficulty <= 1.0


def test_thai_complex_reasoning_goes_api():
    # "Compare and explain why Sev-1 differs from Sev-3" in Thai → hard.
    d = _R.decide("เปรียบเทียบและอธิบายว่าทำไม Sev-1 ต่างจาก Sev-3")
    assert d.route == "api"
    assert d.difficulty >= 0.7


def test_thai_sensitive_stays_local():
    # "What is the employee salary?" in Thai → privacy override.
    d = _R.decide("เงินเดือนพนักงานเท่าไหร่")
    assert d.route == "local"
    assert d.signals["sensitive_hits"]


def test_thai_word_count_is_reasonable():
    # Thai has no spaces; the char-based estimate should still count words.
    d = _R.decide("จำนวนวันลาขึ้นอยู่กับอะไร")
    assert d.signals["n_words"] >= 4
