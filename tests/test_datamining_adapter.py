from agent_mvp.datamining_adapter import datamining_backend_status, mine_with_fpgrowth


def test_backend_status_reports_fpgrowth_engine(monkeypatch) -> None:
    monkeypatch.delenv("DATAMINING_REPO_PATH", raising=False)

    status = datamining_backend_status()

    assert status["engine"] in {"datamining_fpgrowth", "mlxtend_fpgrowth", "local_fpgrowth_fallback"}
    assert status["summary"]


def test_mine_with_fpgrowth_uses_fpgrowth_engine(monkeypatch) -> None:
    monkeypatch.delenv("DATAMINING_REPO_PATH", raising=False)
    transactions = [
        {"a=1", "b=1"},
        {"a=1", "b=1"},
        {"a=1", "c=1"},
        {"b=1", "c=1"},
    ]

    result = mine_with_fpgrowth(
        transactions,
        min_support=0.25,
        min_confidence=0.5,
        min_lift=0.8,
        max_len=3,
        max_antecedent=2,
        max_rules=10,
        min_count=2,
    )

    assert "fpgrowth" in result["engine"]
    assert result["rules"]
    assert result["frequent_itemsets"] > 0
    assert all(rule["matched_rows"] >= 2 for rule in result["rules"])


def test_mine_with_fpgrowth_filters_rules_below_min_count(monkeypatch) -> None:
    monkeypatch.delenv("DATAMINING_REPO_PATH", raising=False)
    transactions = [
        {"a=1", "b=1"},
        {"a=1", "b=1"},
        {"a=1", "c=1"},
        {"rare=1", "only_once=1"},
    ]

    result = mine_with_fpgrowth(
        transactions,
        min_support=0.25,
        min_confidence=0.5,
        min_lift=0.8,
        max_len=3,
        max_antecedent=2,
        max_rules=20,
        min_count=2,
    )

    assert result["parameters"]["min_count"] == 2
    assert result["parameters"]["effective_min_support"] >= 0.5
    assert result["warnings"]
    assert all(rule["matched_rows"] >= 2 for rule in result["rules"])


def test_mine_with_fpgrowth_handles_min_count_larger_than_sample(monkeypatch) -> None:
    monkeypatch.delenv("DATAMINING_REPO_PATH", raising=False)
    transactions = [
        {"a=1", "b=1"},
        {"a=1", "b=1"},
        {"a=1", "c=1"},
        {"b=1", "c=1"},
    ]

    result = mine_with_fpgrowth(
        transactions,
        min_support=0.2,
        min_confidence=0.7,
        min_lift=1.2,
        max_len=3,
        max_antecedent=2,
        max_rules=20,
        min_count=10,
    )

    assert result["parameters"]["min_count"] == 10
    assert result["parameters"]["effective_min_support"] == 1.0
    assert result["rules"] == []
    assert "小样本不足" in " ".join(result["warnings"])
