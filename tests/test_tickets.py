from repopilot.tickets import classify_priority, build_basic_issue


def test_critical_payment_bug():
    result = classify_priority("Payment fails for all users in production with 500 error")
    assert result.priority == "Critical"


def test_low_cosmetic_bug():
    result = classify_priority("Button color and spacing are slightly wrong")
    assert result.priority == "Low"


def test_issue_template_contains_sections():
    issue = build_basic_issue("App crashes after login")
    assert "Steps to reproduce" in issue
    assert "Suggested labels" in issue
