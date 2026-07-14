from tinyllm.doctor.schema import CheckResult, aggregate_status


def test_required_failure_fails_report() -> None:
    checks = [CheckResult("required", "fail", "failed", required=True)]
    assert aggregate_status(checks) == "fail"


def test_optional_unavailable_warns() -> None:
    checks = [
        CheckResult("required", "pass", "passed", required=True),
        CheckResult("optional", "unavailable", "missing"),
    ]
    assert aggregate_status(checks) == "warn"
