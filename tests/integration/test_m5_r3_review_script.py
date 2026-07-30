from __future__ import annotations

from scripts.finalize_m5_r3_p2_content_review import build_parser


def test_m5_r3_content_review_cli_confirmation_defaults_closed() -> None:
    args = build_parser().parse_args(
        [
            "--private-raw",
            "/private/raw.json",
            "--judgments",
            "/private/judgments.jsonl",
            "--output",
            "reports/m5/raw/review.json",
        ]
    )

    assert args.maintainer_confirmed is False
    assert str(args.public_result) == "reports/m5/raw/m5_r3_p2.json"
