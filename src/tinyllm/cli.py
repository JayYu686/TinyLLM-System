"""Command-line interface for TinyLLM-System."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tinyllm import __version__
from tinyllm.doctor.collector import DoctorCollector
from tinyllm.doctor.render import render_json, render_text


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="tinyllm",
        description="Hardware-aware LLM lifecycle tooling for consumer multi-GPU systems.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="global_json",
        help="Emit stable machine-readable JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser(
        "doctor", help="Inspect the local host without modifying system state."
    )
    doctor.add_argument(
        "--json", action="store_true", dest="command_json", help="Emit stable JSON."
    )
    doctor.add_argument(
        "--distributed",
        action="store_true",
        help="Include NUMA, GPU topology, P2P, NVLink, and NCCL tool checks.",
    )
    doctor.add_argument(
        "--output",
        type=Path,
        help="Write the same rendered report to an existing directory.",
    )
    doctor.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used for Git and storage checks (default: current directory).",
    )
    return parser


def _output_error(message: str, *, json_output: bool) -> None:
    if json_output:
        payload = {"status": "error", "error": {"code": "CLI_OUTPUT_ERROR", "message": message}}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)


def _run_doctor(args: argparse.Namespace) -> int:
    json_output = bool(args.global_json or args.command_json)
    project_root: Path = args.project_root
    if not project_root.is_dir():
        _output_error(f"project root does not exist: {project_root}", json_output=json_output)
        return 2
    report = DoctorCollector(project_root).collect(distributed=bool(args.distributed))
    rendered = render_json(report) if json_output else render_text(report)
    output: Path | None = args.output
    if output is not None:
        if not output.parent.is_dir():
            _output_error(
                f"output parent directory does not exist: {output.parent}",
                json_output=json_output,
            )
            return 2
        if output.exists() and output.is_dir():
            _output_error(f"output path is a directory: {output}", json_output=json_output)
            return 2
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 3 if report.status == "fail" else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TinyLLM-System CLI and return a stable exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _run_doctor(args)
    parser.error(f"unsupported command: {args.command}")
    return 2
