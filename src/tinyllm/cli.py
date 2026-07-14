"""Typer command-line interface for TinyLLM-System."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import click
import typer
from typer.main import get_command

from tinyllm import __version__
from tinyllm.doctor.collector import DoctorCollector
from tinyllm.doctor.render import render_json, render_text

app = typer.Typer(
    name="tinyllm",
    help="Hardware-aware LLM lifecycle tooling for consumer multi-GPU systems.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@dataclass(frozen=True, slots=True)
class CLIState:
    """Global CLI settings propagated to subcommands."""

    json_output: bool = False


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tinyllm {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed TinyLLM-System version.",
        ),
    ] = False,
) -> None:
    """Initialize global command context."""

    del version
    ctx.obj = CLIState(json_output=json_output)


def _output_error(message: str, *, json_output: bool) -> None:
    if json_output:
        payload = {"status": "error", "error": {"code": "CLI_OUTPUT_ERROR", "message": message}}
        typer.echo(json.dumps(payload, sort_keys=True), err=True)
    else:
        typer.echo(f"error: {message}", err=True)


@app.command()
def doctor(
    ctx: typer.Context,
    distributed: Annotated[
        bool,
        typer.Option(
            "--distributed",
            help="Include NUMA, GPU topology, P2P, NVLink, and NCCL tool checks.",
        ),
    ] = False,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the rendered report to this file."),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root used for Git and storage checks.",
        ),
    ] = None,
) -> None:
    """Inspect the local host without modifying system state."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    project_root = project_root or Path.cwd()
    if not project_root.is_dir():
        _output_error(f"project root does not exist: {project_root}", json_output=json_output)
        raise typer.Exit(code=2)

    report = DoctorCollector(project_root).collect(distributed=distributed)
    rendered = render_json(report) if json_output else render_text(report)
    if output is not None:
        if not output.parent.is_dir():
            _output_error(
                f"output parent directory does not exist: {output.parent}",
                json_output=json_output,
            )
            raise typer.Exit(code=2)
        if output.exists() and output.is_dir():
            _output_error(f"output path is a directory: {output}", json_output=json_output)
            raise typer.Exit(code=2)
        try:
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            _output_error(f"cannot write output: {exc}", json_output=json_output)
            raise typer.Exit(code=2) from exc
    typer.echo(rendered)
    if report.status == "fail":
        raise typer.Exit(code=3)


def build_parser() -> click.Command:
    """Return the Click command generated from the public Typer application."""

    return get_command(app)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TinyLLM-System CLI and return a stable exit code."""

    command = build_parser()
    try:
        result = command.main(
            args=list(argv) if argv is not None else None,
            prog_name="tinyllm",
            standalone_mode=False,
        )
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return exc.exit_code
    return int(result) if isinstance(result, int) else 0
