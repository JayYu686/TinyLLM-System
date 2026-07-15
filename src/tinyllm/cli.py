"""Typer command-line interface for TinyLLM-System."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import click
import typer
from typer.main import get_command

from tinyllm import __version__
from tinyllm.data import (
    COMMITPACKFT_LICENSE_ALLOWLIST,
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    CommitPackFTImportConfig,
    DataAcquisitionError,
    DataProcessingError,
    DatasetRegistryError,
    DatasetRegistryErrorCode,
    OASST1ImportConfig,
    PackingError,
    TokenizerContractError,
    open_registered_dataset,
    prepare_m2_dataset,
    summarize_registered_dataset,
)
from tinyllm.doctor.collector import DoctorCollector
from tinyllm.doctor.render import render_json, render_text
from tinyllm.evaluation import (
    BaselineContractError,
    BaselinePreflightError,
    BaselineRuntimeError,
    EvaluationContractError,
    complete_baseline_human_review,
    preflight_baseline_gpu,
    run_baseline_evaluation,
    run_contamination_check,
)
from tinyllm.schemas.artifacts import DEFAULT_ARTIFACT_ROOT
from tinyllm.training import (
    CheckpointError,
    TrainingConfigError,
    TrainingError,
    TrainingErrorCode,
    run_single_device_training,
)

app = typer.Typer(
    name="tinyllm",
    help="Hardware-aware LLM lifecycle tooling for consumer multi-GPU systems.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
data_app = typer.Typer(
    name="data",
    help="Inspect and build versioned dataset artifacts.",
    no_args_is_help=True,
)
eval_app = typer.Typer(
    name="eval",
    help="Build and run versioned model-quality evaluation contracts.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")


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


def _output_error(
    message: str,
    *,
    json_output: bool,
    error_code: str = "CLI_OUTPUT_ERROR",
) -> None:
    if json_output:
        payload = {"status": "error", "error": {"code": error_code, "message": message}}
        typer.echo(json.dumps(payload, sort_keys=True), err=True)
    else:
        typer.echo(f"error: {message}", err=True)


@data_app.command("inspect")
def data_inspect(
    ctx: typer.Context,
    source: Annotated[
        str,
        typer.Option("--source", help="Pinned source to inspect: all, oasst1, or commitpackft."),
    ] = "all",
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
    dataset_version: Annotated[
        str | None,
        typer.Option(
            "--dataset-version",
            help="Verify and inspect one committed m2-sft Dataset Version.",
        ),
    ] = None,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Private TinyLLM Artifact Root."),
    ] = DEFAULT_ARTIFACT_ROOT,
) -> None:
    """Show the pinned source contract or verify one committed Dataset Version."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if dataset_version is not None:
        if source != "all":
            _output_error(
                "--source cannot be combined with --dataset-version",
                json_output=json_output,
            )
            raise typer.Exit(code=2)
        try:
            dataset = open_registered_dataset(
                artifact_root=artifact_root,
                dataset_version=dataset_version,
            )
            summary = summarize_registered_dataset(
                dataset,
                operation="inspect",
                created=None,
            )
        except DatasetRegistryError as exc:
            _output_error(
                str(exc),
                json_output=json_output,
                error_code=str(exc.code),
            )
            code = 2 if exc.code == DatasetRegistryErrorCode.INVALID_INPUT else 3
            raise typer.Exit(code=code) from exc
        if json_output:
            typer.echo(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"verified: {summary.dataset_version} packs={summary.packed_sequences} "
                f"tokens={summary.total_tokens}"
            )
        return
    if source not in {"all", "oasst1", "commitpackft"}:
        _output_error("data source must be all, oasst1, or commitpackft", json_output=json_output)
        raise typer.Exit(code=2)

    records: list[dict[str, object]] = []
    if source in {"all", "oasst1"}:
        records.append(
            {
                "source": OASST1_SOURCE.to_dict(),
                "import_config": OASST1ImportConfig().to_dict(),
            }
        )
    if source in {"all", "commitpackft"}:
        records.append(
            {
                "source": COMMITPACKFT_SOURCE.to_dict(),
                "import_config": CommitPackFTImportConfig().to_dict(),
                "source_license_allowlist": sorted(COMMITPACKFT_LICENSE_ALLOWLIST),
            }
        )

    if json_output:
        typer.echo(
            json.dumps(
                {"status": "ok", "stage": "import_contract", "sources": records},
                indent=2,
                sort_keys=True,
            )
        )
        return
    for record in records:
        descriptor = cast(dict[str, object], record["source"])
        typer.echo(
            f"{descriptor['name']}: {descriptor['dataset_id']}@{descriptor['revision']} "
            f"license={descriptor['dataset_card_license']}"
        )


@data_app.command("prepare")
def data_prepare(
    ctx: typer.Context,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Private TinyLLM Artifact Root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    processing_config: Annotated[
        Path,
        typer.Option("--processing-config", help="Strict M2.2 processing YAML."),
    ] = Path("configs/data/m2_processing.yaml"),
    tokenization_config: Annotated[
        Path,
        typer.Option("--tokenization-config", help="Strict M2.3a Tokenizer YAML."),
    ] = Path("configs/data/m2_tokenization.yaml"),
    packing_config: Annotated[
        Path,
        typer.Option("--packing-config", help="Strict M2.3b Packing YAML."),
    ] = Path("configs/data/m2_packing.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Git project root for build lineage."),
    ] = Path("."),
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Refuse network access and require verified cache hits."),
    ] = False,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Build and atomically register the fixed M2 dataset from formal YAML."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not artifact_root.is_absolute():
        _output_error("Artifact Root must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    if not project_root.is_dir():
        _output_error("project root does not exist", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        summary = prepare_m2_dataset(
            project_root=project_root.resolve(),
            artifact_root=artifact_root,
            processing_config_path=processing_config,
            tokenization_config_path=tokenization_config,
            packing_config_path=packing_config,
            offline=offline,
        )
    except (DataProcessingError, PackingError, TokenizerContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="DATA_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except DataAcquisitionError as exc:
        _output_error(str(exc), json_output=json_output, error_code="DATA_ACQUISITION_ERROR")
        raise typer.Exit(code=3) from exc
    except DatasetRegistryError as exc:
        _output_error(str(exc), json_output=json_output, error_code=str(exc.code))
        code = 2 if exc.code == DatasetRegistryErrorCode.INVALID_INPUT else 3
        raise typer.Exit(code=code) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="DATA_PREPARE_FAILED")
        raise typer.Exit(code=3) from exc

    if json_output:
        typer.echo(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        action = "registered" if summary.created else "verified-existing"
        typer.echo(
            f"{action}: {summary.dataset_version} packs={summary.packed_sequences} "
            f"tokens={summary.total_tokens}"
        )


@eval_app.command("contamination")
def eval_contamination(
    ctx: typer.Context,
    evaluation_set: Annotated[
        Path,
        typer.Option("--evaluation-set", help="Strict public evaluation JSONL."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", help="Strict evaluation-set build YAML."),
    ],
    dataset_version: Annotated[
        str,
        typer.Option("--dataset-version", help="Committed m2-sft Dataset Version."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Private TinyLLM Artifact Root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    tokenization_config: Annotated[
        Path,
        typer.Option("--tokenization-config", help="Pinned M2 Tokenization YAML."),
    ] = Path("configs/data/m2_tokenization.yaml"),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Check frozen evaluation items against verified Train Token fingerprints."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not artifact_root.is_absolute():
        _output_error("Artifact Root must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        report = run_contamination_check(
            artifact_root=artifact_root,
            dataset_version=dataset_version,
            evaluation_set_path=evaluation_set,
            evaluation_config_path=config,
            tokenization_config_path=tokenization_config,
        )
    except EvaluationContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="EVALUATION_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except TokenizerContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="TOKENIZER_CONTRACT_ERROR")
        raise typer.Exit(code=2) from exc
    except DataAcquisitionError as exc:
        _output_error(str(exc), json_output=json_output, error_code="DATA_ACQUISITION_ERROR")
        raise typer.Exit(code=3) from exc
    except DatasetRegistryError as exc:
        _output_error(str(exc), json_output=json_output, error_code=str(exc.code))
        code = 2 if exc.code == DatasetRegistryErrorCode.INVALID_INPUT else 3
        raise typer.Exit(code=code) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="EVALUATION_FAILED")
        raise typer.Exit(code=3) from exc

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{report.status}: {report.evaluation_suite_version} "
            f"contaminated_items={report.contaminated_items}"
        )
    if report.status == "contaminated":
        raise typer.Exit(code=6)


@eval_app.command("baseline")
def eval_baseline(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Strict M2.4c formal or Smoke Baseline YAML."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Private TinyLLM Artifact Root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Git project root used for evaluation lineage."),
    ] = Path("."),
    device: Annotated[
        str,
        typer.Option("--device", help="Evaluation device: cuda or cpu."),
    ] = "cuda",
    gpu_index: Annotated[
        int | None,
        typer.Option("--gpu-index", help="Physical GPU index selected after busy/heat preflight."),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option("--offline/--online", help="Require verified local model and dataset caches."),
    ] = True,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Run the frozen pre-training model Baseline into a private traceable Run."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if device not in {"cpu", "cuda"}:
        _output_error("device must be cuda or cpu", json_output=json_output)
        raise typer.Exit(code=2)
    if not artifact_root.is_absolute():
        _output_error("Artifact Root must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    if device == "cuda" and gpu_index is None:
        _output_error("CUDA Baseline requires --gpu-index", json_output=json_output)
        raise typer.Exit(code=2)
    if device == "cpu" and gpu_index is not None:
        _output_error("--gpu-index is valid only with CUDA", json_output=json_output)
        raise typer.Exit(code=2)

    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if gpu_index is not None:
            preflight_baseline_gpu(gpu_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = run_baseline_evaluation(
            config_path=config,
            project_root=project_root,
            artifact_root=artifact_root,
            device=cast(Literal["cpu", "cuda"], device),
            offline=offline,
        )
    except BaselineContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except BaselinePreflightError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_PREFLIGHT_FAILED")
        raise typer.Exit(code=3) from exc
    except BaselineRuntimeError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_EVALUATION_FAILED")
        raise typer.Exit(code=6) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_EVALUATION_FAILED")
        raise typer.Exit(code=6) from exc
    finally:
        if gpu_index is not None:
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible

    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{result.status}: {result.run_id} "
            f"domain={result.domain.evaluated_items} general_tasks={len(result.general.tasks)}"
        )


@eval_app.command("baseline-review")
def eval_baseline_review(
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Awaiting Baseline Run ID."),
    ],
    judgments: Annotated[
        Path,
        typer.Option("--judgments", help="Private strict human-rubric Judgment JSONL."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Private TinyLLM Artifact Root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Commit all maintainer rubric Judgments and finalize an awaiting Baseline."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not artifact_root.is_absolute():
        _output_error("Artifact Root must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result = complete_baseline_human_review(
            run_id=run_id,
            artifact_root=artifact_root,
            judgments_path=judgments,
        )
    except BaselineContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_REVIEW_ERROR")
        raise typer.Exit(code=2) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="BASELINE_REVIEW_FAILED")
        raise typer.Exit(code=6) from exc
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{result.status}: {result.run_id} "
            f"reviewed={result.domain.human_reviewed} passed={result.domain.human_passed}"
        )


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


@app.command("train")
def train_command(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Validated M1 YAML training configuration."),
    ],
    device: Annotated[
        str,
        typer.Option("--device", help="Runtime device override: auto, cpu, or cuda."),
    ] = "auto",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Artifact root override for a new Run."),
    ] = None,
    resume_run: Annotated[
        Path | None,
        typer.Option("--resume-run", help="Existing Run directory used as restore source."),
    ] = None,
    resume_mode: Annotated[
        str,
        typer.Option("--resume-mode", help="Restore policy: exact, warm, or transfer."),
    ] = "exact",
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Run native single-device training from a strict YAML configuration."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if device not in {"auto", "cpu", "cuda"}:
        _output_error("device must be auto, cpu, or cuda", json_output=json_output)
        raise typer.Exit(code=2)
    if resume_mode not in {"exact", "warm", "transfer"}:
        _output_error("resume mode must be exact, warm, or transfer", json_output=json_output)
        raise typer.Exit(code=2)
    if resume_run is not None and not resume_run.is_dir():
        _output_error("resume Run directory does not exist", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result = run_single_device_training(
            config_path=config,
            output_root=output,
            device=cast(Literal["auto", "cpu", "cuda"], device),
            resume_run=resume_run,
            resume_mode=cast(Literal["exact", "warm", "transfer"], resume_mode),
        )
    except TrainingConfigError as exc:
        _output_error(str(exc), json_output=json_output)
        raise typer.Exit(code=2) from exc
    except TrainingError as exc:
        _output_error(f"{exc.code}: {exc}", json_output=json_output)
        preflight_codes = {
            TrainingErrorCode.ACCELERATOR_UNAVAILABLE,
            TrainingErrorCode.DISTRIBUTED_LAUNCH_REQUIRED,
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            TrainingErrorCode.UNSUPPORTED_PRECISION,
        }
        raise typer.Exit(code=3 if exc.code in preflight_codes else 4) from exc
    except CheckpointError as exc:
        _output_error(f"{exc.code}: {exc}", json_output=json_output)
        raise typer.Exit(code=5) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output)
        raise typer.Exit(code=4) from exc

    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{result.status}: {result.run_id} step={result.global_step} "
            f"checkpoint={result.checkpoint_id}"
        )
    if result.status == "terminated":
        raise typer.Exit(code=143)


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
