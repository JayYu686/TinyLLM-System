"""Typer command-line interface for TinyLLM-System."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NoReturn, cast

import click
import typer
from typer.main import get_command

from tinyllm import __version__
from tinyllm.benchmark.config import BenchmarkProfile, DDPBenchmarkConfigError
from tinyllm.benchmark.inference import InferenceBenchmarkError, run_inference_benchmark
from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkConfigError,
    load_inference_benchmark_config,
)
from tinyllm.benchmark.schema import BenchmarkGroup
from tinyllm.benchmark.supervisor import (
    BenchmarkPreflightError,
    BenchmarkRunError,
    run_formal_benchmark,
)
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
from tinyllm.deployment import (
    DeploymentError,
    DeploymentErrorCode,
    promote_production,
    resolve_model,
    rollback_production,
    show_deployment,
)
from tinyllm.deployment.gate import assemble_m7_production_gate
from tinyllm.doctor.collector import DoctorCollector
from tinyllm.doctor.render import render_json, render_text
from tinyllm.evaluation import (
    BaselineContractError,
    BaselinePreflightError,
    BaselineRuntimeError,
    EvaluationContractError,
    M6AssemblyError,
    M6BaseImportError,
    M6CandidateImportError,
    M6ComparisonError,
    M6ContractError,
    M6DomainError,
    M6DomainPassSummary,
    M6GeneralError,
    M6ModelIdentity,
    M6PromotionError,
    assemble_m6_base_evaluation,
    assemble_m6_base_v2_evaluation,
    assemble_m6_candidate_evaluation,
    compare_m6_evaluations,
    complete_baseline_human_review,
    finalize_m6_domain_pass,
    import_m2_base_evidence,
    import_m5_candidate_evidence,
    load_m6_base_import,
    load_m6_candidate_import,
    load_m6_comparison,
    load_m6_evaluation,
    load_m6_release_config,
    preflight_baseline_gpu,
    promote_m6_candidate,
    run_baseline_evaluation,
    run_contamination_check,
    run_m6_domain_pass,
    run_m6_general_pass,
    write_m6_comparison,
)
from tinyllm.lineage import (
    DEFAULT_INDEX_RELATIVE_PATH,
    RunIndexError,
    RunIndexErrorCode,
    list_indexed_runs,
    rebuild_run_index,
    show_indexed_run,
)
from tinyllm.schemas import canonical_config_hash
from tinyllm.schemas.artifacts import DEFAULT_ARTIFACT_ROOT
from tinyllm.serving.config import ServingConfigError, load_gateway_config
from tinyllm.training import (
    CheckpointError,
    TrainingConfigError,
    TrainingError,
    TrainingErrorCode,
    run_single_device_training,
)
from tinyllm.training.smoke_preflight import parse_gpu_indices

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
benchmark_app = typer.Typer(
    name="benchmark",
    help="Run evidence-first training performance benchmarks.",
    no_args_is_help=True,
)
run_app = typer.Typer(
    name="run",
    help="Rebuild and query the local Run lineage index.",
    no_args_is_help=True,
)
deploy_app = typer.Typer(
    name="deploy",
    help="Resolve and atomically manage immutable model deployments.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(run_app, name="run")
app.add_typer(deploy_app, name="deploy")


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


def _run_index_error(exc: RunIndexError, *, json_output: bool) -> NoReturn:
    _output_error(str(exc), json_output=json_output, error_code=exc.code.value)
    usage_errors = {RunIndexErrorCode.INVALID_INPUT}
    raise typer.Exit(code=2 if exc.code in usage_errors else 3)


def _deployment_error(exc: DeploymentError, *, json_output: bool) -> NoReturn:
    """Map deployment failures onto the frozen CLI exit classes."""

    _output_error(str(exc), json_output=json_output, error_code=exc.code.value)
    if exc.code == DeploymentErrorCode.INVALID_INPUT:
        raise typer.Exit(code=2)
    if exc.code == DeploymentErrorCode.GATE_REJECTED:
        raise typer.Exit(code=6)
    raise typer.Exit(code=7)


@deploy_app.command("resolve")
def deploy_resolve(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", help="Production Alias or immutable M6/M7 model version."),
    ] = "production",
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Resolve one deployment and verify model and Tokenizer artifacts."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = resolve_model(artifact_root, model)
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"verified: {result.model_version} status={result.status} "
            f"model_sha256={result.model_artifact_sha256}"
        )


@deploy_app.command("show")
def deploy_show(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", help="Production Alias or immutable M6/M7 model version."),
    ] = "production",
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Show a verified path-free deployment projection."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = show_deployment(artifact_root, model)
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{result['status']}: {result['model_version']} "
            f"candidate={result['candidate_model_version']}"
        )


@deploy_app.command("promote")
def deploy_promote(
    ctx: typer.Context,
    gate: Annotated[
        Path,
        typer.Option("--gate", help="Absolute accepted M7 Production Gate JSON."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Publish an accepted M7 Production record and update its Alias."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = promote_production(artifact_root, gate)
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"Production: {result.production_version}")


@deploy_app.command("gate")
def deploy_gate(
    ctx: typer.Context,
    benchmark: Annotated[Path, typer.Option("--benchmark", help="M7 benchmark summary JSON.")],
    contract: Annotated[Path, typer.Option("--contract", help="M7 API contract evidence JSON.")],
    recovery: Annotated[Path, typer.Option("--recovery", help="M7 recovery evidence JSON.")],
    rollback: Annotated[Path, typer.Option("--rollback", help="M7 rollback evidence JSON.")],
    security: Annotated[Path, typer.Option("--security", help="M7 security audit JSON.")],
    m6_comparison: Annotated[
        Path, typer.Option("--m6-comparison", help="Accepted M6 comparison JSON.")
    ],
    m6_candidate_evaluation: Annotated[
        Path, typer.Option("--m6-candidate-evaluation", help="M6 Candidate evaluation JSON.")
    ],
    environment: Annotated[
        Path, typer.Option("--environment", help="Frozen serving environment evidence.")
    ],
    hardware: Annotated[Path, typer.Option("--hardware", help="Frozen serving hardware evidence.")],
    output: Annotated[Path, typer.Option("--output", help="New absolute M7 Gate JSON path.")],
    candidate: Annotated[
        str, typer.Option("--candidate", help="Immutable M6 Candidate version.")
    ] = "qwen3-0-6b-m6-d16c2357",
    serving_config: Annotated[
        Path, typer.Option("--serving-config", help="Validated M7 serving YAML.")
    ] = Path("configs/serving/m7_gateway.yaml"),
    benchmark_config: Annotated[
        Path, typer.Option("--benchmark-config", help="Frozen M7 benchmark YAML.")
    ] = Path("configs/benchmark/m7_inference.yaml"),
    benchmark_gateway_config: Annotated[
        Path,
        typer.Option(
            "--benchmark-gateway-config", help="Gateway YAML used for the formal benchmark."
        ),
    ] = Path("configs/serving/m7_gateway_benchmark.yaml"),
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", help="Absolute private Artifact Store root.")
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Recompute an M7 Production Gate solely from immutable evidence."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = assemble_m7_production_gate(
            artifact_root=artifact_root,
            candidate_model_version=candidate,
            benchmark_path=benchmark,
            contract_path=contract,
            recovery_path=recovery,
            rollback_path=rollback,
            security_path=security,
            m6_comparison_path=m6_comparison,
            m6_candidate_evaluation_path=m6_candidate_evaluation,
            serving_config_path=serving_config,
            benchmark_config_path=benchmark_config,
            benchmark_gateway_config_path=benchmark_gateway_config,
            environment_path=environment,
            hardware_path=hardware,
            output_path=output,
        )
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{result.status}: {result.gate_id} production_eligible="
            f"{str(result.production_eligible).lower()}"
        )


@deploy_app.command("rollback")
def deploy_rollback(
    ctx: typer.Context,
    target: Annotated[
        str | None,
        typer.Option("--target", help="Prior immutable M7 version; default is Alias history."),
    ] = None,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Atomically point the Production Alias at a prior accepted deployment."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = rollback_production(artifact_root, target)
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"Production: {result.production_version} previous={result.previous_production_version}"
        )


@run_app.command("rebuild")
def run_rebuild(
    ctx: typer.Context,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional absolute SQLite index output."),
    ] = None,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Atomically rebuild SQLite from every immutable ``runs/**/run.json``."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        result = rebuild_run_index(artifact_root, output_path=output)
    except RunIndexError as exc:
        _run_index_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"succeeded: indexed_runs={result.indexed_runs} "
            f"source_tree_sha256={result.source_tree_sha256}"
        )


@run_app.command("list")
def run_list(
    ctx: typer.Context,
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    index: Annotated[
        Path | None,
        typer.Option("--index", help="Optional absolute SQLite index path."),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Optional exact Run status filter."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=1000, help="Maximum Runs to return."),
    ] = 50,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """List the newest Runs from the rebuildable query index."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    index_path = index or artifact_root / DEFAULT_INDEX_RELATIVE_PATH
    try:
        result = list_indexed_runs(index_path, status=status, limit=limit)
    except RunIndexError as exc:
        _run_index_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        for entry in result.runs:
            typer.echo(f"{entry.run_id}\t{entry.status}\t{entry.strategy or '-'}")


@run_app.command("show")
def run_show(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Exact immutable Run ID.")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    index: Annotated[
        Path | None,
        typer.Option("--index", help="Optional absolute SQLite index path."),
    ] = None,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Show one Run projection without opening private raw artifacts."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    index_path = index or artifact_root / DEFAULT_INDEX_RELATIVE_PATH
    try:
        result = show_indexed_run(index_path, run_id)
    except RunIndexError as exc:
        _run_index_error(exc, json_output=json_output)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{result.run_id}\n"
            f"status: {result.status}\n"
            f"created_at: {result.created_at.isoformat()}\n"
            f"manifest: {result.manifest_relative_path}"
        )


@app.command("compare")
def compare_models(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ],
    baseline: Annotated[
        Path,
        typer.Option("--baseline", help="Complete private M6 Base evaluation JSON."),
    ],
    candidate: Annotated[
        Path,
        typer.Option("--candidate", help="Complete private M6 Candidate evaluation JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Absolute path for the atomic comparison JSON."),
    ],
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Compare one Base/Candidate pair and apply the M6 Candidate Gate."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not output.is_absolute():
        _output_error("M6 comparison output must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        release_config = load_m6_release_config(config)
        base_result, base_sha256 = load_m6_evaluation(baseline)
        candidate_result, candidate_sha256 = load_m6_evaluation(candidate)
        result = compare_m6_evaluations(
            release_config,
            base_result,
            candidate_result,
            base_evaluation_sha256=base_sha256,
            candidate_evaluation_sha256=candidate_sha256,
        )
        write_m6_comparison(output, result)
    except (M6ContractError, M6ComparisonError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_COMPARISON_INPUT_ERROR")
        raise typer.Exit(code=2) from exc
    except OSError as exc:
        _output_error(
            "cannot persist M6 comparison",
            json_output=json_output,
            error_code="M6_ARTIFACT_ERROR",
        )
        raise typer.Exit(code=3) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{result.status}: candidate={result.candidate_evaluation_id} "
            f"eligible={result.candidate_eligible}"
        )
    if not result.candidate_eligible:
        raise typer.Exit(code=6)


@app.command("promote")
def promote_model(
    ctx: typer.Context,
    comparison: Annotated[
        Path,
        typer.Option("--comparison", help="Accepted M6 comparison JSON."),
    ],
    registry_root: Annotated[
        Path,
        typer.Option("--registry-root", help="Absolute private model Registry root."),
    ],
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Atomically register an accepted M6 model as Candidate."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not registry_root.is_absolute():
        _output_error("M6 Registry root must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result, comparison_sha256 = load_m6_comparison(comparison)
        record = promote_m6_candidate(
            result,
            comparison_sha256=comparison_sha256,
            registry_root=registry_root,
        )
    except M6ContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_PROMOTION_INPUT_ERROR")
        raise typer.Exit(code=2) from exc
    except M6PromotionError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_PROMOTION_REJECTED")
        raise typer.Exit(code=6) from exc
    except OSError as exc:
        _output_error(
            "cannot access M6 Registry",
            json_output=json_output,
            error_code="M6_REGISTRY_ERROR",
        )
        raise typer.Exit(code=3) from exc
    if json_output:
        typer.echo(record.model_dump_json(indent=2))
    else:
        typer.echo(f"Candidate: {record.model_version}")


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


@eval_app.command("m6-import-base")
def eval_m6_import_base(
    ctx: typer.Context,
    source_run: Annotated[
        Path,
        typer.Option("--source-run", help="Absolute completed M2 formal Base Run."),
    ],
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="Absolute verified Qwen3 Base snapshot."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Absolute atomic M6 Base-import JSON."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Git project containing the frozen domain suite."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Verify and convert reusable M2 Base evidence into the M6 contract."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not source_run.is_absolute() or not model_dir.is_absolute() or not output.is_absolute():
        _output_error("M6 Base import paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result = import_m2_base_evidence(
            release_config_path=config,
            source_run=source_run,
            model_dir=model_dir,
            project_root=project_root,
            output_path=output,
        )
    except M6ContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except M6BaseImportError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_BASE_IMPORT_FAILED")
        raise typer.Exit(code=3) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        domain_evidence = (
            f"nonthinking={result.nonthinking.correct_items}/300"
            if result.nonthinking is not None
            else "domain=replay-required"
        )
        typer.echo(f"succeeded: source={result.source_run_id} {domain_evidence}")


@eval_app.command("m6-import-candidate")
def eval_m6_import_candidate(
    ctx: typer.Context,
    source_run: Annotated[
        Path,
        typer.Option("--source-run", help="Absolute completed M5 formal Full-SFT Run."),
    ],
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="Absolute frozen Candidate model export."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Absolute atomic M6 Candidate-import JSON."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Verify the frozen M5 10M snapshot selected as the M6 Candidate."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not source_run.is_absolute() or not model_dir.is_absolute() or not output.is_absolute():
        _output_error("M6 Candidate import paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result = import_m5_candidate_evidence(
            release_config_path=config,
            source_run=source_run,
            model_dir=model_dir,
            output_path=output,
        )
    except M6ContractError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except M6CandidateImportError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CANDIDATE_IMPORT_FAILED")
        raise typer.Exit(code=3) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"succeeded: source={result.source_run_id} "
            f"checkpoint={result.model.training_checkpoint_id}"
        )


def _validated_m6_domain_mode_and_paths(
    *,
    evidence_import: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    mode: str,
    json_output: bool,
) -> Literal["thinking", "nonthinking"]:
    if mode not in {"thinking", "nonthinking"}:
        _output_error("M6 mode must be thinking or nonthinking", json_output=json_output)
        raise typer.Exit(code=2)
    if not all(
        path.is_absolute() for path in (evidence_import, model_dir, tokenizer_dir, output_dir)
    ):
        _output_error("M6 domain artifact paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    return cast(Literal["thinking", "nonthinking"], mode)


def _run_m6_domain_cli(
    *,
    model_identity: M6ModelIdentity,
    expected_config_sha256: str,
    config: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    project_root: Path,
    gpu_index: int,
    mode: Literal["thinking", "nonthinking"],
    json_output: bool,
) -> M6DomainPassSummary:
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        preflight_baseline_gpu(gpu_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        return run_m6_domain_pass(
            release_config_path=config,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=output_dir,
            project_root=project_root,
            physical_gpu_index=gpu_index,
            model_identity=model_identity,
            mode=mode,
            expected_config_sha256=expected_config_sha256,
        )
    except BaselinePreflightError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_PREFLIGHT_FAILED")
        raise typer.Exit(code=3) from exc
    except M6DomainError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_DOMAIN_FAILED")
        raise typer.Exit(code=6) from exc
    finally:
        if previous_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible


def _render_m6_domain_result(result: M6DomainPassSummary, *, json_output: bool) -> None:
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{result.status}: {result.evaluation_id} "
            f"objective_correct={result.objective_correct_items}/260"
        )


@eval_app.command("m6-domain")
def eval_m6_domain(
    ctx: typer.Context,
    base_import: Annotated[
        Path,
        typer.Option("--base-import", help="Verified M6 Base-import JSON."),
    ],
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="Absolute model snapshot to evaluate."),
    ],
    tokenizer_dir: Annotated[
        Path,
        typer.Option("--tokenizer-dir", help="Absolute pinned tokenizer snapshot."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Absolute absent private output directory."),
    ],
    gpu_index: Annotated[
        int,
        typer.Option("--gpu-index", help="Physical GPU selected after busy/heat preflight."),
    ],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Generation mode: thinking or nonthinking."),
    ] = "thinking",
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Clean Git project containing the suite."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Run one M6 Base domain mode under the frozen release protocol."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    checked_mode = _validated_m6_domain_mode_and_paths(
        evidence_import=base_import,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        mode=mode,
        json_output=json_output,
    )
    try:
        imported = load_m6_base_import(base_import)
        release = load_m6_release_config(config)
    except (M6BaseImportError, M6ContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    result = _run_m6_domain_cli(
        model_identity=imported.model,
        expected_config_sha256=(
            canonical_config_hash(release)
            if release.protocol_version != "m6-release-v1"
            else imported.config_sha256
        ),
        config=config,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        project_root=project_root,
        gpu_index=gpu_index,
        mode=checked_mode,
        json_output=json_output,
    )
    _render_m6_domain_result(result, json_output=json_output)


@eval_app.command("m6-candidate-domain")
def eval_m6_candidate_domain(
    ctx: typer.Context,
    candidate_import: Annotated[
        Path,
        typer.Option("--candidate-import", help="Verified M6 Candidate-import JSON."),
    ],
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="Absolute frozen Candidate model export."),
    ],
    tokenizer_dir: Annotated[
        Path,
        typer.Option("--tokenizer-dir", help="Absolute pinned Base tokenizer snapshot."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Absolute absent private output directory."),
    ],
    gpu_index: Annotated[
        int,
        typer.Option("--gpu-index", help="Physical GPU selected after busy/heat preflight."),
    ],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Generation mode: thinking or nonthinking."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Clean Git project containing the suite."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Run one M6 Candidate domain mode under the frozen release protocol."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    checked_mode = _validated_m6_domain_mode_and_paths(
        evidence_import=candidate_import,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        mode=mode,
        json_output=json_output,
    )
    try:
        imported = load_m6_candidate_import(candidate_import)
    except (M6CandidateImportError, M6ContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    result = _run_m6_domain_cli(
        model_identity=imported.model,
        expected_config_sha256=imported.config_sha256,
        config=config,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        project_root=project_root,
        gpu_index=gpu_index,
        mode=checked_mode,
        json_output=json_output,
    )
    _render_m6_domain_result(result, json_output=json_output)


@eval_app.command("m6-candidate-general")
def eval_m6_candidate_general(
    ctx: typer.Context,
    candidate_import: Annotated[
        Path,
        typer.Option("--candidate-import", help="Verified M6 Candidate-import JSON."),
    ],
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="Absolute frozen Candidate model export."),
    ],
    tokenizer_dir: Annotated[
        Path,
        typer.Option("--tokenizer-dir", help="Absolute pinned Base tokenizer snapshot."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private TinyLLM Artifact Root."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Absolute absent private output directory."),
    ],
    gpu_index: Annotated[
        int,
        typer.Option("--gpu-index", help="Physical GPU selected after busy/heat preflight."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Clean Git project containing Task Adapters."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Run the complete frozen M6 Candidate general-regression suite."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not all(
        path.is_absolute()
        for path in (candidate_import, model_dir, tokenizer_dir, artifact_root, output_dir)
    ):
        _output_error("M6 general artifact paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        imported = load_m6_candidate_import(candidate_import)
        preflight_baseline_gpu(gpu_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        result = run_m6_general_pass(
            release_config_path=config,
            artifact_root=artifact_root,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=output_dir,
            project_root=project_root,
            physical_gpu_index=gpu_index,
            model_identity=imported.model,
            expected_config_sha256=imported.config_sha256,
        )
    except (M6CandidateImportError, M6ContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except BaselinePreflightError as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_PREFLIGHT_FAILED")
        raise typer.Exit(code=3) from exc
    except (M6GeneralError, BaselineRuntimeError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_GENERAL_FAILED")
        raise typer.Exit(code=6) from exc
    finally:
        if previous_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"succeeded: {result.evaluation_id} "
            f"acc_norm={result.general.aggregate_basis_points / 100:.2f}%"
        )


@eval_app.command("m6-domain-review")
def eval_m6_domain_review(
    ctx: typer.Context,
    pass_directory: Annotated[
        Path,
        typer.Option("--pass-directory", help="Absolute M6 mode output directory."),
    ],
    judgments: Annotated[
        Path,
        typer.Option("--judgments", help="Private ordered 40-item Judgment JSONL."),
    ],
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Git project containing the frozen suite."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable content-free machine-readable JSON."),
    ] = False,
) -> None:
    """Finalize one M6 domain pass with complete maintainer review."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if not pass_directory.is_absolute() or not judgments.is_absolute():
        _output_error("M6 review artifact paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        result = finalize_m6_domain_pass(
            project_root=project_root,
            pass_directory=pass_directory,
            judgments_path=judgments,
        )
    except (M6DomainError, BaselineContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_REVIEW_FAILED")
        raise typer.Exit(code=6) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"succeeded: mode={result.mode} score={result.correct_items}/300 "
            f"format={result.format_valid_items}/300"
        )


@eval_app.command("m6-assemble")
def eval_m6_assemble(
    ctx: typer.Context,
    role: Annotated[
        str,
        typer.Option("--role", help="Evaluation role: base or candidate."),
    ],
    evidence_import: Annotated[
        Path,
        typer.Option("--evidence-import", help="Verified Base/Candidate import JSON."),
    ],
    thinking_pass: Annotated[
        Path,
        typer.Option("--thinking-pass", help="Finalized absolute Thinking pass directory."),
    ],
    thinking_judgments: Annotated[
        Path,
        typer.Option("--thinking-judgments", help="Approved Thinking Judgment JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Absolute evaluation.json output path."),
    ],
    nonthinking_pass: Annotated[
        Path | None,
        typer.Option(
            "--nonthinking-pass",
            help="Finalized Candidate Non-thinking pass directory.",
        ),
    ] = None,
    nonthinking_judgments: Annotated[
        Path | None,
        typer.Option(
            "--nonthinking-judgments",
            help="Approved Candidate Non-thinking Judgment JSONL.",
        ),
    ] = None,
    general_pass: Annotated[
        Path | None,
        typer.Option("--general-pass", help="Completed Candidate general pass directory."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M6 release-policy YAML."),
    ] = Path("configs/eval/m6_release.yaml"),
    project_root: Annotated[
        Path,
        typer.Option("--project-root", help="Clean Git project root."),
    ] = Path("."),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable path-free machine-readable JSON."),
    ] = False,
) -> None:
    """Assemble one complete content-free M6 evaluation artifact."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if role not in {"base", "candidate"}:
        _output_error("M6 assembly role must be base or candidate", json_output=json_output)
        raise typer.Exit(code=2)
    required = (evidence_import, thinking_pass, thinking_judgments, output)
    if not all(path.is_absolute() for path in required):
        _output_error("M6 assembly artifact paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        release = load_m6_release_config(config)
        if role == "base":
            if release.protocol_version != "m6-release-v1":
                if (
                    nonthinking_pass is None
                    or nonthinking_judgments is None
                    or general_pass is not None
                    or not nonthinking_pass.is_absolute()
                    or not nonthinking_judgments.is_absolute()
                ):
                    raise M6AssemblyError(
                        "M6 holdout Base assembly requires absolute Non-thinking evidence"
                    )
                result = assemble_m6_base_v2_evaluation(
                    release_config_path=config,
                    base_import_path=evidence_import,
                    thinking_pass_directory=thinking_pass,
                    thinking_judgments_path=thinking_judgments,
                    nonthinking_pass_directory=nonthinking_pass,
                    nonthinking_judgments_path=nonthinking_judgments,
                    output_path=output,
                    project_root=project_root,
                )
            else:
                if any(
                    path is not None
                    for path in (nonthinking_pass, nonthinking_judgments, general_pass)
                ):
                    raise M6AssemblyError(
                        "M6 v1 Base assembly reuses imported Non-thinking and general evidence"
                    )
                result = assemble_m6_base_evaluation(
                    release_config_path=config,
                    base_import_path=evidence_import,
                    thinking_pass_directory=thinking_pass,
                    thinking_judgments_path=thinking_judgments,
                    output_path=output,
                    project_root=project_root,
                )
        else:
            if (
                nonthinking_pass is None
                or nonthinking_judgments is None
                or general_pass is None
                or not all(
                    path.is_absolute()
                    for path in (nonthinking_pass, nonthinking_judgments, general_pass)
                )
            ):
                raise M6AssemblyError(
                    "Candidate assembly requires absolute Non-thinking and general evidence"
                )
            result = assemble_m6_candidate_evaluation(
                release_config_path=config,
                candidate_import_path=evidence_import,
                thinking_pass_directory=thinking_pass,
                thinking_judgments_path=thinking_judgments,
                nonthinking_pass_directory=nonthinking_pass,
                nonthinking_judgments_path=nonthinking_judgments,
                general_pass_directory=general_pass,
                output_path=output,
                project_root=project_root,
            )
    except (M6AssemblyError, M6BaseImportError, M6CandidateImportError, M6ContractError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="M6_ASSEMBLY_FAILED")
        raise typer.Exit(code=3) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"succeeded: {result.evaluation_id} role={result.model.role}")


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


@app.command("serve")
def serve_command(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Strict M7 Gateway YAML configuration."),
    ] = Path("configs/serving/m7_gateway.yaml"),
    model: Annotated[
        str,
        typer.Option("--model", help="Production Alias or immutable M6/M7 model version."),
    ] = "production",
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit startup errors as stable JSON."),
    ] = False,
) -> None:
    """Start the authenticated local Gateway in the isolated serving environment."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    try:
        gateway_config = load_gateway_config(config)
        resolved = resolve_model(artifact_root, model)
    except ServingConfigError as exc:
        _output_error(str(exc), json_output=json_output, error_code="SERVING_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    token = os.environ.get(gateway_config.bearer_token_env)
    if token is None or len(token) < 32:
        _output_error(
            f"environment variable {gateway_config.bearer_token_env} must contain a "
            "Bearer Token of at least 32 characters",
            json_output=json_output,
            error_code="SERVING_AUTH_CONFIG_ERROR",
        )
        raise typer.Exit(code=7)
    try:
        import uvicorn

        from tinyllm.serving.backend import VLLMHTTPBackend
        from tinyllm.serving.gateway import create_gateway
        from tinyllm.serving.observability import StructuredEventLog
        from tinyllm.serving.supervisor import BackendSupervisor
    except ImportError as exc:
        _output_error(
            "serving dependencies are unavailable; use the isolated serving profile",
            json_output=json_output,
            error_code="SERVING_DEPENDENCY_ERROR",
        )
        raise typer.Exit(code=7) from exc
    supervisor = (
        BackendSupervisor(
            config=gateway_config,
            resolved_model=resolved,
            artifact_root=artifact_root,
        )
        if gateway_config.manage_backend
        else None
    )
    backend = VLLMHTTPBackend(
        gateway_config.backend_base_url,
        request_timeout_seconds=gateway_config.request_timeout_seconds,
        health_timeout_seconds=gateway_config.backend_health_timeout_seconds,
        internal_token=supervisor.internal_token if supervisor is not None else None,
    )
    gateway = create_gateway(
        config=gateway_config,
        resolved_model=resolved,
        backend=backend,
        bearer_token=token,
        supervisor=supervisor,
        event_log=StructuredEventLog(
            artifact_root / "deployments" / "runtime" / gateway_config.config_id / "events.jsonl"
        ),
    )
    try:
        uvicorn.run(
            gateway,
            host=gateway_config.host,
            port=gateway_config.port,
            log_level="info",
            access_log=False,
            http="h11",
            h11_max_incomplete_event_size=16_384,
            proxy_headers=False,
            server_header=False,
            date_header=False,
        )
    except (OSError, RuntimeError) as exc:
        _output_error(
            "Gateway failed to start or terminated unexpectedly",
            json_output=json_output,
            error_code="SERVING_RUNTIME_ERROR",
        )
        raise typer.Exit(code=7) from exc


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


@benchmark_app.command("train")
def benchmark_train(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M3 DDP benchmark YAML."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Private benchmark Run root."),
    ],
    evidence_dir: Annotated[
        Path,
        typer.Option("--evidence-dir", help="New private supervisor evidence directory."),
    ],
    profile_name: Annotated[
        str,
        typer.Option("--profile", help="Scaling profile: strong or weak."),
    ],
    repeat: Annotated[
        int,
        typer.Option("--repeat", min=1, help="Independent repeat number."),
    ],
    gpu_indices: Annotated[
        str,
        typer.Option("--gpu-indices", help="Ordered physical GPU indices, for example 4,5."),
    ],
    group: Annotated[
        str,
        typer.Option(
            "--group",
            help="Controlled group: standard, same_numa, or cross_numa.",
        ),
    ] = "standard",
    timeout_seconds: Annotated[
        int,
        typer.Option("--timeout-seconds", min=1, help="Bounded torchrun timeout."),
    ] = 7200,
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Run one formal M3 DDP training benchmark repetition."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    if profile_name not in {"strong", "weak"}:
        _output_error("profile must be strong or weak", json_output=json_output)
        raise typer.Exit(code=2)
    if group not in {"standard", "same_numa", "cross_numa"}:
        _output_error(
            "group must be standard, same_numa, or cross_numa",
            json_output=json_output,
        )
        raise typer.Exit(code=2)
    if not output_root.is_absolute() or not evidence_dir.is_absolute():
        _output_error("benchmark output paths must be absolute", json_output=json_output)
        raise typer.Exit(code=2)
    try:
        parsed_indices = parse_gpu_indices(gpu_indices)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output)
        raise typer.Exit(code=2) from exc
    try:
        result = run_formal_benchmark(
            config_path=config,
            output_root=output_root,
            evidence_dir=evidence_dir,
            profile=cast(BenchmarkProfile, profile_name),
            group=cast(BenchmarkGroup, group),
            repeat=repeat,
            gpu_indices=parsed_indices,
            timeout_seconds=timeout_seconds,
        )
    except DDPBenchmarkConfigError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BENCHMARK_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except BenchmarkPreflightError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BENCHMARK_PREFLIGHT_FAILED")
        raise typer.Exit(code=3) from exc
    except BenchmarkRunError as exc:
        _output_error(str(exc), json_output=json_output, error_code="BENCHMARK_RUN_FAILED")
        raise typer.Exit(code=4) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="BENCHMARK_RUN_FAILED")
        raise typer.Exit(code=4) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"pass: {result.run_id} profile={result.profile} world_size={result.world_size} "
            f"repeat={result.repeat} tokens_per_second={result.tokens_per_second:.3f}"
        )


@benchmark_app.command("inference")
def benchmark_inference(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Frozen M7 inference benchmark YAML."),
    ] = Path("configs/benchmark/m7_inference.yaml"),
    model: Annotated[
        str,
        typer.Option("--model", help="Production Alias or immutable M6/M7 model version."),
    ] = "production",
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Absolute private Artifact Store root."),
    ] = DEFAULT_ARTIFACT_ROOT,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="New absolute private benchmark directory."),
    ] = Path("."),
    direct_url: Annotated[
        str,
        typer.Option("--direct-url", help="Loopback Direct vLLM base URL."),
    ] = "http://127.0.0.1:8001",
    gateway_url: Annotated[
        str,
        typer.Option("--gateway-url", help="Loopback TinyLLM Gateway base URL."),
    ] = "http://127.0.0.1:8000",
    environment: Annotated[
        Path,
        typer.Option("--environment", help="Private serving environment JSON."),
    ] = Path("environment.json"),
    hardware: Annotated[
        Path,
        typer.Option("--hardware", help="Private serving hardware JSON."),
    ] = Path("hardware.json"),
    gateway_config: Annotated[
        Path,
        typer.Option("--gateway-config", help="Gateway YAML used during the benchmark."),
    ] = Path("configs/serving/m7_gateway_benchmark.yaml"),
    command_json: Annotated[
        bool,
        typer.Option("--json", help="Emit stable machine-readable JSON."),
    ] = False,
) -> None:
    """Run the frozen request-level Direct/Gateway inference benchmark."""

    state = cast(CLIState, ctx.obj)
    json_output = state.json_output or command_json
    token = os.environ.get("TINYLLM_GATEWAY_BEARER_TOKEN")
    if token is None or len(token) < 32:
        _output_error(
            "TINYLLM_GATEWAY_BEARER_TOKEN must contain at least 32 characters",
            json_output=json_output,
            error_code="SERVING_AUTH_CONFIG_ERROR",
        )
        raise typer.Exit(code=7)
    direct_token = os.environ.get("TINYLLM_VLLM_INTERNAL_TOKEN")
    if direct_token is None or len(direct_token) < 32:
        _output_error(
            "TINYLLM_VLLM_INTERNAL_TOKEN must contain at least 32 characters",
            json_output=json_output,
            error_code="SERVING_INTERNAL_AUTH_CONFIG_ERROR",
        )
        raise typer.Exit(code=7)
    if not output_dir.is_absolute() or not environment.is_absolute() or not hardware.is_absolute():
        _output_error(
            "M7 benchmark output and evidence paths must be absolute",
            json_output=json_output,
            error_code="INFERENCE_BENCHMARK_INPUT_ERROR",
        )
        raise typer.Exit(code=2)
    try:
        benchmark_config = load_inference_benchmark_config(config)
        gateway_benchmark_config = load_gateway_config(gateway_config)
        resolved = resolve_model(artifact_root, model)
        environment_sha256 = hashlib.sha256(environment.read_bytes()).hexdigest()
        hardware_sha256 = hashlib.sha256(hardware.read_bytes()).hexdigest()
        result = asyncio.run(
            run_inference_benchmark(
                config=benchmark_config,
                resolved_model=resolved,
                direct_url=direct_url,
                direct_bearer_token=direct_token,
                gateway_url=gateway_url,
                gateway_bearer_token=token,
                output_dir=output_dir,
                environment_sha256=environment_sha256,
                hardware_sha256=hardware_sha256,
                gateway_config_sha256=canonical_config_hash(gateway_benchmark_config),
            )
        )
    except InferenceBenchmarkConfigError as exc:
        _output_error(str(exc), json_output=json_output, error_code="INFERENCE_CONFIG_ERROR")
        raise typer.Exit(code=2) from exc
    except DeploymentError as exc:
        _deployment_error(exc, json_output=json_output)
    except (InferenceBenchmarkError, OSError, RuntimeError) as exc:
        _output_error(str(exc), json_output=json_output, error_code="INFERENCE_BENCHMARK_FAILED")
        raise typer.Exit(code=7) from exc
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{result.status}: requests={result.total_requests} "
            f"success_rate={result.success_rate_basis_points / 100:.2f}%"
        )
    if result.status != "succeeded":
        raise typer.Exit(code=7)


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
