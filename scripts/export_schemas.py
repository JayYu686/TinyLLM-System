#!/usr/bin/env python3
"""Export deterministic public JSON Schemas from Pydantic models."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from pydantic import BaseModel

from tinyllm.benchmark import (
    BenchmarkProfileAggregate,
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkConfig,
    DDPBenchmarkMatrixSummary,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
    ResolvedBenchmarkProfile,
)
from tinyllm.data import (
    BalanceRejectedRecord,
    CommitPackFTImportConfig,
    DataImportManifest,
    DataProcessingManifest,
    DatasetArtifactFile,
    DatasetCommitMarker,
    DatasetRegistration,
    DatasetShardMetadata,
    DatasetShardPack,
    DistributedSamplerState,
    ImportedSample,
    M2AcquisitionManifest,
    M2DatasetManifest,
    M2PackingConfig,
    M2ProcessingConfig,
    M2TokenizationConfig,
    OASST1ImportConfig,
    PackedSequence,
    PinnedDataArtifact,
    PipelineRejectedRecord,
    ProcessedSample,
    RegisteredDatasetSummary,
    RejectedRecord,
    SamplerState,
    TokenizationRejectedRecord,
    TokenizedSample,
)
from tinyllm.evaluation import (
    AuthoredProvenance,
    BaselineEvaluationResult,
    BaselineRunConfig,
    ContaminationMatch,
    ContaminationReport,
    DomainBaselineSummary,
    DomainItemResult,
    EvaluationBuildConfig,
    EvaluationItem,
    EvaluationSetManifest,
    ExactMatchScorer,
    GeneralBaselineSummary,
    GeneralTaskResult,
    HumanReviewCommit,
    HumanRubricJudgment,
    HumanRubricScorer,
    JsonObjectScorer,
    MultipleChoiceScorer,
    RequiredTermsScorer,
)
from tinyllm.schemas.checkpoint import CheckpointCommitMarker, CheckpointManifest
from tinyllm.schemas.resume import ResumeResult
from tinyllm.schemas.run import RunManifest
from tinyllm.schemas.training_run import TrainingRunResult
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.ddp_recovery_schema import DDPRecoveryResult
from tinyllm.training.ddp_schema import (
    DDPCorrectnessSummary,
    DDPPartitionEvidence,
    DDPTrainingResult,
)
from tinyllm.training.fsdp2_config import FSDP2CorrectnessConfig, FSDP2RecoveryConfig
from tinyllm.training.fsdp2_recovery_schema import FSDP2RecoveryResult
from tinyllm.training.fsdp2_schema import (
    FSDP2CorrectnessSummary,
    FSDP2RankEvidence,
    FSDP2RankFailureEvidence,
    FSDP2TrainingResult,
)
from tinyllm.training.m4_dataset import M4DatasetViewManifest
from tinyllm.training.m4_dependencies import M4DependencySmokeResult
from tinyllm.training.m4_model_schema import M4ModelArtifactFile, M4ModelArtifactManifest
from tinyllm.training.m4_qwen_config import M4QwenFSDP2Config
from tinyllm.training.m4_qwen_schema import M4QwenRankMemory, M4QwenRunResult
from tinyllm.training.m5_config import M5SFTConfig
from tinyllm.training.metrics import TrainerState, TrainingStepMetrics

SCHEMAS: dict[str, type[BaseModel]] = {
    "balance-rejected-record-v1.schema.json": BalanceRejectedRecord,
    "baseline-evaluation-result-v1.schema.json": BaselineEvaluationResult,
    "baseline-run-config-v1.schema.json": BaselineRunConfig,
    "benchmark-profile-aggregate-v1.schema.json": BenchmarkProfileAggregate,
    "benchmark-timing-summary-v1.schema.json": BenchmarkTimingSummary,
    "checkpoint-manifest-v1.schema.json": CheckpointManifest,
    "checkpoint-commit-marker-v1.schema.json": CheckpointCommitMarker,
    "commitpackft-import-config-v1.schema.json": CommitPackFTImportConfig,
    "communication-measurement-v1.schema.json": CommunicationMeasurement,
    "contamination-match-v1.schema.json": ContaminationMatch,
    "contamination-report-v1.schema.json": ContaminationReport,
    "data-import-manifest-v1.schema.json": DataImportManifest,
    "data-processing-manifest-v1.schema.json": DataProcessingManifest,
    "ddp-correctness-summary-v1.schema.json": DDPCorrectnessSummary,
    "ddp-benchmark-config-v1.schema.json": DDPBenchmarkConfig,
    "ddp-benchmark-matrix-summary-v1.schema.json": DDPBenchmarkMatrixSummary,
    "ddp-benchmark-run-result-v1.schema.json": DDPBenchmarkRunResult,
    "ddp-partition-evidence-v1.schema.json": DDPPartitionEvidence,
    "ddp-recovery-result-v1.schema.json": DDPRecoveryResult,
    "ddp-training-result-v1.schema.json": DDPTrainingResult,
    "dataset-artifact-file-v1.schema.json": DatasetArtifactFile,
    "dataset-commit-marker-v1.schema.json": DatasetCommitMarker,
    "dataset-registration-v1.schema.json": DatasetRegistration,
    "dataset-shard-metadata-v1.schema.json": DatasetShardMetadata,
    "dataset-shard-pack-v1.schema.json": DatasetShardPack,
    "domain-baseline-summary-v1.schema.json": DomainBaselineSummary,
    "domain-item-result-v1.schema.json": DomainItemResult,
    "distributed-sampler-state-v1.schema.json": DistributedSamplerState,
    "evaluation-authored-provenance-v1.schema.json": AuthoredProvenance,
    "evaluation-build-config-v1.schema.json": EvaluationBuildConfig,
    "evaluation-item-v1.schema.json": EvaluationItem,
    "evaluation-set-manifest-v1.schema.json": EvaluationSetManifest,
    "exact-match-scorer-v1.schema.json": ExactMatchScorer,
    "general-baseline-summary-v1.schema.json": GeneralBaselineSummary,
    "general-task-result-v1.schema.json": GeneralTaskResult,
    "fsdp2-correctness-config-v1.schema.json": FSDP2CorrectnessConfig,
    "fsdp2-correctness-summary-v1.schema.json": FSDP2CorrectnessSummary,
    "fsdp2-recovery-config-v1.schema.json": FSDP2RecoveryConfig,
    "fsdp2-recovery-result-v1.schema.json": FSDP2RecoveryResult,
    "fsdp2-rank-evidence-v1.schema.json": FSDP2RankEvidence,
    "fsdp2-rank-failure-evidence-v1.schema.json": FSDP2RankFailureEvidence,
    "fsdp2-training-result-v1.schema.json": FSDP2TrainingResult,
    "human-rubric-scorer-v1.schema.json": HumanRubricScorer,
    "human-rubric-judgment-v1.schema.json": HumanRubricJudgment,
    "human-review-commit-v1.schema.json": HumanReviewCommit,
    "imported-sample-v1.schema.json": ImportedSample,
    "json-object-scorer-v1.schema.json": JsonObjectScorer,
    "m2-processing-config-v1.schema.json": M2ProcessingConfig,
    "m2-acquisition-manifest-v1.schema.json": M2AcquisitionManifest,
    "m2-dataset-manifest-v1.schema.json": M2DatasetManifest,
    "m2-packing-config-v1.schema.json": M2PackingConfig,
    "m2-tokenization-config-v1.schema.json": M2TokenizationConfig,
    "m1-training-config-v1.schema.json": M1TrainingConfig,
    "m4-dependency-smoke-result-v1.schema.json": M4DependencySmokeResult,
    "m4-dataset-view-manifest-v1.schema.json": M4DatasetViewManifest,
    "m4-model-artifact-file-v1.schema.json": M4ModelArtifactFile,
    "m4-model-artifact-manifest-v1.schema.json": M4ModelArtifactManifest,
    "m4-qwen-fsdp2-config-v1.schema.json": M4QwenFSDP2Config,
    "m4-qwen-rank-memory-v1.schema.json": M4QwenRankMemory,
    "m4-qwen-run-result-v1.schema.json": M4QwenRunResult,
    "m5-sft-config-v1.schema.json": M5SFTConfig,
    "oasst1-import-config-v1.schema.json": OASST1ImportConfig,
    "multiple-choice-scorer-v1.schema.json": MultipleChoiceScorer,
    "pipeline-rejected-record-v1.schema.json": PipelineRejectedRecord,
    "pinned-data-artifact-v1.schema.json": PinnedDataArtifact,
    "packed-sequence-v1.schema.json": PackedSequence,
    "processed-sample-v1.schema.json": ProcessedSample,
    "rejected-record-v1.schema.json": RejectedRecord,
    "registered-dataset-summary-v1.schema.json": RegisteredDatasetSummary,
    "required-terms-scorer-v1.schema.json": RequiredTermsScorer,
    "rank-benchmark-metrics-v1.schema.json": RankBenchmarkMetrics,
    "resolved-benchmark-profile-v1.schema.json": ResolvedBenchmarkProfile,
    "run-manifest-v1.schema.json": RunManifest,
    "resume-result-v1.schema.json": ResumeResult,
    "sampler-state-v1.schema.json": SamplerState,
    "trainer-state-v1.schema.json": TrainerState,
    "training-step-metrics-v1.schema.json": TrainingStepMetrics,
    "training-run-result-v1.schema.json": TrainingRunResult,
    "tokenization-rejected-record-v1.schema.json": TokenizationRejectedRecord,
    "tokenized-sample-v1.schema.json": TokenizedSample,
}


def render_schema(model: type[BaseModel]) -> str:
    """Render one schema using canonical formatting."""

    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write schemas, or verify that committed snapshots are current."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when a committed schema differs instead of rewriting it.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "schemas"
    if not args.check:
        output_dir.mkdir(exist_ok=True)
    stale: list[str] = []
    for filename, model in SCHEMAS.items():
        path = output_dir / filename
        rendered = render_schema(model)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                stale.append(filename)
        else:
            path.write_text(rendered, encoding="utf-8")
    if stale:
        parser.error(
            "stale schema snapshots: " + ", ".join(stale) + "; run scripts/export_schemas.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
