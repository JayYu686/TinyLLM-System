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
    M5DualModeCorrectionMixtureManifest,
    M5FormalDatasetManifest,
    M5FormatRepairMixtureManifest,
    M5MixtureManifest,
    M5R3ContentReviewJudgment,
    M5R3ContentReviewResult,
    M5R3FormalContaminationReport,
    M5R3FormalCPUSmoke,
    M5R3FormalShardArtifact,
    M5R3FormalSourceConfig,
    M5R3FormalSourceResult,
    M5R3MixtureConfig,
    M5R3MixtureManifest,
    M5R3P0CandidateAudit,
    M5R3P0Config,
    M5R3P0ContaminationReport,
    M5R3P0Result,
    M5R3P1CandidateAudit,
    M5R3P1ContaminationReport,
    M5R3P1CPUSmoke,
    M5R3P1Result,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
    M5R3P2Config,
    M5R3P2CPUSmoke,
    M5R3P2GenerationDelta,
    M5R3P2Result,
    M5R3SourceAudit,
    M5R3SourceAuditConfig,
    M5R3TeacherSourceStrategyConfig,
    M5R3TeacherSourceStrategyReview,
    M5ReasoningContaminationReport,
    M5ReasoningDataConfig,
    M5ReasoningDatasetManifest,
    M5TeacherPilotResult,
    M5TeacherSmokeResult,
    M6DomainGeneralizationMixtureManifest,
    M6GateRepairMixtureManifest,
    M6GateReplayMixtureManifest,
    OASST1ImportConfig,
    PackedSequence,
    PinnedDataArtifact,
    PipelineRejectedRecord,
    ProcessedSample,
    ReasoningContaminationMatch,
    ReasoningRejectedRecord,
    ReasoningSample,
    ReasoningTask,
    ReasoningTaskSetManifest,
    ReasoningVerifierResult,
    RegisteredDatasetSummary,
    RejectedRecord,
    SamplerState,
    TeacherGenerationRecord,
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
    M6BaseImportResult,
    M6BootstrapConfig,
    M6BootstrapInterval,
    M6CandidateImportResult,
    M6ComparisonResult,
    M6DomainExecutionConfig,
    M6DomainItemScore,
    M6DomainModeResult,
    M6DomainPassSummary,
    M6DomainTranscript,
    M6EvaluationResult,
    M6GateCheck,
    M6GateConfig,
    M6GeneralComparison,
    M6GeneralExecutionConfig,
    M6GeneralPassSummary,
    M6GeneralResult,
    M6GeneralTaskConfig,
    M6GeneralTaskResult,
    M6ModeComparison,
    M6ModelIdentity,
    M6NonthinkingGenerationConfig,
    M6PromotionRecord,
    M6ReleaseConfig,
    M6ThinkingGenerationConfig,
    MultipleChoiceScorer,
    RequiredTermsScorer,
)
from tinyllm.evaluation.m5_r2_schema import (
    M5R2DiagnosticDecision,
    M5R2OfflineAnalysis,
    M5R2ReplayConfig,
    M5R2ReplayItemResult,
    M5R2ReplaySummary,
)
from tinyllm.evaluation.m5_reasoning_schema import (
    M5AblationSelection,
    M5FormatFailureAnalysis,
    M5FormatRepairGateResult,
    M5ReasoningEvaluationConfig,
    M5ReasoningEvaluationSummary,
    M5ReasoningItemResult,
)
from tinyllm.evaluation.m5_thinking_budget_schema import (
    M5ThinkingBudgetEvaluationConfig,
    M5ThinkingBudgetEvaluationSummary,
    M5ThinkingBudgetGateResult,
    M5ThinkingBudgetItemResult,
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
from tinyllm.training.m5_ablation_schema import M5AblationRunResult, M5CheckpointManifest
from tinyllm.training.m5_config import M5SFTConfig
from tinyllm.training.m5_failure_schema import M5FailurePathEvidence
from tinyllm.training.m5_formal_schema import (
    M5FormalCampaignResult,
    M5FormalCheckpointManifest,
    M5FormalEnvironment,
    M5FormalEvaluationSnapshot,
    M5FormalHardware,
    M5FormalRunResult,
    M5FormalStagedEvaluation,
)
from tinyllm.training.m5_lora_schema import (
    M5LoRACampaignResult,
    M5LoRACheckpointManifest,
    M5LoRAEnvironment,
    M5LoRAHardware,
    M5LoRARunResult,
)
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
    "m5-ablation-mixture-manifest-v1.schema.json": M5MixtureManifest,
    "m5-formal-dataset-manifest-v1.schema.json": M5FormalDatasetManifest,
    "m5-dual-mode-correction-mixture-manifest-v1.schema.json": (
        M5DualModeCorrectionMixtureManifest
    ),
    "m5-format-repair-mixture-manifest-v1.schema.json": M5FormatRepairMixtureManifest,
    "m6-gate-repair-mixture-manifest-v1.schema.json": M6GateRepairMixtureManifest,
    "m6-gate-replay-mixture-manifest-v1.schema.json": M6GateReplayMixtureManifest,
    "m6-domain-generalization-mixture-manifest-v1.schema.json": (
        M6DomainGeneralizationMixtureManifest
    ),
    "m5-r3-formal-contamination-report-v1.schema.json": M5R3FormalContaminationReport,
    "m5-r3-formal-cpu-smoke-v1.schema.json": M5R3FormalCPUSmoke,
    "m5-r3-formal-shard-artifact-v1.schema.json": M5R3FormalShardArtifact,
    "m5-r3-formal-source-config-v1.schema.json": M5R3FormalSourceConfig,
    "m5-r3-formal-source-result-v1.schema.json": M5R3FormalSourceResult,
    "m5-r3-mixture-config-v1.schema.json": M5R3MixtureConfig,
    "m5-r3-mixture-manifest-v1.schema.json": M5R3MixtureManifest,
    "m5-r3-source-audit-config-v1.schema.json": M5R3SourceAuditConfig,
    "m5-r3-source-audit-v1.schema.json": M5R3SourceAudit,
    "m5-r3-teacher-source-strategy-config-v1.schema.json": M5R3TeacherSourceStrategyConfig,
    "m5-r3-teacher-source-strategy-review-v1.schema.json": M5R3TeacherSourceStrategyReview,
    "m5-r3-p0-candidate-audit-v1.schema.json": M5R3P0CandidateAudit,
    "m5-r3-p0-config-v1.schema.json": M5R3P0Config,
    "m5-r3-p0-contamination-report-v1.schema.json": M5R3P0ContaminationReport,
    "m5-r3-p0-result-v1.schema.json": M5R3P0Result,
    "m5-r3-p1-candidate-audit-v1.schema.json": M5R3P1CandidateAudit,
    "m5-r3-p1-contamination-report-v1.schema.json": M5R3P1ContaminationReport,
    "m5-r3-p1-cpu-smoke-v1.schema.json": M5R3P1CPUSmoke,
    "m5-r3-p1-result-v1.schema.json": M5R3P1Result,
    "m5-r3-p1-stage-generation-v1.schema.json": M5R3P1StageGeneration,
    "m5-r3-p1-task-context-v1.schema.json": M5R3P1TaskContext,
    "m5-r3-p2-config-v1.schema.json": M5R3P2Config,
    "m5-r3-p2-cpu-smoke-v1.schema.json": M5R3P2CPUSmoke,
    "m5-r3-p2-generation-delta-v1.schema.json": M5R3P2GenerationDelta,
    "m5-r3-p2-result-v1.schema.json": M5R3P2Result,
    "m5-r3-content-review-judgment-v1.schema.json": M5R3ContentReviewJudgment,
    "m5-r3-content-review-result-v1.schema.json": M5R3ContentReviewResult,
    "m5-ablation-run-result-v1.schema.json": M5AblationRunResult,
    "m5-ablation-selection-v1.schema.json": M5AblationSelection,
    "m5-format-failure-analysis-v1.schema.json": M5FormatFailureAnalysis,
    "m5-format-repair-gate-result-v1.schema.json": M5FormatRepairGateResult,
    "m5-checkpoint-manifest-v1.schema.json": M5CheckpointManifest,
    "m5-formal-checkpoint-manifest-v1.schema.json": M5FormalCheckpointManifest,
    "m5-formal-campaign-result-v1.schema.json": M5FormalCampaignResult,
    "m5-formal-environment-v1.schema.json": M5FormalEnvironment,
    "m5-formal-evaluation-snapshot-v1.schema.json": M5FormalEvaluationSnapshot,
    "m5-formal-hardware-v1.schema.json": M5FormalHardware,
    "m5-formal-run-result-v1.schema.json": M5FormalRunResult,
    "m5-formal-staged-evaluation-v1.schema.json": M5FormalStagedEvaluation,
    "m5-failure-path-evidence-v1.schema.json": M5FailurePathEvidence,
    "m5-lora-checkpoint-manifest-v1.schema.json": M5LoRACheckpointManifest,
    "m5-lora-campaign-result-v1.schema.json": M5LoRACampaignResult,
    "m5-lora-environment-v1.schema.json": M5LoRAEnvironment,
    "m5-lora-hardware-v1.schema.json": M5LoRAHardware,
    "m5-lora-run-result-v1.schema.json": M5LoRARunResult,
    "m5-reasoning-evaluation-config-v1.schema.json": M5ReasoningEvaluationConfig,
    "m5-reasoning-evaluation-summary-v1.schema.json": M5ReasoningEvaluationSummary,
    "m5-reasoning-item-result-v1.schema.json": M5ReasoningItemResult,
    "m5-thinking-budget-evaluation-config-v1.schema.json": M5ThinkingBudgetEvaluationConfig,
    "m5-thinking-budget-evaluation-summary-v1.schema.json": M5ThinkingBudgetEvaluationSummary,
    "m5-thinking-budget-gate-result-v1.schema.json": M5ThinkingBudgetGateResult,
    "m5-thinking-budget-item-result-v1.schema.json": M5ThinkingBudgetItemResult,
    "m5-r2-diagnostic-decision-v1.schema.json": M5R2DiagnosticDecision,
    "m5-r2-offline-analysis-v1.schema.json": M5R2OfflineAnalysis,
    "m5-r2-replay-config-v1.schema.json": M5R2ReplayConfig,
    "m5-r2-replay-item-result-v1.schema.json": M5R2ReplayItemResult,
    "m5-r2-replay-summary-v1.schema.json": M5R2ReplaySummary,
    "m5-reasoning-data-config-v1.schema.json": M5ReasoningDataConfig,
    "m5-reasoning-contamination-report-v1.schema.json": M5ReasoningContaminationReport,
    "m5-reasoning-dataset-manifest-v1.schema.json": M5ReasoningDatasetManifest,
    "m5-teacher-smoke-result-v1.schema.json": M5TeacherSmokeResult,
    "m5-teacher-pilot-result-v1.schema.json": M5TeacherPilotResult,
    "m6-bootstrap-config-v1.schema.json": M6BootstrapConfig,
    "m6-bootstrap-interval-v1.schema.json": M6BootstrapInterval,
    "m6-base-import-result-v1.schema.json": M6BaseImportResult,
    "m6-candidate-import-result-v1.schema.json": M6CandidateImportResult,
    "m6-comparison-result-v1.schema.json": M6ComparisonResult,
    "m6-domain-execution-config-v1.schema.json": M6DomainExecutionConfig,
    "m6-domain-item-score-v1.schema.json": M6DomainItemScore,
    "m6-domain-mode-result-v1.schema.json": M6DomainModeResult,
    "m6-domain-pass-summary-v1.schema.json": M6DomainPassSummary,
    "m6-domain-transcript-v1.schema.json": M6DomainTranscript,
    "m6-evaluation-result-v1.schema.json": M6EvaluationResult,
    "m6-gate-check-v1.schema.json": M6GateCheck,
    "m6-gate-config-v1.schema.json": M6GateConfig,
    "m6-general-comparison-v1.schema.json": M6GeneralComparison,
    "m6-general-execution-config-v1.schema.json": M6GeneralExecutionConfig,
    "m6-general-pass-summary-v1.schema.json": M6GeneralPassSummary,
    "m6-general-result-v1.schema.json": M6GeneralResult,
    "m6-general-task-config-v1.schema.json": M6GeneralTaskConfig,
    "m6-general-task-result-v1.schema.json": M6GeneralTaskResult,
    "m6-mode-comparison-v1.schema.json": M6ModeComparison,
    "m6-model-identity-v1.schema.json": M6ModelIdentity,
    "m6-nonthinking-generation-config-v1.schema.json": M6NonthinkingGenerationConfig,
    "m6-promotion-record-v1.schema.json": M6PromotionRecord,
    "m6-release-config-v1.schema.json": M6ReleaseConfig,
    "m6-thinking-generation-config-v1.schema.json": M6ThinkingGenerationConfig,
    "oasst1-import-config-v1.schema.json": OASST1ImportConfig,
    "multiple-choice-scorer-v1.schema.json": MultipleChoiceScorer,
    "pipeline-rejected-record-v1.schema.json": PipelineRejectedRecord,
    "pinned-data-artifact-v1.schema.json": PinnedDataArtifact,
    "packed-sequence-v1.schema.json": PackedSequence,
    "processed-sample-v1.schema.json": ProcessedSample,
    "rejected-record-v1.schema.json": RejectedRecord,
    "registered-dataset-summary-v1.schema.json": RegisteredDatasetSummary,
    "reasoning-rejected-record-v1.schema.json": ReasoningRejectedRecord,
    "reasoning-contamination-match-v1.schema.json": ReasoningContaminationMatch,
    "reasoning-sample-v1.schema.json": ReasoningSample,
    "reasoning-task-set-manifest-v1.schema.json": ReasoningTaskSetManifest,
    "reasoning-task-v1.schema.json": ReasoningTask,
    "reasoning-verifier-result-v1.schema.json": ReasoningVerifierResult,
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
    "teacher-generation-record-v1.schema.json": TeacherGenerationRecord,
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
