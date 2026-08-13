"""Reproducible training benchmark contracts and aggregation."""

from tinyllm.benchmark.aggregate import build_m3_matrix_summary, load_benchmark_evidence
from tinyllm.benchmark.config import (
    DDPBenchmarkConfig,
    DDPBenchmarkConfigError,
    ResolvedBenchmarkProfile,
    load_ddp_benchmark_config,
    resolve_benchmark_profile,
    validate_formal_m3_config,
)
from tinyllm.benchmark.inference import InferenceBenchmarkError, run_inference_benchmark
from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkConfig,
    InferenceBenchmarkConfigError,
    InferenceBenchmarkSummary,
    InferenceRequestResult,
    load_inference_benchmark_config,
)
from tinyllm.benchmark.schema import (
    BenchmarkProfileAggregate,
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkMatrixSummary,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
)

__all__ = [
    "BenchmarkProfileAggregate",
    "BenchmarkTimingSummary",
    "CommunicationMeasurement",
    "DDPBenchmarkConfig",
    "DDPBenchmarkConfigError",
    "DDPBenchmarkMatrixSummary",
    "DDPBenchmarkRunResult",
    "InferenceBenchmarkConfig",
    "InferenceBenchmarkConfigError",
    "InferenceBenchmarkError",
    "InferenceBenchmarkSummary",
    "InferenceRequestResult",
    "RankBenchmarkMetrics",
    "ResolvedBenchmarkProfile",
    "build_m3_matrix_summary",
    "load_ddp_benchmark_config",
    "load_benchmark_evidence",
    "load_inference_benchmark_config",
    "resolve_benchmark_profile",
    "run_inference_benchmark",
    "validate_formal_m3_config",
]
