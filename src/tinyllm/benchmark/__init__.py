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
    "RankBenchmarkMetrics",
    "ResolvedBenchmarkProfile",
    "build_m3_matrix_summary",
    "load_ddp_benchmark_config",
    "load_benchmark_evidence",
    "resolve_benchmark_profile",
    "validate_formal_m3_config",
]
