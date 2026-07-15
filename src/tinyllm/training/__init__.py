"""Single-device and distributed training infrastructure."""

from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointSelection,
    CheckpointStore,
)
from tinyllm.training.config import (
    DistributedConfig,
    M1TrainingConfig,
    TrainingConfigError,
    load_training_config,
)
from tinyllm.training.ddp import run_ddp_correctness
from tinyllm.training.ddp_checkpoint import (
    DDPCheckpointStore,
    LoadedDDPCheckpoint,
    build_rank_state,
    restore_local_rng_state,
    validate_local_rng_state,
)
from tinyllm.training.ddp_recovery_schema import DDPRecoveryResult
from tinyllm.training.ddp_resume import restore_ddp_trainer
from tinyllm.training.ddp_schema import (
    DDPCorrectnessSummary,
    DDPPartitionEvidence,
    DDPTrainingResult,
)
from tinyllm.training.distributed import TorchrunEnvironment, validate_sampler_partitions
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.metrics import InMemoryMetricSink, TrainerState, TrainingStepMetrics
from tinyllm.training.resume import ResumeMode, restore_from_config, restore_trainer
from tinyllm.training.run import run_single_device_training
from tinyllm.training.seed import seed_everything
from tinyllm.training.trainer import (
    SingleDeviceTrainer,
    TrainingResult,
    build_m1_cpu_trainer,
    build_m1_cuda_trainer,
)

__all__ = [
    "M1TrainingConfig",
    "DistributedConfig",
    "CheckpointContext",
    "CheckpointError",
    "CheckpointErrorCode",
    "CheckpointSelection",
    "CheckpointStore",
    "InMemoryMetricSink",
    "DDPCorrectnessSummary",
    "DDPCheckpointStore",
    "DDPPartitionEvidence",
    "DDPRecoveryResult",
    "DDPTrainingResult",
    "LoadedDDPCheckpoint",
    "ResumeMode",
    "SingleDeviceTrainer",
    "TrainerState",
    "TorchrunEnvironment",
    "TrainingConfigError",
    "TrainingError",
    "TrainingErrorCode",
    "TrainingResult",
    "TrainingStepMetrics",
    "build_m1_cpu_trainer",
    "build_m1_cuda_trainer",
    "build_rank_state",
    "load_training_config",
    "restore_from_config",
    "restore_ddp_trainer",
    "restore_local_rng_state",
    "restore_trainer",
    "run_single_device_training",
    "run_ddp_correctness",
    "seed_everything",
    "validate_sampler_partitions",
    "validate_local_rng_state",
]
