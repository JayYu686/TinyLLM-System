"""Single-device and distributed training infrastructure."""

from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointSelection,
    CheckpointStore,
)
from tinyllm.training.config import M1TrainingConfig, TrainingConfigError, load_training_config
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
    "CheckpointContext",
    "CheckpointError",
    "CheckpointErrorCode",
    "CheckpointSelection",
    "CheckpointStore",
    "InMemoryMetricSink",
    "ResumeMode",
    "SingleDeviceTrainer",
    "TrainerState",
    "TrainingConfigError",
    "TrainingError",
    "TrainingErrorCode",
    "TrainingResult",
    "TrainingStepMetrics",
    "build_m1_cpu_trainer",
    "build_m1_cuda_trainer",
    "load_training_config",
    "restore_from_config",
    "restore_trainer",
    "run_single_device_training",
    "seed_everything",
]
