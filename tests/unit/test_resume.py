from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy
import pytest
import torch

from tinyllm.schemas import CheckpointManifest, canonical_config_hash, generate_run_id
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointStore,
    ResumeMode,
    SingleDeviceTrainer,
    build_m1_cpu_trainer,
    restore_from_config,
    restore_trainer,
)
from tinyllm.training.config import M1TrainingConfig, training_config_from_mapping


def resume_config(
    *,
    vocab_size: int = 24,
    seed: int = 23,
    learning_rate: float = 0.01,
    output_dir: str = "runs/resume-unit",
    resume: Literal["none", "auto", "exact", "warm", "transfer"] = "none",
) -> M1TrainingConfig:
    return training_config_from_mapping(
        {
            "schema_version": "1.0",
            "run": {"name": "resume-unit", "seed": seed},
            "model": {
                "vocab_size": vocab_size,
                "hidden_size": 24,
                "num_layers": 1,
                "num_heads": 4,
                "intermediate_size": 48,
                "max_sequence_length": 12,
                "rope_theta": 10_000.0,
                "rms_norm_epsilon": 1.0e-6,
                "dropout": 0.0,
                "tie_word_embeddings": True,
            },
            "data": {
                "kind": "toy",
                "vocab_size": vocab_size,
                "sequence_length": 10,
                "num_samples": 32,
            },
            "training": {
                "max_steps": 6,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "warmup_steps": 2,
            },
            "precision": {
                "dtype": "fp32",
                "allow_tf32": False,
                "use_grad_scaler": False,
            },
            "checkpoint": {
                "output_dir": output_dir,
                "save_steps": 3,
                "keep_last": 2,
                "resume": resume,
            },
        }
    )


def resume_context(config: M1TrainingConfig) -> CheckpointContext:
    config_hash = canonical_config_hash(config)
    return CheckpointContext(
        run_id=generate_run_id(
            config.run.name,
            config_hash,
            now=datetime(2026, 7, 14, tzinfo=UTC),
            nonce="beef",
        ),
        dataset_version="toy-resume-v1",
        git_commit="b" * 40,
        environment={"python": "3.11", "torch": str(torch.__version__), "device": "cpu"},
    )


def save_trainer(
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    config: M1TrainingConfig,
    context: CheckpointContext,
) -> CheckpointManifest:
    assert trainer.sampler is not None
    return store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        sampler=trainer.sampler,
        trainer_state=trainer.state,
        config=config,
        context=context,
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_cpu_exact_resume_matches_uninterrupted_training_bit_for_bit(tmp_path: Path) -> None:
    config = resume_config()
    context = resume_context(config)

    uninterrupted = build_m1_cpu_trainer(config)
    uninterrupted_result = uninterrupted.train()

    interrupted = build_m1_cpu_trainer(config)
    interrupted.train(target_global_step=3)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, interrupted, config, context)
    payload = store.load_training_state(manifest.checkpoint_id)

    resumed_config = resume_config(output_dir="different-output", resume="auto")
    resumed = build_m1_cpu_trainer(resumed_config)
    random.random()
    numpy.random.random()
    torch.rand(3)
    result = restore_trainer(
        store=store,
        trainer=resumed,
        mode=ResumeMode.EXACT,
        context=context,
    )

    assert result.target_global_step == 3
    assert result.skipped_invalid_checkpoints == ()
    saved_rng = cast(dict[str, Any], payload["rng"])
    assert random.getstate() == saved_rng["python"]
    numpy_state = cast(tuple[Any, ...], numpy.random.get_state())
    saved_numpy_state = cast(tuple[Any, ...], saved_rng["numpy"])
    assert numpy_state[0] == saved_numpy_state[0]
    numpy.testing.assert_array_equal(numpy_state[1], saved_numpy_state[1])
    assert numpy_state[2:] == saved_numpy_state[2:]
    torch.testing.assert_close(
        torch.get_rng_state(), cast(torch.Tensor, saved_rng["torch"]), rtol=0, atol=0
    )

    resumed_result = resumed.train()

    assert resumed_result.metrics == uninterrupted_result.metrics[3:]
    assert resumed_result.state == uninterrupted_result.state
    for resumed_parameter, uninterrupted_parameter in zip(
        resumed.model.parameters(), uninterrupted.model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, uninterrupted_parameter, rtol=0, atol=0)
    assert_nested_equal(resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict())
    assert_nested_equal(
        resumed.scheduler.state_dict(),  # type: ignore[no-untyped-call]
        uninterrupted.scheduler.state_dict(),  # type: ignore[no-untyped-call]
    )


@pytest.mark.parametrize(
    ("config", "context_variant", "reason"),
    [
        (resume_config(learning_rate=0.02), "none", "config:training.learning_rate"),
        (resume_config(), "dataset", "lineage:dataset_version"),
        (resume_config(), "world_size", "lineage:world_size"),
        (resume_config(), "git", "lineage:git_commit"),
        (resume_config(), "environment", "lineage:environment"),
    ],
)
def test_exact_resume_rejects_config_and_lineage_drift(
    tmp_path: Path,
    config: M1TrainingConfig,
    context_variant: Literal["none", "dataset", "world_size", "git", "environment"],
    reason: str,
) -> None:
    source_config = resume_config()
    source_context = resume_context(source_config)
    source = build_m1_cpu_trainer(source_config)
    source.train(target_global_step=2)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, source, source_config, source_context)
    target = build_m1_cpu_trainer(config)
    if context_variant == "dataset":
        target_context = replace(source_context, dataset_version="different-data")
    elif context_variant == "world_size":
        target_context = replace(source_context, world_size=2)
    elif context_variant == "git":
        target_context = replace(source_context, git_commit="c" * 40)
    elif context_variant == "environment":
        target_context = replace(
            source_context,
            environment={"python": "3.12", "torch": "other", "device": "cpu"},
        )
    else:
        target_context = source_context

    with pytest.raises(CheckpointError) as caught:
        restore_trainer(
            store=store,
            trainer=target,
            mode=ResumeMode.EXACT,
            context=target_context,
            checkpoint_id=manifest.checkpoint_id,
        )

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE
    assert caught.value.context["reason"] == reason
    assert target.is_pristine


def test_auto_resume_reports_newer_corrupt_candidate_but_explicit_restore_rejects_it(
    tmp_path: Path,
) -> None:
    config = resume_config()
    context = resume_context(config)
    source = build_m1_cpu_trainer(config)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    source.train(target_global_step=1)
    first = save_trainer(store, source, config, context)
    source.train(target_global_step=2)
    second = save_trainer(store, source, config, context)
    with (store.root / second.checkpoint_id / "training_state.pt").open("ab") as stream:
        stream.write(b"corrupt")

    target = build_m1_cpu_trainer(config)
    result = restore_trainer(
        store=store,
        trainer=target,
        mode=ResumeMode.EXACT,
        context=context,
    )
    assert result.checkpoint_id == first.checkpoint_id
    assert result.skipped_invalid_checkpoints == (second.checkpoint_id,)

    explicit_target = build_m1_cpu_trainer(config)
    with pytest.raises(CheckpointError) as caught:
        restore_trainer(
            store=store,
            trainer=explicit_target,
            mode=ResumeMode.EXACT,
            context=context,
            checkpoint_id=second.checkpoint_id,
        )
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT


def test_exact_resume_explicitly_rejects_precision_drift(tmp_path: Path) -> None:
    config = resume_config()
    context = resume_context(config)
    source = build_m1_cpu_trainer(config)
    source.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, source, config, context)

    target = build_m1_cpu_trainer(config)
    target.config = config.model_copy(
        update={"precision": config.precision.model_copy(update={"dtype": "bf16"})}
    )
    with pytest.raises(CheckpointError) as caught:
        restore_trainer(
            store=store,
            trainer=target,
            mode=ResumeMode.EXACT,
            context=context,
            checkpoint_id=manifest.checkpoint_id,
        )
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE
    assert caught.value.context["reason"] == "config:precision.dtype"


def test_warm_resume_loads_complete_model_and_resets_runtime_state(tmp_path: Path) -> None:
    source_config = resume_config()
    source_context = resume_context(source_config)
    source = build_m1_cpu_trainer(source_config)
    source.train(target_global_step=2)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, source, source_config, source_context)

    target = build_m1_cpu_trainer(resume_config(seed=99, resume="warm"))
    result = restore_from_config(
        store=store,
        trainer=target,
        checkpoint_id=manifest.checkpoint_id,
    )

    assert result is not None
    assert result.mode == "warm"
    assert target.state.global_step == 0
    assert not target.optimizer.state
    assert target.sampler is not None
    assert target.sampler.cursor == 0
    for target_parameter, source_parameter in zip(
        target.model.parameters(), source.model.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)


def test_transfer_resume_reports_and_skips_shape_incompatible_weights(tmp_path: Path) -> None:
    source_config = resume_config(vocab_size=24)
    source_context = resume_context(source_config)
    source = build_m1_cpu_trainer(source_config)
    source.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, source, source_config, source_context)

    target = build_m1_cpu_trainer(resume_config(vocab_size=28, resume="transfer"))
    result = restore_from_config(
        store=store,
        trainer=target,
        checkpoint_id=manifest.checkpoint_id,
    )

    assert result is not None
    assert result.mode == "transfer"
    assert result.loaded_model_keys
    assert "token_embeddings.weight" in result.incompatible_checkpoint_keys
    assert "lm_head.weight" in result.incompatible_checkpoint_keys
    assert target.state.global_step == 0
    assert not target.optimizer.state


def test_restore_rejects_non_pristine_target_and_none_policy_is_noop(tmp_path: Path) -> None:
    config = resume_config()
    context = resume_context(config)
    source = build_m1_cpu_trainer(config)
    source.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_trainer(store, source, config, context)

    target = build_m1_cpu_trainer(config)
    assert restore_from_config(store=store, trainer=target, context=context) is None
    target.train(target_global_step=1)
    with pytest.raises(CheckpointError) as caught:
        restore_trainer(
            store=store,
            trainer=target,
            mode=ResumeMode.WARM,
            checkpoint_id=manifest.checkpoint_id,
        )
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE
    assert caught.value.context["reason"] == "target_not_pristine"
