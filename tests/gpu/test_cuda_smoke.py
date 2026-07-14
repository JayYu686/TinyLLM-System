from pathlib import Path

import pytest

from tinyllm.training import build_m1_cuda_trainer, load_training_config


@pytest.mark.gpu
def test_cuda_bf16_smoke() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.is_bf16_supported()
    left = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    right = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    result = left @ right
    torch.cuda.synchronize()
    assert result.dtype == torch.bfloat16
    assert result.float().mean().item() == pytest.approx(16.0)


@pytest.mark.gpu
def test_native_m1_bf16_trainer_smoke() -> None:
    config = load_training_config(Path("configs/pretrain/tinygpt_debug_rtx3090_bf16_smoke.yaml"))
    trainer = build_m1_cuda_trainer(config)

    result = trainer.train(target_global_step=2)

    assert trainer.autocast_dtype is not None
    assert result.state.global_step == 2
    assert all(metric.loss > 0 for metric in result.metrics)
