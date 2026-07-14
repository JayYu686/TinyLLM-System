import pytest


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
