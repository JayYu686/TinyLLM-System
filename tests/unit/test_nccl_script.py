import argparse

import pytest

from scripts.run_nccl_matrix import parse_gpu_group


def test_parse_nccl_gpu_group() -> None:
    assert parse_gpu_group("1,2,3") == (1, 2, 3)


@pytest.mark.parametrize("value", ["", "1,1", "-1", "a,b"])
def test_parse_nccl_gpu_group_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gpu_group(value)
