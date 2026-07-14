"""Pinned M2 upstream identities and conservative source-license policy."""

from __future__ import annotations

from tinyllm.data.schema import DatasetSource

OASST1_SOURCE = DatasetSource(
    name="oasst1",
    dataset_id="OpenAssistant/oasst1",
    revision="fdf72ae0827c1cda404aff25b6603abec9e3399b",
    dataset_card_url=(
        "https://huggingface.co/datasets/OpenAssistant/oasst1/blob/"
        "fdf72ae0827c1cda404aff25b6603abec9e3399b/README.md"
    ),
    dataset_card_license="apache-2.0",
    dataset_card_sha256="68483ac2fcc2f3f7779f453352363827678070ef35b6d746ccb2ca6958540fff",
)

COMMITPACKFT_SOURCE = DatasetSource(
    name="commitpackft",
    dataset_id="bigcode/commitpackft",
    revision="fc56fe33c030c6daa414c2b112c932b8eed085e6",
    dataset_card_url=(
        "https://huggingface.co/datasets/bigcode/commitpackft/blob/"
        "fc56fe33c030c6daa414c2b112c932b8eed085e6/README.md"
    ),
    dataset_card_license="mit",
    dataset_card_sha256="69799047051e9d487d20e8930d73de1b3023ee1c75cf36dc3d2dfca2643f0bb0",
)

COMMITPACKFT_LICENSE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)


def normalize_license(value: str) -> str:
    """Return a conservative canonical SPDX-like label without guessing aliases."""

    return value.strip().lower().replace("_", "-")
