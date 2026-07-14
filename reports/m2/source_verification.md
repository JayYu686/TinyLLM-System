# M2 pinned-source verification

Verification date: 2026-07-14 (Asia/Shanghai)

This report records a read-only availability and Dataset Card check. It does not claim that the
full datasets were downloaded, processed, or licensed beyond the policy in the data contract.

| Source | Pinned revision | Raw Dataset Card | HTTP | SHA256 |
| -- | -- | -- | --: | -- |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `README.md` | 200 | `68483ac2fcc2f3f7779f453352363827678070ef35b6d746ccb2ca6958540fff` |
| `bigcode/commitpackft` | `fc56fe33c030c6daa414c2b112c932b8eed085e6` | `README.md` | 200 | `69799047051e9d487d20e8930d73de1b3023ee1c75cf36dc3d2dfca2643f0bb0` |

The pinned OASST1 card declares Apache-2.0 and documents `ready_for_export`, positive-review, and
non-deleted message semantics. The pinned CommitPackFT card declares MIT for the dataset and also
exposes a per-sample source-repository `license` field. TinyLLM applies the per-sample allowlist;
the dataset-level declaration does not replace that check.

Reproduce the availability check without downloading dataset payloads:

```bash
curl -fsS -o oasst1.README.md \
  https://huggingface.co/datasets/OpenAssistant/oasst1/raw/fdf72ae0827c1cda404aff25b6603abec9e3399b/README.md
curl -fsS -o commitpackft.README.md \
  https://huggingface.co/datasets/bigcode/commitpackft/raw/fc56fe33c030c6daa414c2b112c932b8eed085e6/README.md
sha256sum oasst1.README.md commitpackft.README.md
```
