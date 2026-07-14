# M2.3a pinned Qwen3 tokenizer smoke

Execution date: 2026-07-14 (Asia/Shanghai)

Status: **PASS**

This smoke used the pinned Qwen3-0.6B tokenizer files and the repository's synthetic CC0-1.0
samples. No model weights were downloaded or loaded. It verifies artifact integrity, backend
compatibility, Non-thinking ChatML rendering, offset alignment, and Assistant-only labels; it is
not evidence of a completed dataset build or model quality.

## Verified artifacts

| Item | Actual result |
| -- | -- |
| Model/tokenizer revision | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| `tokenizer.json` | 11,422,654 bytes; SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 bytes; SHA256 `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| Backend | `tokenizers==0.21.4` |
| Vocabulary | 151,669 including Added Tokens |
| Pad / ChatML start / ChatML end IDs | 151643 / 151644 / 151645 |
| Template | `qwen3-chatml-nonthinking-v1`; no Think block |

## Observed encoding

| Synthetic sample | Tokens | Supervised tokens | Supervised `<|im_end|>` | Result |
| -- | --: | --: | --: | -- |
| CommitPackFT code edit | 34 | 5 | 1 | PASS |
| OASST1 conversation | 23 | 6 | 1 | PASS |

For both samples, every system/user token, assistant header token, and post-response newline was
masked with `-100`. Assistant content and its terminating `<|im_end|>` retained the real token ID.
The full Token ID and Label arrays are retained in
[raw/qwen3_tokenizer_smoke.json](raw/qwen3_tokenizer_smoke.json).

Reproduce after placing the two pinned files in a private cache directory:

```bash
.venv/bin/python scripts/run_m2_tokenizer_smoke.py --tokenizer-dir /path/to/private/cache
```

## Boundaries

- The tokenizer files were read from a private cache and are not committed to Git.
- The raw evidence contains only Token IDs for public synthetic fixtures, not upstream dataset text.
- Token balancing, Packing, final Dataset Manifest, Registry, and full-source builds remain #39/#40.
