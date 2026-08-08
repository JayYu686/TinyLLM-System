# M5 public summary

M5 completed both reviewed Qwen3 post-training routes on RTX 3090 hardware.

- Qwen3-0.6B completed a four-GPU BF16 Full-SFT campaign over 50M supervised tokens.
  A fresh process stopped at 2,002,739 tokens and an independent `torchrun` process
  performed Exact Resume to 50M. Five immutable snapshots at the 10M–50M targets were
  evaluated with the same 200-item dual-mode M5 development suite.
- The 10M Full-SFT snapshot was the strongest joint development point: 95.0% Thinking
  and 47.5% Non-thinking accuracy, compared with 70.5% and 37.0% for the pinned Base.
  The 50M endpoint reached 91.5% and 39.0%, so the complete curve is retained as real
  over-training evidence and the 10M snapshot is prioritized for M6 comparison.
- Qwen3-8B completed a single-GPU BF16 LoRA campaign over 10M supervised tokens, with
  Exact Resume at 5,000,444 tokens and an adapter-only Safetensors export. Its final M5
  development result was 99.0% Thinking and 72.0% Non-thinking accuracy, with no visible
  reasoning leakage.
- Seven failure classes are covered by versioned CPU fault injection and production-shared
  guards: OOM, non-finite values, corrupt checkpoints, insufficient disk, dataset drift,
  wrong world size, and child-process exit.

All results retain pinned model/data revisions, resolved configuration hashes, clean Git
commits, software/hardware snapshots, checkpoint hashes, raw-result hashes, and explicit
Thinking-controller intervention counts. M5 uses a private development suite; M6 owns the
independent release evaluation and Candidate promotion decision. Both trained models remain
in Development until that gate completes.
