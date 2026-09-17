# Exploration Threads

Threads for optimizing, extending, and experimenting with TRL. Each thread is an area of work that can be explored independently.

---

## 1. Kernel coverage across trainers (red)

TRL benchmarked Hub kernels (Flash Attention variants) and Liger fused losses only on SFT with a single GPU config (H100, Qwen3-8B). No systematic testing exists for DPO, KTO, GRPO, RLOO, Distillation, or Reward trainers. The gap is wide: which kernels actually help each trainer, on which hardware, and where are the incompatibilities?

**What's been done:** SFT + FA2/FA3 benchmarks documented in `docs/source/kernels_hub.md`. Liger fused losses vendored into `trl/losses/` for DPO, KTO, GRPO, JSD. Combining FA + Liger documented but not benchmarked per-trainer.

**What hasn't:** Per-trainer kernel benchmarks across GPU tiers. Testing non-attention kernels (fused normalization, fused activations). Custom Triton kernels for trainer-specific hotspots like advantage computation in GRPO/RLOO. Paged attention for training workloads. **Flash Attention 4 is completely absent** — `FLASH_ATTENTION_VARIANTS` in `sft_trainer.py:393` only lists FA2/FA3. This gates padding-free training and packing, so FA4 is actively blocked even if transformers supports it.

Optional things:

1. Should also be repro'd: The existing benchmarks for the kernels_hub.md kernel implementations for DPO/KTO/GRPO/JSD
2. Potentially new kernels can be developed for OPD or any of the experimental trainers (can be checked)

---

## 2. Strip telemetry, model cards, and logging integrations from the base (yellow)

`_BaseTrainer` phones home to HF Hub on every init (`_send_telemetry`), generates model cards on save, and every trainer individually wires up wandb/comet/trackio/weave. For a personal fork this is dead weight — ~200 lines across `base_trainer.py` and `utils.py`, plus conditional imports in every trainer file. Removing it simplifies the base class and cuts init time.

**What's there:** `_send_telemetry` in `base_trainer.py:78`, `create_model_card` in `base_trainer.py:140`, `generate_model_card` + `log_table_to_comet_experiment` + `get_comet_experiment_url` + `get_trackio_space_url` in `utils.py`. Wandb/comet/trackio imports scattered across GRPO, RLOO, Distillation, callbacks.

**Why it matters:** Cleaner base class, faster startup, no accidental data leaks to third-party services.

---

## 3. Kill pandas in the training hot path (no, we might wanna do benchmarks for before and after on this.) (yellow)

`pandas` is imported at module top-level in `grpo_trainer.py`, `rloo_trainer.py`, `callbacks.py`, and `utils.py`. It's only used for logging completion tables and `LogCompletionsCallback` — not for any computation. Replacing it with plain dicts/lists or lightweight table formatting eliminates a heavy import and simplifies the dependency surface.

**Impact:** `import pandas` alone takes ~300ms. Every trainer that imports it pays that cost even if completions logging is off.

---

## 4. torch.compile coverage in trainers (Maybe it's for a good reason? I don't think this is an area of attack)

`torch.compile` is used only inside `trl/losses/` (the vendored Liger fused losses). None of the trainer `compute_loss` or `training_step` methods use it. There's an opportunity to compile hot functions — the per-token log-prob loop in GRPO (`_get_per_token_logps`), the advantage normalization, the KL divergence computation — without touching the generation path.

**Risk:** Dynamic shapes from variable-length completions can cause recompilation storms. Needs careful profiling and `dynamic=True` marking.

---

## 5. Generation backend abstraction (red, needs more attacking. genuinely wonder if we can make almost a clone of TRL but use the vllm backend to do all the forward passes and compare.)

Currently only vLLM is supported as an external generation backend (`trl/generation/vllm_generation.py`, `vllm_client.py`). The `VLLMGeneration` class is directly instantiated in GRPO, RLOO, and Distillation trainers. There's no pluggable interface — adding SGLang, TensorRT-LLM, or a custom backend means forking each trainer.

**Opportunity:** Define a `GenerationBackend` protocol and make the trainers backend-agnostic. The existing `DistributedBackend` in `trl/distributed.py` is a good pattern to follow — small class, uniform API, backend detection at init.

---

## 6. GRPO is 3500 lines — decompose it (nah, it's fine)

`grpo_trainer.py` is 3498 lines, nearly 2x the next largest trainer (Distillation at 1996). It handles generation, reward scoring, advantage computation, KL estimation, completion logging, vLLM lifecycle, continuous batching, and the actual policy gradient — all in one file. This makes it hard to modify any single piece without understanding the whole thing.

**Approach:** Extract generation + scoring into a composable pipeline. Keep the loss computation in the trainer. This also enables reuse — RLOO (1863 lines) shares much of the generation/scoring logic.

---

## 7. Custom trainer backends beyond HF Trainer (red, should be tried)

All TRL trainers inherit `transformers.Trainer` via `_BaseTrainer`. This gives you the HF training loop, but also locks you into its checkpointing, evaluation, logging, and scheduling assumptions. For experiments with novel training loops (async RL, multi-agent, curriculum learning), the Trainer abstraction can fight you.

**Idea:** Build a minimal `TrainingLoop` base that owns only the forward/backward/step cycle and delegates everything else to composable hooks. Keep the HF Trainer path as one implementation. This is the most ambitious thread — start it last.

---

## 8. Dependency trimming (eh, it's fine but i think trl already has very few dependencies. i'd rather trim down on the code and think deeply about the dependenies later)

The core has only 5 deps (accelerate, datasets, transformers, jinja2, packaging) which is lean. But the optional extras carry dead weight: `bco` (scikit-learn for a removed trainer), `harbor`, `kernels` (passthrough to transformers), `scikit` (standalone duplicate), `vlm` (num2words pinned to a single version). The `dev` extra bundles everything including things that should be separate.

**Quick wins:** Remove `bco`, `scikit`, `kernels` extras. Unpin `num2words`. Clean up `dev` to only include what's actually needed for running tests on the stable trainers.

---

## 9. Import time optimization (again, it's a very minor thing imo. idk how much time we can realistically shave across devices)

TRL uses `_LazyModule` for top-level imports, which is good. But individual trainer files do eager top-level imports of heavy modules: `numpy`, `pandas`, `torch`, `transformers`, `accelerate`, `huggingface_hub`, `datasets` — all at import time. When you do `from trl import GRPOTrainer`, the lazy module defers it, but the moment you actually use the class, the full import chain fires.

**Explore:** Profile `python -X importtime -c "from trl import GRPOTrainer"` to identify the actual bottlenecks. Consider making pandas, wandb, comet, and logging deps lazy within the trainer files themselves (import inside the function that uses them, not at module top).

---

## 10. Canonical trainer set

Trimming trainers to a focused set. Stable distillation gets cut. Experimental goes from 22 trainers to 13 (8 keep + 5 study).

### Stable — keep (6):
- **SFT** — supervised fine-tuning, foundation of everything
- **DPO** — offline preference optimization from paired data
- **GRPO** — online RL with group-relative advantages (DeepSeek)
- **RLOO** — online RL with REINFORCE leave-one-out baseline
- **KTO** — unpaired preference (thumbs up/down, no pairs needed)
- **Reward** — train reward models from preference data

### Stable — remove (1):
- **Distillation** — cutting the whole stable distillation path

### Experimental — keep (8):
- **Async GRPO** — decouples generation from training via async vLLM server
- **BEMA** — DPO override for reference model handling
- **GMPO** — GRPO variant with geometric mean importance ratios, just overrides `_compute_loss`
- **GOLD** — on-policy logit distillation, extends SFTTrainer with teacher model
- **Nash MD** — game-theoretic alignment, extends OnlineDPOTrainer
- **Online DPO** — DPO with online generation and reward scoring
- **ORPO** — no reference model needed, combines SFT + odds-ratio preference
- **XPO** — exploratory preference optimization, extends OnlineDPOTrainer

### Experimental — study (5):
- **A2PO** — two-stage: offline value estimation then on-policy advantage regression (2025 paper)
- **BCO** — binary classifier alignment, like KTO but learned classifier (needs scikit-learn)
- **CPO** — contrastive preference optimization, DPO without reference model (overlaps ORPO?)
- **SDFT** — self-distillation for continual learning, student is own teacher (2026 paper)
- **TPO** — token-level preference optimization

### Experimental — remove (11):
- **Async Distillation** — distillation path cut
- **GKD** — redundant with GOLD
- **GSPO** — just a token-level GRPO variant override
- **IW OPD** — importance-weighted offline policy distillation, niche
- **MiniLLM** — extends GRPO for knowledge distillation, distillation path cut
- **PRM** — process reward models, better served by different abstractions
- **SDPO** — step-level DPO, niche variant
- **Server Distillation** — distillation path cut
- **SSD** — self-play distillation, niche
