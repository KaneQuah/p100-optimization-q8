# P100 performance by card count

What this fork (`main` at `a3f475397`) does on Tesla P100s with one, two and four cards, for the models that fit each tier, and how
layer split, tensor split and expert split compare. The format follows
[wispoffates' PERFORMANCE.md](https://github.com/wispoffates/llama.cpp/blob/p100-series/docs/p100/PERFORMANCE.md), reorganised by
how many cards you have. Everything was measured in one session on 2026-11-04.

## Setup

- **Hardware:** 4x Tesla P100-PCIE-16GB (150 W cap, PCIe Gen3 x8 each) behind one PEX 8749 switch with P2P enabled (ACS off on the
  downstream ports); the switch uplink is a chipset Gen3 x4 link. Also present but not used: an RTX 4060 and a Radeon. Driver 580, CUDA 12.9.
  1 card = one P100, 2 cards = the first two, 4 cards = all four.
- **Builds:** "now" is this repo's `main` (`a3f475397`). "prev" is the same fork at the b11757 upstream merge (`dd6377725`), before the
  Kmic pull, mmvq-f16, fold attention, expert split and the other recent work. Prev has no expert split, so that mode has no prev column.
- **Command:** `llama-bench -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 512,2048 -p 512,2048 -n 128 -r 2`. pp512 is measured at `-ub 512`,
  pp2048 at `-ub 2048`, tg128 at `-ub 512`. Tokens per second, mean of 2 repetitions. GPUs are cooled to 60 C between runs.
- **Splits:** `layer` is `-sm layer`. `tensor` is `-sm tensor` with `LLAMA_SM_TENSOR_EP=0`. `tensor + expert split` is `-sm tensor` with
  `LLAMA_SM_TENSOR_EP=1` and applies to MoE models only (gemma-4 26B-A4B, Qwen3.6 35B-A3B, Qwen3-Coder-Next); dense models have no experts to split.
  The all-reduce is the default for the tier: our P2P all-reduce on 4 cards, NCCL on 2 cards.
- **Not measured:** serving (llama-server) numbers, accuracy (KL divergence) and long-context depth. Raw rows are in [`perf-data/`](perf-data/).

## Summary

| cards | best split for dense models | best split for MoE models | what to expect |
|---|---|---|---|
| 1 | single card | single card | 4B at 85 t/s down to a 27B IQ4_XS at 18 t/s; 26B-A4B MoE at 60 t/s |
| 2 | tensor (+35-37% decode, +6-7% prefill vs layer on 27B) | layer (tensor is 16-17% slower at decode on both MoE models) | tensor split pays off for dense models only |
| 4 | tensor (2.5x decode, 2.3x prefill vs layer on 27B) | tensor + expert split | 35B-A3B 1303 t/s prefill, 90 t/s decode; Coder-Next 80B 835 / 65 t/s |

- **Tensor split is the right choice for dense models on 2 and 4 cards.** On 4 cards 27B Q8_0 goes from 17.5 t/s decode with layer split to
  45.5 t/s.
- **For MoE models the answer depends on the card count.** On 2 cards layer split is fastest at decode for both gemma-4 26B-A4B (60.3 vs 50.4 t/s) and
  35B-A3B (76.1 vs 63.5 t/s): the experts are small and the all-reduce cost dominates. On 4 cards tensor split wins over layer split
  (35B-A3B +17% decode and +39% prefill, Coder-Next +10% and +38%), and expert split adds another +22% (35B-A3B) to +34% (Coder-Next) prefill and +3-5% decode.
- **Expert split does not help on 2 cards.** It is neutral for gemma-4 and gives +0.9-1.6% on 35B-A3B, which is inside run-to-run noise;
  the default (`auto`) only engages it on 4 cards.
- **Versus the previous build** decode is within about 1% everywhere except where expert split adds to it, and 27B Q6_K layer split on
  4 cards reads 4% lower (18.5 -> 17.7; the layer path is not touched by this work, so I treat it as noise until re-measured).

## One card (16 GB)

Models up to about 13 GiB of weights fit with a q8_0 KV cache. The 27B IQ4_XS is the largest dense model that fits. Layer, tensor and expert split are all the same thing on one card.

| model | size GiB | split | pp512 | pp2048 | tg128 | tg128 prev | tg vs prev | pp2048 vs prev |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B-Q4_K_M | 2.5 | single card | 1242.8 | 1375.2 | 85.3 | 84.9 | +0.6% | +0.6% |
| Qwen3-8B-Q4_K_M | 4.7 | single card | 683.9 | 698.4 | 63.9 | 63.6 | +0.4% | +0.6% |
| Qwen3-8B-Q8_0 | 8.1 | single card | 731.7 | 705.8 | 53.5 | 53.4 | +0.1% | +0.1% |
| gemma-4-26B-A4B-it-UD-IQ4_XS | 12.6 | single card | 668.4 | 759.9 | 60.4 | 60.7 | -0.5% | -0.1% |
| Qwen3.8-27B-UD-IQ4_XS | 13.3 | single card | 200.1 | 218.7 | 18.3 | 18.2 | +0.6% | +0.0% |

## Two cards (32 GB)

Models up to about 27 GiB of weights. Layer split puts half the layers on each card; tensor split shards every layer across both.

| model | size GiB | split | pp512 | pp2048 | tg128 | tg128 prev | tg vs prev | pp2048 vs prev |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gemma-4-26B-A4B-it-UD-IQ4_XS | 12.6 | layer | 681.2 | 774.1 | 60.3 | 60.7 | -0.6% | -0.0% |
| gemma-4-26B-A4B-it-UD-IQ4_XS | 12.6 | tensor | 602.4 | 687.3 | 50.4 | 50.4 | +0.2% | +0.0% |
| gemma-4-26B-A4B-it-UD-IQ4_XS | 12.6 | tensor + expert split | 602.6 | 686.2 | 50.4 | 50.4 | +0.2% | -0.1% |
| Qwen3.8-27B-UD-Q6_K | 20.5 | layer | 214.0 | 227.0 | 17.9 | 17.8 | +0.2% | +0.4% |
| Qwen3.8-27B-UD-Q6_K | 20.5 | tensor | 231.9 | 242.5 | 24.2 | 24.1 | +0.4% | +0.1% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | layer | 741.8 | 863.7 | 76.1 | 75.8 | +0.4% | -0.5% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | tensor | 717.6 | 837.2 | 63.5 | 63.0 | +0.8% | -0.3% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | tensor + expert split | 733.3 | 847.0 | 64.0 | 63.0 | +1.6% | +0.9% |
| Qwen3.8-27B-Q8_0 | 27.0 | layer | 222.3 | 228.8 | 17.6 | 17.6 | +0.1% | +0.3% |
| Qwen3.8-27B-Q8_0 | 27.0 | tensor | 236.3 | 243.8 | 24.0 | 24.0 | +0.1% | +0.0% |

## Four cards (64 GB)

Models up to about 45 GiB of weights. This is the tier the tensor-split and expert-split work targets.

| model | size GiB | split | pp512 | pp2048 | tg128 | tg128 prev | tg vs prev | pp2048 vs prev |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-UD-Q6_K | 20.5 | layer | 215.2 | 228.3 | 17.7 | 18.5 | -4.2% | +0.3% |
| Qwen3.8-27B-UD-Q6_K | 20.5 | tensor | 504.9 | 531.7 | 44.6 | 44.5 | +0.2% | -0.2% |
| Qwen3.8-27B-Q8_0 | 27.0 | layer | 218.0 | 228.8 | 17.5 | 17.5 | -0.2% | -0.2% |
| Qwen3.8-27B-Q8_0 | 27.0 | tensor | 515.4 | 534.8 | 45.5 | 45.5 | -0.0% | -0.3% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | layer | 745.8 | 863.3 | 74.4 | 74.6 | -0.3% | -0.9% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | tensor | 1036.3 | 1203.4 | 87.2 | 87.2 | -0.0% | -0.3% |
| Qwen3.6-35B-A3B-UD-Q5_K_S | 23.2 | tensor + expert split | 1303.0 | 1479.0 | 90.1 | 87.2 | +3.3% | +22.5% |
| Qwen3-Coder-Next-Q4_K_M | 45.2 | layer | 441.2 | 554.5 | 56.5 | 56.8 | -0.5% | +0.2% |
| Qwen3-Coder-Next-Q4_K_M | 45.2 | tensor | 621.5 | 765.8 | 61.9 | 62.1 | -0.4% | -0.3% |
| Qwen3-Coder-Next-Q4_K_M | 45.2 | tensor + expert split | 834.9 | 1028.5 | 65.2 | 62.1 | +5.0% | +34.0% |

Column notes: "tg128 prev" and "pp2048 vs prev" compare with the b11757 build in the same split mode (the expert split rows compare with the prev tensor row).
Differences under about 1% are run-to-run noise.

## Observations

- **Tensor split on 2 cards costs decode speed on MoE models.** gemma-4 26B-A4B: layer 60.3 t/s, tensor 50.4 t/s, single card 60.4 t/s. 35B-A3B: layer 76.1, tensor 63.5.
  If a model fits on one card, 2 cards add nothing for it with layer split and lose speed with tensor split; use the second card for something else.
- **Dense models on 2 cards:** tensor split is +35-37% decode over layer split (24.2 vs 17.9 on 27B Q6_K, 24.0 vs 17.6 on Q8_0) with only the 2 cards,
  and on 4 cards that grows to 2.5x (45 t/s).
- **Prefill scales less than decode with cards for the MoE models**, and layer split prefill on a MoE is flat from 2 to 4 cards (35B-A3B 864 -> 863 t/s at pp2048):
  layer split runs the cards one after the other, so more cards only add memory.
- **Quant choice on 4 cards:** the 27B at Q8_0 decodes slightly faster than at Q6_K (45.5 vs 44.6 t/s) and prefills the same within 1%, so with four cards the higher quant costs nothing in speed.

## Reproduce

`/scratch/build/perf-tiers.sh` (the script is in the maintainer's scratch space, not the repo) runs every cell; the exact command is in Setup above.
The tables here come from the CSVs in `perf-data/`, one file per (tier, build, split, expert-split, model).
