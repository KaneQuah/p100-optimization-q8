# P100 fork: P2P tensor split on 4x Tesla P100

This fork of llama.cpp makes `--split-mode tensor` fast on four Tesla P100-PCIE-16GB cards (sm_60, CUDA 12.9). The cards sit behind one PCIe switch on Gen3 x8 links: no NVLink, no NCCL support for Pascal, no dp4a, no tensor cores. Layer split and single-GPU runs are kept fast too, but tensor split is the goal.

Based on upstream master 8f574b0ad (b11579, merged 2026-10-13).

## Why tensor split needs work on this hardware

Tensor split does an all-reduce after every attention and FFN block (128 per token on Qwen3.8-27B). Upstream has no usable all-reduce for Pascal: NCCL does not support it, the internal one needs Volta, so it falls back to a butterfly of small copies from one host thread. A GPU pulling from its peers gets only 5 to 6.6 GB/s here, so the all-reduce is limited by the PCIe links, not by waiting for slow GPUs. The work is to move fewer bytes per link, avoid host involvement, and make the compute kernels fast on sm_60.

## What is different from upstream

- **P2P all-reduce** (`ggml/src/ggml-cuda/allreduce-p2p.cu`): one kernel per GPU, no host sync. The first version (cf4422b4a, default on 4x P100 since 82e758c54) pulled whole tensors from the peers. 42f758187 made it a reduce-scatter + all-gather, so each link carries 1.5x the tensor instead of 3x: the all-reduce of a 10 MB prefill tensor went from 6.13 to 2.86 ms. Results are bit-identical to the butterfly. See [multi-gpu.md](multi-gpu.md) for the switches.
- **Decode kernels for sm_60**: MMVQ geometry, staging and vdr changes (Aug to Oct), IQ-type tuning (f60579bd8), fusion of small ops and same-input mat-vecs (ec9d83344), the gated-delta-net decode chain 14 -> 4 kernels per layer (074fa771a), and q8_0 K/V decode and verify flash attention (8f5aa0f70).
- **Batches of 9 to 32 columns** run as MMVQ passes of 8 (60e676bd7). Without dp4a they used to dequantize the whole matrix and call cuBLAS, which cost about 5x an 8-column step. Decode at 16 sequences on an 8B Q4_K_M went from 61 to 156 tok/s.
- **Prefill**: cuBLAS-GEMM flash attention for long KV (d537b1899), chunked gated-delta-net prefill for layer split (031b7b1a7), and MoE MMQ tile sized by tokens per expert (3f02696fe).

## Results

4x P100, `-sm tensor -fa 1 -ub 512`, q8_0 KV, llama-bench, GPUs cooled to 60 C or below before each run (GPU0 throttles near 78 C). Tokens per second.

| model | pp2048 stock | pp2048 now | tg128 stock | tg128 now |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Q8_0 | 284 | 514 | 22.1 | 47.0 |
| 35B-A3B Q5_K_S | 586 | 1020 | 49.3 | 93.6 |
| Coder-Next 80B Q4_K_M | 384 | 648 | 40.1 | 66.3 |

"Stock" is upstream 8f574b0ad measured on 2026-10-15; "now" is this fork at 60e676bd7 measured on 2026-10-17. Same machine and flags, different sessions, interleaved A/B runs within each.

## Using it

```
CUDA_VISIBLE_DEVICES=0,1,2,3 llama-server -m model.gguf -ngl 99 -sm tensor -fa on -ub 2048 -b 2048
```

- With tensor split, `-ub 2048` beats the default 512 for prefill (pp2048: 27B +6%, 35B-A3B +21%, 80B +20%) and leaves decode unchanged. For layer split keep `-ub 512`.
- The P2P all-reduce is on by default for exactly 4 P100. `GGML_CUDA_ALLREDUCE=none` turns it off. `GGML_CUDA_AR_P2P_BF16=1` is off by default (perplexity +0.02 to 0.25%, greedy output can change).
- Do not run tensor split with `CUDA_LAUNCH_BLOCKING=1`: the all-reduce kernels spin-wait on each other and deadlock.

## Limits

- Measured only on compute capability 6.0 (P100). cc 6.1 (P40, GTX 10xx) is untested and shares some of the tuning gates. The HIP build excludes the chunked MMVQ path and has not compiled it.
- On the 35B-A3B the expert FFNs run on only 2 of the 4 GPUs in each layer, because the K-quant block size (256) does not divide the expert width (512) in four, so the other two wait. The 80B is probably the same but was not traced.
- `--split-mode tensor` is still experimental upstream.
