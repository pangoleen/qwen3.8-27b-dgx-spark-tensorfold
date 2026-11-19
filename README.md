# Qwen3.8-27B on one DGX Spark: TensorFold native

Serve **Qwen3.8-27B** on a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API with
[TensorFold](https://github.com/ashhart/TensorFold)'s native engine, **v1.2.9**: RadixArk's NVFP4 weights, incoai's
DFlash2 drafter, an FP8 KV cache (8 requests at once at the full 262,144-token window) and prefix reuse at message
starts. Every setting below was measured on one Spark against the alternatives listed in
[What was tested](#what-was-tested).

```bash
git clone <this repo> && cd Qwen3.8-27B-DGX-Spark-TensorFold-Native
./start.sh          # installs TensorFold 1.2.9 and the pinned weights on first run, then serves
./stop.sh
```

The API is at `http://<spark>:8890/v1`, model id `qwen3.8-27b`.

## What it runs

| Item | Value |
| --- | --- |
| Engine | TensorFold native 1.2.9 release archive (`tensorfold-1.2.9-linux-aarch64`, sha256-checked), sm_121 kernels built in, no container |
| Weights | [`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) @ `554ebba9` (NVFP4 MLP and head, FP8 attention and DeltaNet), 20.8 GiB |
| Drafter | [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) @ `dedf8df6`, TensorFold's default drafter precision |
| KV cache | `--kv-cache fp8`: 9.09 GiB a stream at 262,144 tokens, 8 streams at once (bf16: 4) |
| Prefix reuse | `TF_QWEN_REQUEST_STARTS=1`: prompt passes cut at message starts, so an agent's next turn resumes at its own history |
| Context | 262,144 tokens a request |
| Thinking | off by default (`THINKING=1` turns it on); all measurements below are with it off |

Requirements: a DGX Spark (or another GB10 with 128 GB), NVIDIA driver R580 or newer, about 25 GB of disk for the
weights. Settings live in [`scripts/config.sh`](scripts/config.sh); override them from the environment or a `.env` file.

## Performance

One DGX Spark, 2026-11-18, thinking off, temperature 0 for the speed benches. The comparison column is a tuned SGLang
build on the **same weights and drafter** (RadixArk NVFP4 + incoai DFlash2, 16 draft tokens, 4 requests at once),
measured the same day with the same tools.

**Decode** ([sparkDash](https://github.com/MiaAI-Lab/sparkDash) collectors, 400 tokens), tok/s

| | This recipe | SGLang, same weights |
| --- | ---: | ---: |
| Prose, 1 request | **52.7-54.9** | 40.8 |
| Prose, 4 requests (total) | **146.7-152.8** | 105.1 |
| Code, 1 request | **128.8-137.2** | 123.1 |
| Code, 4 requests (total) | 367.8-394.0 | **387.7** |

**Long-prompt decode** (ctxsweep: a long prompt plus a summarize task, 256 tokens), tok/s

| Prompt | This recipe | SGLang |
| --- | ---: | ---: |
| 4k | **82.4** | 62.3 |
| 16k | **82.4** | 69.2 |
| 32k | **76.7** | 72.1 |

**Prefill** (sparkDash), tok/s

| Prompt | This recipe | SGLang |
| --- | ---: | ---: |
| 8k | 1,380 | **2,206** |
| 16k | 1,348 | **2,240** |
| 32k | 1,250 | **2,119** |
| 64k | 1,086 | **1,770** |

**Agent task** ([pi](https://github.com/badlogic/pi-mono) coding agent: a main agent launches 4 sub-agents to fix 8
bugs across 4 files, then verifies; temperature 0.8; one fixed working directory, one warm-up run not counted)

| | This recipe | SGLang |
| --- | ---: | ---: |
| Wall time, median of 6 (mean) | 75.2 s (83.2) | **61.4 s** (63.9) |
| Runs with all 8 checks passed | 6 of 6 | 6 of 6 |

In short: TensorFold decodes faster, by 30% for one prose request and up to 40% at four; SGLang prefills about 60%
faster, because its NVFP4 prompt path runs FP4 matmuls while TensorFold 1.2.9 runs NVFP4 prompts in BF16. Agent
traffic is prefill-heavy (each turn resends its history; about 85% of it comes from cache here), so on the agent task
SGLang still finishes about 18% sooner (this recipe takes 22% longer). For chat, single-user coding and long generations this recipe is the faster
one; for many short agent turns, SGLang is.

## What was tested

Each line changed one thing from the line above or from the recipe; agent-task medians over 3-9 runs.

| Variant | Agent task | Verdict |
| --- | --- | --- |
| TensorFold 0.6.0 (MiaAI's [27B recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold) engine), these weights | 124 s mean | the old engine keeps 3 prompt states for concurrent requests; most agent turns prefilled from zero |
| 1.2.9, bf16 KV | 90 s mean | 4 streams at 262k |
| 1.2.9, **fp8 KV** | 88 s median | 8 streams, same speed: kept |
| 1.2.9, fp8 KV, `TENSORFOLD_PREFILL_POLICY=fcfs`, 1,024-row chunks | 88 s | does not refine prefix reuse (still a 4,096-row grid); slower on MLX: dropped |
| 1.2.9, fp8 KV, **`TF_QWEN_REQUEST_STARTS=1`** | **68 s median** (new dir per run) | turns resume at their own history; kept |
| MLX 4-bit (`Vontra/Qwen3.8-27B-MLX-4bit`), bf16 prompts | 84 s median | 6-23% faster decode (16.1 GiB a token against 20.8), slower prefill; quality unmeasured: dropped |
| MLX 4-bit, `--prompt-precision int8` | 125 s mean | no prefill gain on GB10; 3 of 6 agent runs ran away: dropped |
| z-lab DFlash2 drafter instead of incoai | same as incoai | no difference measured |

## Known limits

- **Prefill** is the gap to SGLang. TensorFold 1.2.9's ModelOpt NVFP4 path takes full-precision prompts only
  (`--prompt-precision int8` applies to MLX affine weights), at about 88 TFLOPS.
- **Identical prompts that arrive together** each prefill their shared start: four sub-agents launched in the same
  instant are admitted together, before any of them has stored the shared system prompt.
- **Replies can differ** from the same server without `TF_QWEN_REQUEST_STARTS` (the chunk boundaries move); resumed
  and fresh replies match each other.
- The agent itself sometimes takes a long path (about 1 run in 8 here, on either engine): pi's `subagent` tool has a
  complex schema, and the model occasionally loops on it.

## How it was measured

Speed: sparkDash's own `DecodeBench` and `PrefillBench` collectors (temperature 0), and `ctxsweep.py` (prompts
calibrated to the server's tokenizer, a unique first line per run so nothing is cached). Agent task: pi through a
recording proxy, 8 automated checks. All runs on one Spark, with nothing else on the GPU.

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart; [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
by Qwen; the NVFP4 export by [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4); the DFlash2 drafter by
[incoai](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). [MiaAI-Lab](https://github.com/MiaAI-Lab)'s Spark recipes
and sparkDash showed the way.
