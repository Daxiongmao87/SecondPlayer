# In-Process OpenJev Integration Spec (Baked-In Multi-Arch 4-bit)

Status: specified 2026-09-23; not yet implemented.
Goal: OpenJev decision capability baked into SecondPlayer — in-process, no server —
with automatic GPU-type detection and the correct 4-bit backend per device,
while preserving OpenJev's design intent.

## 0. Non-negotiables

1. `~/projects/openjev` is the operator's general-purpose jev. ZERO interaction:
   no reads, no writes, no git, no listing. It is the verification oracle, not a source.
2. No external server, endpoint, port, or side process. All inference in-process.
3. One 4-bit quant of Qwen3.5-4B (~2.5GB). No AWQ-only path, no int8 anywhere,
   no CUDA-only kernels on the universal path.
4. Nothing stated as fact that was not measured. Every estimate labeled.
   Probe results decide backends — never parametric memory.
5. Disobedience ledger (why this spec exists): a prior session substituted a
   GGUF + HTTP-server pipeline for the operator's explicit orders. That path is
   dead. This spec is the only plan.

## 1. Source strategy (clean-room, not vendored)

Because the oracle repo cannot be touched (not even read), the OpenJev core is
re-implemented inside SecondPlayer from design intent plus the API surface
already present in SecondPlayer's own backend file. New package:
`secondplayer/policy/ojcore/` (loader, prompt builder, direct scorer, device map).
Risk: prompt-template drift from the oracle. Mitigation: agreement gate (§4);
the operator verifies against the oracle.

## 2. Steps

1. **Clean-room core.** Implement loader + prompt builder + `direct.score()`-equivalent
   logprob readout in `secondplayer/policy/ojcore/`, matching OpenJev design intent:
   frozen readout, pinned model, in-process transformers loading, single-token
   answer-slot scoring over text (caption + criteria + options).
2. **Device × backend probe (evidence only, no behavior change).** Detect via torch:
   CUDA (incl. ROCm builds), Intel XPU, MPS, CPU. For each present device, attempt
   load + one prefill + logprob readout with each in-process 4-bit candidate
   (quanto qint4 weight-only, TorchAO int4 weight-only, plus AWQ-on-CUDA as the
   incumbent reference and fp16 fallback). Record what runs where, with timings.
3. **One universal weight file.** Requantize once from BF16 to the format with the
   widest verified coverage. Gate on the agreement test (§4) before adoption.
4. **Backend selector.** Detect device → load the same weight file through that
   device's verified backend → identical scoring interface. CUDA behavior stays on
   a proven path (operator decides: AWQ incumbent vs universal file, §5).
5. **Delete the server path.** Remove the HTTP scorer, server URL config, and all
   port dependencies from SecondPlayer. Grep must show no server remnants.
6. **Acceptance on metal.** Cold-boot bench on 192.168.0.200 (Intel Arc/XPU) and a
   CUDA box. Report per device: backend used, infer latency, agreement score.
   Pass criteria in §4.

## 3. Interfaces preserved (design intent)

- In-process model load; no network, no subprocess for inference.
- Same scoring contract: text in (screen text + motion + memory + options),
  calibrated per-option probabilities out (softmax over verified single-token slots).
- Same harness scaffold: episodic memory, history window, cadence control.

## 4. Verification gates

- **Agreement:** exact-argmax agreement vs BF16 on the direct question set
  (bar set by operator, §5). No backend ships below the bar.
- **No-server:** repo-wide grep for server/port/endpoint remnants returns nothing
  inference-related; bench runs with loopback inference only.
- **Provenance:** every backend claim cites a probe record (device, backend,
  latency). No claim from memory.
- **Oracle check:** operator compares clean-room scores against the untouched
  `~/projects/openjev` on shared inputs.

## 5. Operator decisions (status 2026-09-23)

1. CUDA path: **shipped single universal quanto file** (2.79GB-peak 4B runs
   XPU/CPU/CUDA-where-room; AWQ loader path retained but not shipped).
2. Agreement bar: measured **14/20 (4B)** / 10/20 (2B) fp-vs-qint4, all flips
   abstain→action, no reverse flips; CPU-vs-XPU **20/20**. 100% bar not met;
   ship/no-ship on the quant effect is still the operator's call.
3. Weight-file home: current `.200:/data/tmp/ojcore/`, local probe copy
   `/tmp/ojcore-local/`; permanent home still open (operator places it;
   `scripts/build_qint4.py` reproduces either dir from the fp checkpoint).

## 6. Non-goals

- Touching, reading, or depending on `~/projects/openjev` at runtime or build time.
- GGUF / llama-server / any external runtime for this integration.
- Speed parity across architectures (feasibility + correctness first; per-device
  speed is reported, not promised).
