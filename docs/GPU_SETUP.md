# GPU bring-up notes (192.168.0.200)

Hardware: Intel Arc A770M (dGPU) + Alder Lake Iris Xe (iGPU). No NVIDIA.
`auto` resolves to `xpu` (Arc) with Laya inference p50 ~156ms fp32,
vs ~1000ms on CPU.

## Installed stack

- Intel GPU apt repo (`noble unified`) + matched **-950** set, all held
  (`apt-mark hold`): `libze1=1.17.6`, `intel-level-zero-gpu=1.3.30049`,
  `libigc1=1.0.17193`, `libigdfcl1=1.0.17193`,
  `intel-opencl-icd=24.26.30049`, `libigdgmm12=22.7.2`.
- venv: `torch==2.8.0+xpu` + `torchvision==0.23.0+xpu` from the
  `download.pytorch.org/whl/xpu` index.

## Why these exact versions

- Loader 1.21.x (repo max) segfaults torch's UR adapter against ICD 1.3.x;
  the matched -950 loader does not.
- OCL 25.18 needs libigc2, which conflicts with L0's libigc1; the -950 OCL
  matches. (oneDNN still needs the OCL ICD present or inference fails with
  "OpenCL device not found".)
- torch 2.14+xpu segfaults in `urAdapterGet` against ICD 1.3 (its UR 0.12
  needs `zeInitDrivers`, which the repo-max ICD lacks). torch 2.8.0+xpu
  works. Do not upgrade torch past 2.8 without re-testing XPU alloc +
  a Laya `decide()`.

## Gotchas

- Keep 5G+ free for torch swaps: `pip uninstall torch` first, install with
  `--no-cache-dir` (rollback: `torch==2.14.0+cpu` from the cpu index).
- `agent` needs the `render` group for `/dev/dri/renderD*` (already has it).
- `bench-tetris.toml` still pins `device="cpu"`; the CPU baseline is
  unaffected by any of this.
- The 1s `reaction_ms` floor now dominates cadence (156ms decisions wait
  out the second). Lowering it is a follow-up experiment, not done here.
