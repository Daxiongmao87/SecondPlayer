"""Game-vision qualification: stock Qwen3.5-4B BF16, official processor.

No action selection, no quantization, no manual multimodal inputs.
Runs on the bench host (model cache + captures local). Offline.
"""
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/projects/secondplayer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

SP = os.path.expanduser("~/projects/secondplayer")
CAP = "/data/tmp/vision-cap"
MODEL = "Qwen/Qwen3.5-4B"

snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")))
snap += sorted(glob.glob("/data/tmp/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))
print("snapshots:", [s.split("/")[-1] for s in snap], flush=True)

t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=False)
print("processor:", type(proc).__name__, flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=False)
model.to("xpu").eval()
torch.xpu.synchronize()
print(f"loaded BF16 on xpu in {time.time() - t0:.1f}s", flush=True)


def resolve(key):
    if key.startswith("cap:"):
        return Image.open(f"{CAP}/{key[4:]}.png").convert("RGB")
    if key.startswith("fix:"):
        return Image.open(f"{SP}/tests/fixtures/tetris/{key[4:]}").convert("RGB")
    raise ValueError(key)


def transform(img, kind):
    if kind is None:
        return img
    if kind == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if kind == "scale2":
        return img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    if kind == "blank":
        return Image.new("RGB", img.size, (0, 0, 0))
    if kind == "shuffle":
        rng = np.random.default_rng(7)
        a = np.asarray(img)
        h, w, _ = a.shape
        ph, pw = 16, 16
        blocks = [a[y:y + ph, x:x + pw].copy()
                  for y in range(0, h, ph) for x in range(0, w, pw)]
        rng.shuffle(blocks)
        out = np.zeros_like(a)
        i = 0
        for y in range(0, h, ph):
            for x in range(0, w, pw):
                bh, bw = min(ph, h - y), min(pw, w - x)
                out[y:y + bh, x:x + bw] = blocks[i][:bh, :bw]
                i += 1
        return Image.fromarray(out)
    raise ValueError(kind)


def ask(images, question):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": question})
    text = proc.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    kw = proc(text=[text], images=images, return_tensors="pt")
    kw = {k: v.to("xpu") for k, v in kw.items() if isinstance(v, torch.Tensor)}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**kw, max_new_tokens=24, do_sample=False,
                             pad_token_id=proc.tokenizer.eos_token_id)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    gen = out[0][kw["input_ids"].shape[1]:]
    return proc.tokenizer.decode(gen, skip_special_tokens=True), ms


def parse_letter(raw, options):
    letters = {o[0] for o in options}
    m = re.search(r"\b([A-Z])\b", raw.upper())
    if m and m.group(1) in letters:
        return m.group(1)
    for c in raw.upper():
        if c in letters:
            return c
    return ""


manifest = json.load(open(f"{SP}/probes/2026-09-25-vision/vision-manifest.json"))
if os.environ.get("QOFFSET"):
    manifest = manifest[int(os.environ["QOFFSET"]):]
if os.environ.get("QLIMIT"):
    manifest = manifest[: int(os.environ["QLIMIT"])]
# Preflight: every image key must resolve before the expensive model load.
for q in manifest:
    for k in q["images"]:
        if k.startswith("cap:"):
            assert os.path.exists(f"{CAP}/{k[4:]}.png"), k
        elif k.startswith("fix:"):
            assert os.path.exists(f"{SP}/tests/fixtures/tetris/{k[4:]}"), k
        else:
            raise ValueError(k)
results = []
out_path = f"{SP}/probes/2026-09-25-vision/vision-bf16.jsonl"
fresh = not os.environ.get("QOFFSET") and not os.environ.get("QLIMIT")
if fresh and os.path.exists(out_path):
    os.remove(out_path)
import gc

for qi, q in enumerate(manifest):
    imgs = [transform(resolve(k), q["transform"]) for k in q["images"]]
    err: str | None = None
    try:
        raw, ms = ask(imgs, q["question"])
    except Exception:
        # One retry after releasing cached blocks (UR pinned-memory growth).
        gc.collect()
        torch.xpu.empty_cache()
        try:
            raw, ms = ask(imgs, q["question"])
        except Exception as exc2:
            raw, ms, err = f"ERROR {exc2}", -1.0, str(exc2)[:100]
    if err is None:
        if q["options"] == ["OCR"]:
            got = raw.strip().upper().replace(" ", "")
            ok: bool | None = got == q["answer"]
            ambiguous = False
        else:
            clean = raw.strip().upper().rstrip(".")
            ambiguous = clean not in {o[0] for o in q["options"]}
            got = parse_letter(raw, q["options"])
            ok = got == q["answer"]
    else:
        got, ok, ambiguous = "", None, False
    results.append({"id": q["id"], "cat": q["cat"], "ok": ok, "got": got,
                    "exp": q["answer"], "ms": round(ms, 1), "raw": raw[:400],
                    "pair": q["pair"], "control": q["control"], "error": err,
                    "ambiguous": ambiguous})
    with open(out_path, "a") as f:
        f.write(json.dumps(results[-1]) + "\n")
    tag = "PASS" if ok else ("ERROR" if ok is None else "FAIL")
    print(f"{q['id']} {tag} {q['cat']} exp={q['answer']} got={got} "
          f"{ms:.0f}ms :: {raw[:80]!r}", flush=True)
    if ok is None:
        print("infra error; aborting chunk for fresh retry", flush=True)
        sys.exit(1)
    del imgs
    torch.xpu.empty_cache()
    if qi % 10 == 9:
        gc.collect()

main = [r for r in results if not r["control"] and r["ok"] is not None]
errs = [r for r in results if r["ok"] is None]
ctl = [r for r in results if r["control"] and r["ok"] is not None]
print(f"== main {sum(r['ok'] for r in main)}/{len(main)} "
      f"control {sum(r['ok'] for r in ctl)}/{len(ctl)} errors {len(errs)} ==", flush=True)
by_cat: dict[str, list] = {}
for r in main:
    by_cat.setdefault(r["cat"], []).append(r["ok"])
for cat, oks in sorted(by_cat.items()):
    print(f"   {cat}: {sum(oks)}/{len(oks)}", flush=True)
lat = sorted(r["ms"] for r in main if r["ms"] >= 0)
if lat:
    print(f"   latency ms: p50={lat[len(lat)//2]:.0f} p95={lat[int(len(lat)*0.95)]:.0f} "
          f"max={lat[-1]:.0f}", flush=True)
pairs: dict[str, list] = {}
for r in results:
    if r["pair"]:
        pairs.setdefault(r["pair"], []).append(r)
both = sum(1 for pid, rs in pairs.items()
           if len(rs) == 1 and rs[0]["ok"] is True and
           next((x for x in results if x["id"] == pid), {"ok": None})["ok"] is True)
print(f"   mirror pairs both-correct: {both}/{len(pairs)}", flush=True)
