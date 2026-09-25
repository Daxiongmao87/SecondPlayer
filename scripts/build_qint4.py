"""Build a shipped quanto qint4 dir from a float checkpoint.

Pipeline: load fp -> optional bf16 cast -> int4 linears + int8 embeddings ->
freeze -> save safetensors + tokenizer + config -> write the head_rows.pt
slot sidecar (the scorer serves letter slots from it instead of ever
unpacking the 248k-row head) -> reload and smoke-score one row.

Example:
    python scripts/build_qint4.py --model Qwen/Qwen3.5-4B --save \
        /data/tmp/ojcore/qwen3.5-4b-qint4e --source-dtype fp32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    from quanto import freeze, qint4, quantize, safe_save

    from secondplayer.policy.ojcore import load_model, load_tokenizer, score
    from secondplayer.policy.ojcore.embed import quantize_embeddings
    from secondplayer.policy.ojcore.scoring import HEAD_ROWS_FILE, slot_ids

    t0 = time.time()
    model, _ = load_model(args.model, args.device, quant=None, offline=True)
    if args.source_dtype == "bf16":
        model.to(torch.bfloat16)
    quantize(model, weights=qint4, activations=None)
    n_embed = quantize_embeddings(model)
    freeze(model)
    os.makedirs(args.save, exist_ok=True)
    safe_save(model.state_dict(), os.path.join(args.save, "model.safetensors"))
    tokenizer = load_tokenizer(args.model, offline=True)
    tokenizer.save_pretrained(args.save)
    cfg = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    cfg.save_pretrained(args.save)
    print(f"quantized linears + {n_embed} embedding(s) in {time.time() - t0:.0f}s", flush=True)

    ids = slot_ids(tokenizer, args.slots)
    weight = model.lm_head.weight
    full = weight.dequantize() if hasattr(weight, "dequantize") else weight.float()
    torch.save(
        {"ids": list(ids), "rows": full[ids].to("cpu").contiguous()},
        os.path.join(args.save, HEAD_ROWS_FILE),
    )
    del model
    print(f"wrote {HEAD_ROWS_FILE} with {len(ids)} slot rows", flush=True)

    tok2 = load_tokenizer(args.save, offline=True)
    model2, compute2 = load_model(args.save, args.device, quant="quanto_qint4", offline=True)
    row = {
        "id": "movement",
        "state": "Game: probe.\nScreen: a paddle near the bottom, a ball above it.",
        "question": "Which direction should the player hold on the controller?",
        "options": [
            {"id": n, "description": d}
            for n, d in [
                ("neutral", "Hold no direction."),
                ("up", "Hold the UP direction."),
                ("down", "Hold the DOWN direction."),
                ("left", "Hold the LEFT direction."),
                ("right", "Hold the RIGHT direction."),
            ]
        ],
    }
    t1 = time.time()
    result = score(model2, tok2, row)
    probs = result["probabilities"]
    top = max(range(len(probs)), key=probs.__getitem__)
    print(
        json.dumps(
            {
                "device": compute2.device,
                "smoke_s": round(time.time() - t1, 2),
                "argmax": result["option_ids"][top],
                "top_prob": round(probs[top], 4),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="float source (hub id or dir)")
    parser.add_argument("--save", required=True, help="output quanto dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-dtype", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--slots", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    main()
