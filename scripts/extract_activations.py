#!/usr/bin/env python3
"""Extract residual-stream activations for the probe sets (GPU).

  python scripts/extract_activations.py --model Qwen/Qwen3-8B --variant matched
  python scripts/extract_activations.py --model /path/to/Qwen3-8B --variant matched --random_init   # control
  python scripts/extract_activations.py --model Qwen/Qwen3-8B --limit 64                             # smoke test

Reads data/ladder/<variant>/{qm9s,ladder}_records.jsonl (scripts/build_ladder.py)
and writes data/activations/<variant>/<model tag>/{qm9s,ladder}/ with
  last.npy, mean.npy   float16 [N, layers, d_model]
  ids.json, meta.json  row order; model, layers, an example of the exact prompt
A finished set is skipped on rerun. Disk: Qwen3-8B (37 x 4096) is 0.3 MB per
string per position, about 9 GB for the default probe set; --layer_step 2 halves it.
On an out-of-memory error the batch size is halved automatically."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import activations as A  # noqa: E402
from probe2circuit.io import load_config, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF id or local folder")
    ap.add_argument("--variant", default="matched")
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--ladder_dir", default=None, help="default data/ladder/<variant>")
    ap.add_argument("--out_dir", default=None, help="default data/activations/<variant>/<model tag>")
    ap.add_argument("--model_tag", default=None)
    ap.add_argument("--sets", default="ladder,qm9s")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--layer_step", type=int, default=1)
    ap.add_argument("--random_init", action="store_true", help="control: same architecture, untrained weights")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder to merge (fine-tuned model, step 3)")
    ap.add_argument("--no_chat", action="store_true", help="raw prompt, no chat template")
    ap.add_argument("--limit", type=int, default=None, help="first N records of each set (smoke test)")
    ap.add_argument("--fields", default="all", choices=["all", "posint", "pos"],
                    help="numbers per peak shown to the model: all = position, intensity, width; "
                         "posint = no width; pos = positions only. Adds -<fields> to the model tag.")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import torch
    cfg = load_config(a.config, a.variant)
    ldir = Path(a.ladder_dir or f"data/ladder/{a.variant}")
    tag = a.model_tag or (Path(a.model.rstrip("/")).name + ("-random" if a.random_init else "")
                          + ("-" + Path(a.adapter.rstrip("/")).name if a.adapter else "")
                          + ("" if a.fields == "all" else "-" + a.fields))
    out = Path(a.out_dir or f"data/activations/{a.variant}/{tag}")
    tok = model = None
    for s in [x for x in a.sets.split(",") if x]:
        recs = [r for r in read_jsonl(ldir / f"{s}_records.jsonl") if r.get("text")]
        if a.limit:
            recs = recs[:a.limit]
        dest = out / s
        if (dest / "meta.json").exists() and not a.force:
            meta = json.loads((dest / "meta.json").read_text())
            same = meta.get("text_hash") in (None, A.text_hash([r["id"] for r in recs], [r["text"] for r in recs],
                                                               a.fields))
            if meta.get("n") == len(recs) and same and meta.get("fields", "all") == a.fields:
                print(f"[acts] {dest}: already done ({meta['n']} prompts), skipped")
                continue
        if model is None:
            print(f"[acts] loading {a.model} ({'random init' if a.random_init else 'pretrained'}, {a.dtype})")
            tok, model = A.load_model(a.model, a.dtype, random_init=a.random_init, adapter=a.adapter)
            print(f"[acts] {model.config.num_hidden_layers} layers, d_model {model.config.hidden_size}, "
                  f"device {next(model.parameters()).device}")
        bs = a.batch_size
        while True:
            try:
                print(f"[acts] {s}: {len(recs)} prompts, batch {bs} -> {dest}")
                meta = A.extract(model, tok, [r["text"] for r in recs], cfg, dest, [r["id"] for r in recs], bs,
                                 a.layer_step, not a.no_chat, fields=a.fields,
                                 extra_meta={"model_arg": a.model, "random_init": a.random_init, "adapter": a.adapter,
                                             "variant": a.variant})
                break
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs //= 2
                torch.cuda.empty_cache()
                print(f"[acts] out of memory: retrying with batch {bs}")
        print(f"[acts] {s}: done in {meta['seconds'] / 60:.1f} min; longest prompt {meta['max_prompt_tokens']} tokens; "
              f"peak-list span median {meta['span_tokens_median']:.0f} tokens")
        print("[acts] the exact prompt of one record (also in meta.json):\n" + meta["example_prompt"][:600])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
