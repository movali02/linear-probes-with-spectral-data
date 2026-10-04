"""Residual-stream activations for the probes (roadmap 4.1).

Each record's string goes into the fixed prompt template, wrapped in the
model's chat template, and is run ONCE with no answer appended. Two positions
are saved for every layer (embeddings = layer 0):

  last   the final prompt token, where the answer would start
  mean   the mean over the tokens of the peak list itself

Arrays are float16 memmaps of shape [N, n_layers + 1, d_model], row order =
ids.json. torch / transformers are imported lazily: nothing else in the
package needs them."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

POSITIONS = ("last", "mean")


FIELDS = {"all": ("pos", "int", "fwhm"), "posint": ("pos", "int"), "pos": ("pos",)}
_FIELD_NAME = {"pos": "position cm-1", "int": "relative intensity", "fwhm": "FWHM cm-1"}


def reduce_fields(text: str, cfg: dict, fields: str = "all") -> str:
    """The same peak list with fewer numbers per peak: 'posint' drops the width,
    'pos' keeps positions only. Tests what a probe reads: two thirds of the
    tokens of a full string are intensity and width digits."""
    if fields == "all":
        return text
    from .strings import _fmt, parse_string
    s = cfg["serialise"]
    keep = FIELDS[fields]
    out = []
    for p in parse_string(text, cfg):
        parts = [_fmt(p["position"], s["pos_decimals"])]
        if "int" in keep:
            parts.append(_fmt(p["intensity"], s["int_decimals"]))
        out.append(s["field_sep"].join(parts))
    return s["peak_sep"].join(out)


def template_for(cfg: dict, fields: str = "all") -> str:
    """The prompt template with the description of the fields matched to `fields`."""
    t = cfg["prompt_template"]
    if fields == "all":
        return t
    full = "(" + ", ".join(_FIELD_NAME[f] for f in FIELDS["all"]) + ")"
    if full not in t:
        raise ValueError("prompt_template no longer contains the field description; update activations.template_for")
    return t.replace(full, "(" + ", ".join(_FIELD_NAME[f] for f in FIELDS[fields]) + ")")


def build_prompt(text: str, cfg: dict, tokenizer=None, chat: bool = True,
                 fields: str = "all") -> tuple[str, tuple[int, int]]:
    """Rendered prompt and the character span of the peak list inside it.
    `text` is the full string of the record; `fields` reduces it first."""
    text = reduce_fields(text, cfg, fields)
    user = template_for(cfg, fields).format(text=text)
    s = user
    if chat and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": user}]
        try:        # Qwen3: no <think> block content; the template still closes it
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    start = s.index(text)
    return s, (start, start + len(text))


def load_model(name: str, dtype: str = "bfloat16", device: str | None = None, random_init: bool = False,
               adapter: str | None = None, seed: int = 0):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    if device == "cpu" and td != torch.float32:
        td = torch.float32
    if random_init:      # control: same architecture, untrained weights (probe reads the input, not the model)
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name)).to(td)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=td)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.to(device).eval()


def text_hash(ids, texts, fields: str = "all") -> str:
    """Fingerprint of what was extracted, so a rebuilt ladder is never mistaken for a finished one."""
    import hashlib
    h = hashlib.sha1(fields.encode())
    for i, t in zip(ids, texts):
        h.update(f"{i}\t{t}\n".encode())
    return h.hexdigest()[:16]


def extract(model, tok, texts: list[str], cfg: dict, out_dir, ids: list[str], batch_size: int = 16,
            layer_step: int = 1, chat: bool = True, max_tokens: int = 4096, log=print, extra_meta=None,
            fields: str = "all") -> dict:
    """Run every prompt once and write last.npy / mean.npy / ids.json / meta.json to out_dir."""
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                       # the last prompt token is then index -1 for every row
    use_chat = chat and bool(getattr(tok, "chat_template", None))
    prompts, spans = zip(*[build_prompt(t, cfg, tok, chat, fields) for t in texts])
    n_layers = int(model.config.num_hidden_layers) + 1
    layers = list(range(0, n_layers, layer_step))
    if layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)
    d = int(model.config.hidden_size)
    N = len(prompts)
    mm = {p: np.lib.format.open_memmap(out_dir / f"{p}.npy", mode="w+", dtype=np.float16, shape=(N, len(layers), d))
          for p in POSITIONS}
    order = np.argsort([-len(p) for p in prompts])  # longest first: an out-of-memory error shows at once
    dev = next(model.parameters()).device
    t0, n_tok_max, n_span = time.time(), 0, []
    with torch.no_grad():
        for b0 in range(0, N, batch_size):
            idx = order[b0:b0 + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, return_offsets_mapping=True,
                      add_special_tokens=not use_chat)
            off = enc.pop("offset_mapping")
            if enc["input_ids"].shape[1] > max_tokens:
                raise ValueError(f"prompt of {enc['input_ids'].shape[1]} tokens exceeds --max_tokens {max_tokens}")
            n_tok_max = max(n_tok_max, int(enc["input_ids"].shape[1]))
            st = torch.tensor([spans[i] for i in idx])
            m = (off[..., 0] >= st[:, :1]) & (off[..., 1] <= st[:, 1:]) & (off[..., 1] > off[..., 0]) \
                & enc["attention_mask"].bool()
            n_span += m.sum(1).tolist()
            enc = {k: v.to(dev) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True).hidden_states
            md = m.to(dev).unsqueeze(-1)
            cnt = md.sum(1).clamp(min=1)
            for k, L in enumerate(layers):
                h = hs[L].float()
                mm["last"][idx, k] = h[:, -1].cpu().numpy().astype(np.float16)
                mm["mean"][idx, k] = ((h * md).sum(1) / cnt).cpu().numpy().astype(np.float16)
            if (b0 // batch_size) % 20 == 0:
                done = min(N, b0 + batch_size)
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {done}/{N} prompts  {rate:.1f}/s  ETA {(N - done) / max(rate, 1e-6) / 60:.1f} min")
    for a in mm.values():
        a.flush()
    k = int(order[0])
    meta = {"model": getattr(model.config, "_name_or_path", ""), "n": N, "layers": layers, "d_model": d,
            "positions": list(POSITIONS), "dtype": "float16", "chat_template": use_chat,
            "max_prompt_tokens": n_tok_max, "span_tokens_min": int(min(n_span)) if n_span else 0,
            "span_tokens_median": float(np.median(n_span)) if n_span else 0,
            "example_prompt": prompts[k], "example_span": prompts[k][spans[k][0]:spans[k][1]],
            "seconds": round(time.time() - t0, 1), "schema": cfg.get("schema"), "fields": fields,
            "text_hash": text_hash(ids, texts, fields), **(extra_meta or {})}
    (out_dir / "ids.json").write_text(json.dumps(list(ids)))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


class Acts:
    """Saved activations of one record set."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.ids = json.loads((self.path / "ids.json").read_text())
        self.row = {i: k for k, i in enumerate(self.ids)}
        self.layers = self.meta["layers"]
        self._mm = {}

    def get(self, position: str, layer_index: int, ids=None) -> np.ndarray:
        """[n, d_model] float32 for one position and one saved layer (index into self.layers)."""
        if position not in self._mm:
            self._mm[position] = np.load(self.path / f"{position}.npy", mmap_mode="r")
        a = self._mm[position]
        rows = np.arange(a.shape[0]) if ids is None else np.array([self.row[i] for i in ids])
        return np.asarray(a[rows, layer_index], dtype=np.float32)
