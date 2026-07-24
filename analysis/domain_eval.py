"""Per-domain held-out evaluation sets with ground-truth domain labels.

Each domain is streamed and packed *independently* (no cross-domain packing), so
every token in a produced sequence carries a single, unambiguous domain label.
"""
from __future__ import annotations
import torch

# (name, hf_dataset_kwargs, text_field). Same three sources as pre-training.
DOMAINS = [
    ("web",  dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True), "text"),
    ("edu",  dict(path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True), "text"),
    ("math", dict(path="open-web-math/open-web-math", split="train", streaming=True), "text"),
]
DOMAIN_NAMES = [d[0] for d in DOMAINS]
DOMAIN_ID = {d[0]: i for i, d in enumerate(DOMAINS)}


def _extract_text(example, field):
    return (example.get(field) or example.get("text") or example.get("content")
            or example.get("article") or "")


def build_domain_eval(tokenizer, max_seq_len, tokens_per_domain=200_000, skip_examples=2000):
    """Return {domain_name: list[dict(input_ids, labels, position_ids, domain_id)]}.

    `skip_examples` skips an initial slice of the stream to reduce overlap with
    what buffered-shuffle training is most likely to have consumed early. For a
    rigorous study, replace with a properly carved held-out split.
    """
    from datasets import load_dataset

    eos = tokenizer.eos_token_id
    assert eos is not None
    out = {}
    for name, kwargs, field in DOMAINS:
        stream = load_dataset(**kwargs)
        if skip_examples:
            stream = stream.skip(skip_examples)
        buf, seqs, ntok = [], [], 0
        for ex in stream:
            text = _extract_text(ex, field)
            if not text:
                continue
            ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            ids.append(eos)
            buf.extend(ids)
            while len(buf) >= max_seq_len + 1:
                window = buf[: max_seq_len + 1]
                seqs.append({
                    "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                    "labels": torch.tensor(window[1:], dtype=torch.long),
                    "position_ids": torch.arange(max_seq_len, dtype=torch.long),
                    "domain_id": DOMAIN_ID[name],
                })
                ntok += max_seq_len
                buf = buf[max_seq_len:]
            if ntok >= tokens_per_domain:
                break
        out[name] = seqs
        print(f"[domain_eval] {name:4s}: {len(seqs)} seqs ({ntok} tokens)")
    return out
