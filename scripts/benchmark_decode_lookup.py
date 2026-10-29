#!/usr/bin/env python3
"""Short offline greedy/prompt-lookup probe using the same local model + adapter.

This measures bounded generation prefixes, NOT complete layouts or a selector
quality comparison. Alternate arm order and synchronize the accelerator. No
production configuration is changed and no remote code is loaded.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", nargs="+", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/upgrade_exploration/prompt_lookup.json")
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 2:
        parser.error("Use positive tokens and at least two alternating repeats")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MAGICANVAS_QWEN_ENABLED"] = "1"
    os.environ["MAGICANVAS_REQUIRE_QWEN_ADAPTER"] = "1"
    from engine.qwen_seed import _load, MODEL_DIR, ADAPTER_DIR
    from engine.qwen_prompt import messages_for
    from engine.runtime import synchronize
    started = perf_counter()
    loaded = _load()
    if loaded is None:
        raise RuntimeError("Local model unavailable")
    model, tokenizer, torch, device, version = loaded
    load_seconds = perf_counter() - started
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope="Generation-prefix feasibility/timing probe; not layout quality or promotion evidence",
                  model=str(MODEL_DIR), adapter=str(ADAPTER_DIR), prompt_version=version,
                  runtime=getattr(model, "_magicanvas_runtime", {}), load_seconds=load_seconds,
                  tokens_limit=args.tokens, repeats=args.repeats, cases=[], production_changed=False)
    for path in args.audits:
        audit = json.loads(path.read_text())
        source, target = audit["source"], audit["target"]
        messages = messages_for(source, target, version=version)
        def encode(messages):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            return tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs = encode(messages)
        if len(source["elements"]) > 16 and inputs["input_ids"].shape[1] > 4096:
            messages = messages_for(source, target, version=version,
                                    relation_token_counter=lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]),
                                    relation_token_budget=900)
            inputs = encode(messages)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + args.tokens > 8192:
            raise ValueError("Probe would exceed production context ceiling")
        def generate(arm, limit):
            options = dict(max_new_tokens=limit, do_sample=False, use_cache=True,
                           repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
            if arm == "prompt_lookup":
                options["prompt_lookup_num_tokens"] = 10
            synchronize(torch, device)
            start = perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, **options)[0, prompt_tokens:].tolist()
            synchronize(torch, device)
            return output, perf_counter() - start
        # Warm both paths so first-use caches do not systematically favor one arm.
        generate("greedy", 8)
        error = None
        try:
            generate("prompt_lookup", 8)
        except Exception as exc:
            error = str(exc)
        rows = []
        if error is None:
            for repeat in range(args.repeats):
                arms = ("greedy", "prompt_lookup") if repeat % 2 == 0 else ("prompt_lookup", "greedy")
                pair = {}
                for arm in arms:
                    tokens, seconds = generate(arm, args.tokens)
                    pair[arm] = dict(seconds=seconds, output_tokens=len(tokens),
                                     token_sha256=sha256(json.dumps(tokens).encode()).hexdigest(),
                                     tokens_per_second=len(tokens) / seconds)
                rows.append(dict(repeat=repeat, order=list(arms), arms=pair,
                                 identical_tokens=pair["greedy"]["token_sha256"] == pair["prompt_lookup"]["token_sha256"]))
        medians = {arm: statistics.median(r["arms"][arm]["seconds"] for r in rows)
                   for arm in ("greedy", "prompt_lookup")} if rows else {}
        result = dict(audit=str(path), audit_sha256=sha256(path.read_bytes()).hexdigest(),
                      elements=len(source["elements"]), prompt_tokens=prompt_tokens,
                      error=error, runs=rows, median_seconds=medians,
                      speed_ratio=medians["greedy"] / medians["prompt_lookup"] if medians else None)
        report["cases"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("elements", "error", "median_seconds", "speed_ratio")}), flush=True)


if __name__ == "__main__":
    main()
