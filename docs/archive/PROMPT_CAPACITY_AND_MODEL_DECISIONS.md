> Historical record. Use [the current documentation map](../README.md) for setup and current status. Results below belong to their recorded snapshot.

# Dense prompt capacity and model selection

## Decisions against the supplied recommendations

- [x] Replace the fixed 4,096 input-token veto with an inventory-aware output
  reservation inside the existing 8,192-token runtime cap. Prompt admission is
  bounded by 6,000 tokens and the actual model context, whichever is smaller.
- [x] For dense inputs crossing the old limit, collect relations across all
  elements, then retain local backing ownership, nonredundant alignment chains
  and useful overlaps within 900 tokenizer-measured tokens. Each relation
  remains a trained pair tuple. Leaf aliases, source text, normalized boxes,
  font-scale convention and output schema are unchanged. Ordinary prompts
  below the old boundary remain unchanged.
- [x] Measure capacity locally before model generation. All 13 additional stress
  tasks fit; twelve previously refused. Actual dense prompts remain roughly
  3,800–4,700 tokens after pruning, so a 2,200–2,800-token claim is unsupported.
- [x] Compare fresh 0.6B and 1.7B proposals on the identical 13 tasks with one
  selector, including exact raw inventory, output flags, visual quality and
  generation time. Larger model size alone does not settle this decision.
- [ ] Promote 1.7B only after a relevant quality improvement and a wider release
  evaluation. The prior 20-task comparison had complete inventories on 1.7B
  but the same final learned-family win count, with longer generation time.

## Recommendations that need more evidence

**Macro compression:** retained as a proposed experiment rather than a default.
Changing leaf elements into synthetic macro elements changes the input
semantics, even if JSON keys look identical. An adapter trained on leaf geometry
has no established 100% completion or quality guarantee for that new input.
The actual grouping audit is also uneven: the four supplied designs currently
produce 5, 13, 24 and 9 composite blocks. One 27-element block joins hero/copy
and five product cards via a translucent panel and caption link. Compressing
that group would preserve the existing over-grouping failure. Correct nested
ownership must precede any macro-only model contract.

**Inert geometry:** empty text and zero-area layers need numerical robustness.
A blanket width/height ≤1 rule would also classify legitimate thin dividers as
inert. Clamping every such item and excluding its defects could hide real
clipping or containment failures. Preserve all IDs and visible-content checks;
handle zero denominators narrowly and disclose any placeholder recovery.

**Repeated cards and grids:** explicit groups/backings already form the six
speaker cards. `container_flow.arrange` already compares single rows, stacks
and a wrapped row. Menu name/price ownership is genuinely missing, and broad
panel ownership merges independent cards in another source. A generic nested
attachment change needs its own frozen-prediction ablation; blindly adding
another grid on top of a wrong block tree cannot solve that problem.

**Image non-overlap:** blanket bans would reject source-supported layered
artwork, badges and photos. Continue to distinguish new painted overlap from
intentional source overlap. The supplied menu image overlap is an outlier to
review under that contract.

**Structural ranker:** leave experimental. The aggregate pilot improved on its
baseline, but the actual campaign validation subset only tied at 5/8. Controlled
shrink negatives are easier than real layouts and cannot justify promotion.
Several newly inspected outputs have zero flags yet detached prices/labels;
a zero-flag tie is therefore not sufficient evidence of safety. A paired
ablation with relevant human preferences is needed before deploying weights.

No new adapter training, model publication or production VLM is required for
the capacity change. Training becomes relevant only if fresh admitted proposals
still fail consistently after the measured inference/realization issues are
separated.

## Completed 13-task comparison

| Metric | Available 0.6B + v2 adapter | Available 1.7B + expanded adapter |
|---|---:|---:|
| Complete JSON | 13/13 | 13/13 |
| Raw valid inventory/geometry | 7/13 | 9/13 |
| Prompt capacity refusals | 0 | 0 |
| Flagged final outputs | 7 | 6 |
| Learned-family winners | 1 | 2 |

Ten final outputs are exactly identical. Three differ: a conference banner
still has clipping/collision problems, a menu banner loses its image-overlap
flag but still separates prices from their rows and underuses the canvas, and
the bar-menu placement changes while remaining visually too small.
The summed defect cost is dominated by billion-point hard-defect sentinels and
must not be read as a meaningful percentage quality improvement. There is no
human target for these tasks, so MAE is deliberately null.

Both arms use the same prompt policy, source geometry and current selector.
The 0.6B selected results come from the completed 59-case current-engine replay;
the 1.7B fresh run already executed that same selector. Every engine digest,
source, target and decoded proposal is verified before reusing those results.
The zero-denominator guard was added after 0.6B generation, so that arm was
replayed through it before comparison. Raw validity never receives recovery
credit. These are comparisons of available model/adapter combinations, not an
isolated causal experiment on parameter count.

Recorded generation totals are about 607s and 1,050s respectively. Runs were
sequential on the available native Apple runtime, with concurrent CPU work
and interruptions; these are observational timings, not an isolated M5 Pro
or Docker benchmark. The larger arm uses more compute and improves inventory and flag
counts here; the three changed visuals remain poor. The default remains 0.6B pending
a wider quality/runtime release gate. A documented environment override selects
the 1.7B base and its matching adapter now.

The unified viewer keeps 59 source/resize tasks. Its stress-model switch shows
either arm for the 13 stress cases; the other 46 keep their recorded predictions.
Main regression: prior 46 documents/winners remain unchanged; the bar-menu case
changes winner with no new flags but remains undersized. All 348 tests and 34
subtests pass. The structural ranker remains disabled; no adapter was retrained
or published.
