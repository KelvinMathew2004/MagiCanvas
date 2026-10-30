# Recovered champion training history

Recovered October 29, 2026 from Git. These are historical evidence, not current training instructions. Use [the current runbook](../../TRAINING_RUNBOOK.md) for the VLM.

Commit `ae0fe91` (October 23, “Qwen LoRA v2”) added the production adapter. Its **40,422,168-byte weight file is byte-identical to the currently installed champion**, SHA-256 `6ebbbe8d20e88504a1ecbb9fb6d5c9466df1c111bde84f099a6f788089d3c63b`. Adapter config and prompt-version marker also match exactly. Weights are not copied into this archive or the Gemini packet.

| Historical revision | Notebook | Data exporter |
|---|---|---|
| Immediately before adapter addition, `210a68b` | [Training cells](QWEN_LORA_COLAB_210a68b.md) | [Exporter](prepare_qwen_training_210a68b.py) |
| Adapter-addition commit, `ae0fe91` | [Training cells](QWEN_LORA_COLAB_ae0fe91.md) | [Exporter](prepare_qwen_training_ae0fe91.py) |

Both notebook versions use Qwen3-0.6B, Unsloth 4-bit LoRA, rank 16 / alpha 32, three epochs, learning rate 1e-4, batch 1 / accumulation 16, 4096-token context, 8-bit AdamW and response-only loss. These agree with the installed adapter's recorded architecture and LoRA parameters.

The preceding notebook uploads `data/qwen_lora/`, uses epoch saves and evaluation every 100 steps. Its exporter fits target text fonts through the engine. The addition-commit notebook changes to source-capped data, preserves reference fonts instead of fitting them, and selects the best validation checkpoint with step saves. Crucially, it describes the source-capped export as the **next experiment**, without replacing the installed adapter's data. Therefore the preceding recipe is relevant evidence, and assuming the champion used the newer source-capped policy would be unjustified.

The exact notebook version executed, uploaded dataset and installed package versions are not proven by co-location in a commit. No original run manifest or dataset hash accompanies the champion. A uniform-data VLM run remains a new experiment, not an exact reproduction; keep image processing and vision-aware response masking in the modern trainer. The recovered recipes do not establish why the more recent 0.6B retraining regressed.

[provenance.json](provenance.json) records full commit IDs and SHA-256 hashes. Recovered files preserve their original bytes; they depend on historical repository APIs and are not standalone executables.
