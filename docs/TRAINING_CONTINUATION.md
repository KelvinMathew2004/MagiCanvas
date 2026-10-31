# Continue training

1. Follow [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) with the new
   `layout_training_bundle_uniform.zip`.
2. Train only `qwen3-vl-2b-instruct` for the current run; defer 0.8B.
3. Write to `MyDrive/MagiCanvas/uniform_v2_candidates`.
4. Resume only checkpoints created with that exact uniform bundle and launcher.
   Keep old `audited_v2_candidates` checkpoints separately.
5. Retain ten-step optimizer/RNG saves and normal epoch validation.
6. Download each complete `final` folder, including processor/tokenizer,
   environment and training manifest, using the guide's ZIP cell.
7. Return the ZIPs for evaluation. Do not retrain 0.6B or replace production.
