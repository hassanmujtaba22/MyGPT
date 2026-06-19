# Training data format

`train.jsonl` is a **JSON Lines** file: one JSON object per line. Each line is a
conversation in the OpenAI/Hugging Face `messages` format:

```json
{"messages": [
  {"role": "system", "content": "You are MyGPT, a helpful assistant."},
  {"role": "user", "content": "A question or instruction"},
  {"role": "assistant", "content": "The response you want the model to learn"}
]}
```

Rules:
- One complete JSON object **per line** (no pretty-printing across lines).
- The `system` message is optional. `user` and `assistant` turns are required.
- Multi-turn conversations are fine — just add more user/assistant pairs.
- The model learns to produce the **assistant** content given everything before it.

## Tips for a good dataset
- **Match real use.** Write examples that look like the prompts you'll actually
  send and the answers you actually want.
- **Be consistent** in tone, format, and length.
- **Quality > quantity.** 100 clean examples beat 1,000 sloppy ones.
- **Cover your edge cases**, including how the model should refuse or say "I don't know".
- Keep an optional held-out `eval.jsonl` (same format) to measure progress;
  point `data.eval_file` at it in the config.

## Validate before training
```bash
python scripts/prepare_data.py --config configs/default.yaml
```
This checks every line parses, has valid roles, and reports token-length stats.
