# MLLMMediaKit

`MLLMMediaKit` owns model-family media behavior. The standard implementation is
Qwen-style image media.

Public methods to understand:

- `prepare(media, *, processor, tokenizer)`: compute token counts and deferred
  sample fields from canonical media metadata.
- `render_text(text, state)`: expand model/template placeholders into the
  language-side media token span.
- `check_truncation(token_ids, keep_len, *, state)`: reject truncation that would
  cut media special tokens.
- `mask_labels(input_ids, labels, *, state, ignore_index)`: hide media special
  tokens from supervised labels.
- `materialize(sample, *, processor)`: perform late media IO and call processor
  media encoders.
- `collate(batch, model_inputs, *, dummy_inputs=None)`: merge media tensors into a
  padded batch.

Private helpers such as dummy text-only media and packed media merge are
collation/finalization details. Do not expose them as user-facing recipe APIs.

For video frame sampling, put frame selection and decoding in `materialize()`.
Put frame counts, token-count estimation, placeholder rendering, and output
field names in `prepare()`/`render_text()`/`collate()`.
