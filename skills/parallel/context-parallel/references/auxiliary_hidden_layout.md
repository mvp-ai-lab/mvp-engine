# Auxiliary Hidden Layout

Use when visual deepstack tensors, router hidden states, cached embeddings, or
other auxiliary hidden tensors enter the LLM next to `inputs_embeds`.

This file owns layout agreement for auxiliary hidden tensors at the LLM
boundary. It does not own media encoder ownership, attention dispatch, or
CP-off/CP-on parity metrics.

## Invariant

- Tensors entering the LLM must share the same sequence ownership as
  `inputs_embeds`.
- If `inputs_embeds` is local sequence/full hidden, auxiliary hidden states must
  be local sequence/full hidden too.
- Sequence/full-hidden transforms must bracket the operation that needs that
  layout and then return to the LLM layout.

## Public Validation

- Contract tests should require local masks or indices to be used when selecting
  auxiliary hidden states.
- `scatter_seq_gather_hidden(...)` alone is not proof that an auxiliary tensor
  was selected back to local token positions.
- If the public contract layer cannot access tensors without running the model,
  put `assert_auxiliary_hidden_layout(...)` in a smoke hook or parity runner.
  Do not replace tensor-shape proof with marker strings.
- A cheap runtime hook can compare auxiliary sequence length against
  `inputs_embeds.shape[1]`.

## Assertion Hooks

Fill `AUXILIARY_HIDDEN_NAMES` in the recipe-local assertion copy. Add a smoke or
contract hook that calls `assert_auxiliary_hidden_layout(inputs_embeds,
auxiliary_tensors)` when the tensors are available at the LLM boundary.

## Validation Targets

- LLM receives local `inputs_embeds` plus global auxiliary hidden tensors.
- Deepstack or router states still use global masks after token slicing.
- No shape error occurs, but parity fails because auxiliary rows are shifted.
