# VLM Media Ownership

Use when forward-consumed fields are not plain token tensors: images, videos,
audio, dense retrieval chunks, router inputs, or other model-family rows.

This file owns dense model-family field ownership and model-side consumption. It
does not own generic CPKit semantics, attention topology metadata, or parity
artifact format.

## Invariant

- Every dense field consumed by forward has a declared owner layout.
- Fields sliced in `train_pre_step` must be consumed as local fields by the
  model patch.
- Replicated media is allowed only when explicitly documented and validated for
  memory/correctness impact.
- Split units must respect model-family boundaries such as merge groups,
  tubelets, frames, or audio windows.

## Public Validation

- Contract tests must derive model-family fields from `CPSequenceSpec` and
  require matching model-side dataflow.
- Helper outputs such as local indices, local media rows, or gathered/scattered
  features must feed media encoder, placeholder merge, attention, or LLM input
  paths.
- Marker strings, unused helper calls, debug assignments, and immediate inverse
  gather/scatter round trips do not count.

## Assertion Hooks

Use `MODEL_FAMILY_SEQUENCE_FIELDS` in the recipe-local copy of
`references/asserts.py` when field names are not inferable from
`CPSequenceSpec`. Put model-family helper names in
`MODEL_FAMILY_HELPER_NAMES` instead of adding them to the generic helper set.
Use `MODEL_FAMILY_NATIVE_LOCAL_FORWARD=True` only when a runtime validation
proves the installed forward natively consumes CP-local media.
The contract helpers
`assert_model_family_dataflow_contract(...)` and
`assert_cp_helper_outputs_drive_dataflow(...)` enforce cheap source-level
ownership checks.

## Validation Targets

- Each context rank still feeds the full media tensor.
- Local media rows are computed but original global metadata is used.
- Visual/audio features are inserted into placeholders with a different layout
  than `inputs_embeds`.
