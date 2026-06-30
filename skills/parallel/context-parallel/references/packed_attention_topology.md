# Packed Attention Topology

Use when the recipe builds packed or varlen attention metadata such as
`pack_segment_ids`, `cu_seq_lens*`, `max_seqlen*`, or model-family equivalents.

This file owns the contract between attention topology metadata and the Q/K/V
tensors consumed by attention. It does not own mesh setup, generic CPKit
slicing, media ownership, or parity artifact format.

## Invariant

- Dense token tensors may be context-local.
- Topology metadata must describe the Q/K/V layout that the attention call
  actually consumes.
- Padding is either isolated in topology or proven absent from Q/K/V.
- Local metadata and post-Ulysses gathered metadata must not be mixed.

## Public Validation

- Contract tests must assert that `cu_seq_lens*[-1]` matches the flattened Q/K/V
  length for the attention path being checked.
- If a wrapper gathers Q/K/V before attention, validate the post-gather length,
  not only the pre-gather local length.
- If the cheap contract layer cannot observe Q/K/V tensors without running the
  model, put this check in a smoke hook or parity runner. Do not replace it with
  source-marker strings.
- When a cheap runtime probe can access Q/K/V shape and metadata in the same
  call, prefer that probe over source-only checks.

## Assertion Hooks

Use `assert_packed_topology_matches_flattened_qkv(...)` from
`references/asserts.py` in a recipe-local contract or smoke test. Call it at the
point where the `cu_seq_lens*` metadata and the actual flattened Q/K/V length
are both available. If the tensors are available only during real execution,
move this assertion into smoke/parity validation.

## Validation Targets

- CP-on loss differs while token slices reconstruct correctly.
- FlashAttention errors mention varlen length or cumulative sequence mismatch.
- The recipe passes public structure tests but reviewer finds local
  `cu_seq_lens` sent into gathered attention.
