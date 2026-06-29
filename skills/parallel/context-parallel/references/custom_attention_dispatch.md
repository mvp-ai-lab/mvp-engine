# Custom Attention Dispatch

Use when the model wraps attention, overrides `forward`, calls a custom attention
interface, or uses external modules not reached by the shared CP runtime only
through `CP_MODULE_CONFIG`.

This file owns proof that the installed attention execution path reaches
CP-compatible dispatch. It does not own the base meaning of `CP_MODULE_CONFIG`,
Q/K/V topology math, or gradient synchronization order.

## Invariant

- `CP_MODULE_CONFIG` names the runtime classes that actually consume CP-sharded
  hidden states.
- The installed attention forward path reaches CP-compatible dispatch.
- Metadata passed into dispatch matches the layout consumed by that dispatch.

## Public Validation

- Contract tests must check executable binding or a runtime probe, not only the
  presence of class names or marker strings.
- Accepted evidence includes instance-local `forward` binding, a patched
  installer that binds an adapter, or a probe that observes the CP dispatch path.
- Class-level monkeypatches are risky because they can leak across instances and
  should be rejected unless the recipe documents why they are isolated.

## Assertion Hooks

Fill `CP_ATTENTION_CLASS_NAMES` in the recipe-local assertion copy when static
class inference is not enough. Use `assert_attention_dispatch_bound(...)` from a
smoke hook or recipe-local probe when the model uses an external attention
interface, wrapper, or dynamic dispatch path.

## Validation Targets

- `CP_MODULE_CONFIG` is present but attention still calls the stock path.
- Source contains the right marker names but no binding to runtime modules.
- CP-off works and CP-on fails inside a model-family attention wrapper.
