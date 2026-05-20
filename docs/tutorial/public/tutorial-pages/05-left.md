# Chapter IV: Skills

Skills are a second interface in MVP-Engine. Code is the interface for machines; skills are the interface for coding agents.

They exist because many training features have a stable pattern but model-specific implementation details.

*The key point is that skills let MVP-Engine have rich capabilities without forcing every capability into a heavy generic abstraction.*

Many training repositories become over-abstracted because they try to make one API handle every model, dataset, and distributed strategy. Skills take a different path: keep the core engine simple, then let agents apply structured patterns to each concrete recipe.

## Design Idea

Use code when the behavior can become one clean API:

- checkpoint I/O
- logging
- config parsing
- distributed primitives

