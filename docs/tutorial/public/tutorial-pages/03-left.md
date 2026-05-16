# Chapter III: Engine

An `Engine` is the runtime contract between shared training infrastructure and recipe-specific logic.

The core engine owns the boring but critical parts:

- distributed setup
- logging
- mixed precision
- gradient accumulation
- checkpointing
- the outer training loop

Your recipe owns the parts that are different for each experiment:

- model construction
- datasets
- optimizer policy
- scheduler policy
- forward semantics

