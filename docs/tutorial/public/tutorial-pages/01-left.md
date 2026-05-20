# Chapter I: Overview

MVP-Engine is built around three main concepts: `engine (core)`, `skills`, and `recipes`.

Think of the system as an alchemy lab.

The `engine` is the lab equipment: the furnace, vessels, thermometers, logs, and storage shelves. It provides stable tools for launching training, loading configs, initializing distributed jobs, logging metrics, and saving checkpoints.

The core should stay simple and reusable. It should not contain the special steps for every potion.

## Engine core

Use `mvp_engine/` for infrastructure that is generic, stable, and shared across experiments.

Good core code is boring in the best way:

1. predictable
2. reusable
3. hard to break
