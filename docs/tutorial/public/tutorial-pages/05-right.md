Use a skill when the feature pattern is repeatable but the glue changes per recipe, such as:

- gradient checkpointing
- tensor parallel plans

This keeps `mvp_engine/` small while still making complex training techniques reusable.

## How to Use Skills

Just ask your coding agent to implement a feature. If there is a skill for that feature, the agent will use it. If not, the agent will write code as normal.

## What You Get

- less generic code in the core engine
- less repeated hand-written glue
- changes adapted to each recipe
- clearer review points
- recipe-local validation when a skill changes training behavior

In practice, skills turn "same idea, different experiments" work into a repeatable coding workflow.
