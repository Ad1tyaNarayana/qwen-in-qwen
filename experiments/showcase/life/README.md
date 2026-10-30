# Toroidal Game of Life guest

`main.c` advances a 24×16 toroidal board through five displayed generations.
It starts with a glider, a blinker and a stable block. The guest computes all
neighbor counts and successor cells. The final unprinted generation is also
computed, and remains part of the exact final-state check.
`verify_local.py` recomputes every printed frame with an independent Python set
implementation and modular coordinates.

Run `python3 experiments/showcase/build.py life`, then
`.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/life`.
The shared [workflow](../README.md), [evidence](evidence/) and
[manifest](../manifest/life.json) describe acceptance and its current status.
