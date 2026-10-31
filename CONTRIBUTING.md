# Contribute

Use a topic branch. Preserve historical reports, evidence, source hashes, and cited Git history. Put fresh verification results outside existing evidence paths. Explain the scope of each new measurement and retain failures as well as successes.

Run `python tools/check_release.py` and `python tools/verify_all.py --tier 0` before proposing a change. Run tier 1 with the tracked-manifest bundle when a change affects retained-output verification. A `SKIPPED` result is not a pass. Never widen a tolerance or remove an assertion to accept historical data.

Do not commit credentials, private host configuration, game data, Doom executables, traces, RAM, pixel dumps, or images. Bundle additions must be individually allowlisted and licensed. Rebuild the tracked bundle manifest only when intentionally changing the bundle; never update a digest to hide a failure.

Use a fresh virtual environment outside the repository. The supported release commands are in `docs/REPRODUCE.md`. Old remote orchestration scripts record a particular investigation and are not portable launch instructions. New GPU runs must pass host and paths explicitly to `tools/gpu_remote.py` or run the workers directly with their documented arguments.
