# Training workflow

- [x] Read poteto-mode Principles and frame the completion predicate.
- [x] Phase A. Frame. Exported model must fit the target byte budgets, load in native run.c, and have measured held-out loss and three prompt samples.
- [x] Phase B. Design the workflow. Inspect downloaded code, isolate the environment, prepare the corpus, measure a pilot, train with a wall-time limit, export, evaluate, and verify hashes after transfer.
- [x] Phase C. Run the loop. Keep the architecture if the pilot meets VRAM and time budgets. Use the same held-out stories for comparison.
- [x] Phase D. Keep the audit trail. Save configuration, modifications, package versions, logs, and decisions.
- [x] Phase E. Verify and hand back. Native C samples, artifact byte counts and hashes, README and evidence JSON.

GPU throughput checkpoint uses 50 pilot updates before choosing the main iteration count. Training gets at most 4200 seconds plus the pilot and final evaluation, below 90 minutes total. No independent agents, commits, or PRs are requested.
