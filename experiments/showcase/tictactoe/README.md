# Tic-tac-toe against the guest CPU

The human is X and moves first. The guest plays O using exhaustive minimax with
alpha-beta pruning and a deterministic lowest-square tie break. It reads only
scripted human square numbers from `moves.json`. The build validates those
bytes and places them in guest DROM. Used squares are skipped; exhaustion fails.
The guest alone chooses all CPU moves, detects wins and prints each board.

This uses the continuing DMC input pattern: build-time external input becomes
read-only guest data consumed at deterministic simulation boundaries. Here the
boundary is a human turn, rather than a Doom tic. No host-solved move or board
is supplied. `host_move` isolates the transport for a future live guest input
ABI. `-DNATIVE -DLIVE_NATIVE` already supports native stdin digits 0 through 8;
a live neural transport is not implemented.

`verify_local.py` checks all four recorded CPU choices against an independent
Python minimax. `check_perfect.c` explores all human continuations against the
same native C policy and observes 681 terminal games with zero human wins.

```sh
python3 experiments/showcase/build.py tictactoe
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/tictactoe
clang -O2 -DNATIVE -Iexperiments/showcase -I.cache/showcase/tictactoe \
  experiments/showcase/tictactoe/check_perfect.c -o .cache/showcase/tictactoe/check-perfect
.cache/showcase/tictactoe/check-perfect
```

See the [workflow](../README.md), [evidence](evidence/) and
[manifest](../manifest/tictactoe.json) for the recorded run.
