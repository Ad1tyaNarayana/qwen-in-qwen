# Scalar MNIST comparison

This uses [the same integer classifier source](../mnist/main.c), coefficients,
images and output as the NDOT guest. `python3 experiments/showcase/build.py mnist
--scalar` disables NDOT and emits scalar RV32IM matvec loops. The two reference
outputs are byte-identical. Each binary has its own independent full-state
reference. See [evidence](evidence/) and the [manifest](../manifest/mnist-scalar.json).
