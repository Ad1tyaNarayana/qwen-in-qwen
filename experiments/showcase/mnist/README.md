# Integer MNIST MLP guest

The Mac trains a 784→32→10 ReLU MLP for eight epochs on all 60,000 training
images, with seed 20261030. Test accuracy is 96.06% float and 96.05% integer.
The native integer C implementation matches every one of the 10,000 test
predictions computed with NumPy integer arithmetic. Test examples 0, 1 and 2
are selected without correctness filtering. The guest predicts 7, 2 and 1.
It prints a 28-column, vertically subsampled ASCII image, label and prediction.

[train.py](train.py) downloads the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist)
into `.cache/showcase/mnist-downloads/`. Downloads are treated only as data.
Compressed sizes, known MNIST MD5 checksums, bounded decompression lengths,
IDX magic/shape and label ranges are checked before training. No downloaded
Python, pickle, archive executable or serialized model is loaded. Compressed
and decoded SHA-256 hashes and source URLs appear in [training.json](evidence/training.json).
The Python TLS certificate configuration failed on the first download attempt;
the implementation uses the existing `curl` with TLS verification enabled.
No certificate checks were disabled and no package was installed.

Weights are signed int8 with per-row scales. Activations are int16. Dot products
accumulate in int32 and use an int64 Q20 rescale. Hidden values use Q8 and ReLU
clamping to 0 through 32767; logits use Q12. Inputs are raw unsigned pixel bytes
copied into aligned RAM. Weights remain in aligned DROM. Only the inner dot
loop differs between accelerated and scalar RV32 builds. Native builds execute
the equivalent scalar integer arithmetic.

NDOT executes 19,056 NDOT4 instructions across the three images. The complete
guest uses 121,329 instructions and 189,092 trace tokens. The scalar guest uses
560,569 instructions and 785,128 trace tokens. The reductions are 4.6202× and
4.1521× respectively. These deterministic counts are measured. The full GPU runs
take
50.590 seconds with NDOT4 and 210.051 seconds scalar, an observed 4.1520×
speedup from one execution per build. Both outputs and DROM images are identical.
The first NDOT4 trace offset is 12,377. The accepted stock prefix checks 16,384
tokens, including 524 completed NDOT4 instructions.

```sh
.venv/bin/python experiments/showcase/mnist/train.py
python3 experiments/showcase/build.py mnist
python3 experiments/showcase/build.py mnist --scalar
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/mnist
.venv/bin/python experiments/showcase/run_reference.py .cache/showcase/mnist-scalar
.venv/bin/python experiments/showcase/verify_local.py
```

The private `.cache/showcase/mnist-model/` directory retains numeric NPZ weights,
full test bytes, integer predictions and the generated coefficient header.
The [workflow](../README.md), [NDOT manifest](../manifest/mnist.json) and
[scalar manifest](../manifest/mnist-scalar.json) bind current acceptance.
