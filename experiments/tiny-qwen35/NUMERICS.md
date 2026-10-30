# Integer profile

The runtime has no floating point. Matrix weights are symmetric int8, with a
signed positive Q24 scale for each output row. Matrix inputs are adaptively
shifted into signed int16. The peak uses ceiling division while choosing the
shift. Its limit is `min(16383, INT32_MAX/(127*n))`. Therefore every intermediate
dot accumulator has absolute value at most `INT32_MAX`. NDOT4's modular
accumulation agrees with the scalar signed sum within this bound. Weight rows
must start at a four-byte boundary and the activation buffer at an eight-byte
boundary. Other rows and tails use scalar RV32IM.

Llama layers preserve the old Q12 arithmetic, Q24 row scales, epsilon 168,
interleaved full RoPE, LUT grids, KV clipping, and signed shifts. Their matrix
payload bytes are copied from the old packer without requantization. This profile
inherits the old packer's range assumptions. It does not promise arbitrary
llama2.c weights have good integer accuracy.

Qwen residual streams, projections, convolution history, MLP values and normalized
outputs are Q12. They saturate at absolute integer value `2**23`, or real 2048.
Every saturation increments a diagnostic counter. Convolution coefficients are
Q12. Convolution sums retain Q20 before SiLU. The normalized DeltaNet q/k vectors
are Q15. Values, recurrent states, corrections and core attention outputs are
Q20, saturated at `2**26`, or real 64. The high precision is necessary for small
valid states near initialization. All high-precision saturations increment the
same diagnostic counter. A clean experiment requires zero saturations.

The causal convolution stores the preceding `kernel-1` raw projected vectors,
oldest first, separately for every channel. The newest projection uses the last
convolution coefficient. Q/k L2 normalization uses integer square root of the
sum of squares plus 1,099,512, the rounded Q40 representation of HF's fixed
`1e-6`. It divides into Q15. Query scaling by `1/sqrt(key_dim)` uses a Q12
constant. Repeated query/key heads share their normalized vectors, with each
value head retaining a separate state matrix.

DeltaNet evaluates the following in order, including the current token's state
update before reading the output.

```
beta = sigmoid(b)
decay = exp(-exp(A_log) * softplus(a + dt_bias))
S = decay * S
correction = beta * (v - k^T S)
S = S + k * correction^T
output = (q / sqrt(key_dim))^T S
output = RMSNormGated(output, z)
```

The positive softplus intermediate remains int64, including the permitted
Q20 endpoint `2**31`. Qwen attention scores saturate at the Q12 activation
bound before narrowing to a word. Saturations increment the clipping diagnostics.

`exp(A_log)` is exported in Q20. Softplus has a Q20 LUT on `[-16,16]`, with
input spacing `1/128`. Above 16 it uses the linear asymptote; below -16 it
uses the endpoint. Sigmoid is Q15 on the same input grid. Qwen's exponential
LUT is Q20 with spacing `1/1024` on `[0,16]`. Its endpoint rounds to zero.
The LUT index truncates toward the lower grid point. The selected image stores
values at cell midpoints, using `i+0.5` in the exponential and `i-2048+0.5`
in sigmoid and softplus. The left-endpoint baseline remains in the evidence. This differs from the
legacy exponential's `1/128` spacing. Full-attention softmax uses the same
Qwen exponential table. The resulting accuracy is measured against HF; LUT
agreement is not represented as exact floating-point equivalence.

Ordinary Qwen RMS weights export `1 + weight` in Q12. Gated RMS weights export
`weight` directly. Gated normalization precedes the SiLU gate. RMS epsilon
is rounded into Q24 for Q12 inputs, shifted into Q40 for Q20 inputs. The
division produces Q12 outputs in both cases. Full attention splits q/gate
within each head, applies the per-head q/k norms before partial RoPE, and
uses a sigmoid output gate before `o_proj`. RoPE tables store signed int16 values rounded from 32767 times cosine or
sine. Rotation products shift by 15. Qwen RoPE pairs the two halves of
the rotated prefix. Text positions are identical across the three mRoPE axes.
Vision and non-default RoPE scaling are outside this packer's supported profile.

All multiplications and sums outside NDOT use int64 or uint64 intermediates.
For the Qwen profile the packer caps matrix input width at 4096, DeltaNet head
width at 1024, convolution kernel at 16, context at 512, and exported Q12
coefficients at `2**23`. The arena gate usually imposes smaller dimensions.
Convolution sums are bounded by `16 * 2**23 * 2**23 = 2**50` before rescaling.
Q20 normalization sums are bounded by `1024 * (2**26)**2 = 2**62`.
Recurrent dot sums are bounded by `1024 * 2**26 * 2**15 = 2**51`.
The state correction and query-scale products fit int64 before saturation.
The matrix sum times row scale is strictly below `2**62`. Softmax totals
are at most `512 * 2**20 = 2**29`. The software reference and sanitizers check
the executed workloads in addition to these range bounds.

All tensor locations and arena offsets come from the trusted packer. This is
not an untrusted model-file loader. The runtime checks the format version,
arena capacity, prompt length, and execution context at boot. The packer and
ELF linker enforce model, code and RAM capacity independently. The linked
image is the final storage gate.

# Offline weight rounding

The selected model uses `midpoint+covariance`. Calibration runs the unmodified
floating HF model on 64 TinyStories excerpts of at most 64 tokens. The packer
retains only each linear projection input covariance. All calibration runs on
the Mac CPU. The checkpoint hash binds the covariance file to its source model.
No activation or prediction from calibration enters guest RAM or neural state.

Each covariance is normalized by its mean diagonal and receives `0.01 * I`
damping. The tied embedding/classifier covariance is mixed equally with `I`
to retain an embedding reconstruction objective. Columns are processed in
descending diagonal order. With `U = cholesky(inverse(H), upper=True)`, the
quantizer rounds column `j` at the fixed Q24 row scale and compensates the
remaining columns by `(W[:,j]-Q[:,j]) / U[j,j] * U[j,j:]`. The stored result is
still int8, with the same positive Q24 row scale. The runtime format and binary
do not change. Nearest row rounding, scale fitting, convolution equalization,
and alternative native rounding probes remain separate rejected experiments.
