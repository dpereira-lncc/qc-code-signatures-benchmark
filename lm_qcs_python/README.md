# LM-QCS Python research implementation

Python implementation of **Short Lee-metric Code-based Signature** (Tan and
Prabowo, VCRIS 2024,
[DOI: 10.1109/VCRIS63677.2024.10813379](https://doi.org/10.1109/VCRIS63677.2024.10813379)),
supporting the three parameter sets in Tables I and II.

## Scope

Implemented:

- `KeyGen`: sample `e1,e2 in U_{ell_e}`, invert `e1`, compute `h=e1^{-1}e2`;
- `Sign`: ephemeral shift, sign bit, sparse `e_bar`, box vectors `u1,u2`,
  Fiat-Shamir challenge, and `s1,s2`;
- `Verify`: reconstruct `t=s1*h-s2`, compute `c^{-1}t`, recompute the challenge,
  and check infinity-norm conditions;
- exact-size serialization for keys and signatures;
- benchmarks with JIT warm-up excluded;
- parameter sets for 128, 192 and 256-bit classical security.

## Important specification choices

The six-page paper does not define a concrete hash-to-challenge algorithm or a
wire format. This implementation therefore uses:

1. SHAKE256 with domain separation;
2. exact uniform rejection sampling over the set of signed weight-`omega_c`
   ternary vectors;
3. rejection of non-invertible challenges, as required by the paper;
4. enumerative (combination-rank + sign bits) encoding of `c`, matching the
   signature-size formula;
5. bit packing of `s1,s2` in `[-gamma,gamma]`.

The scheme is described as “without aborts”. Accordingly, `sign()` makes one
ephemeral draw. If the negligible norm-failure event occurs, the Python API
raises an exception rather than silently introducing rejection sampling.

This is a research implementation, not author-validated official code and not a
constant-time implementation.

## Install

```bash
pip install numpy numba
```

## Run

```bash
python -m lm_qcs_python.run --level 128
python -m lm_qcs_python.run --level 192
python -m lm_qcs_python.run --level 256
```

## Tests

```bash
python -m unittest tests.test_lm_qcs -v
```

## Benchmark

```python
from lm_qcs_python import LMQCS_I
from lm_qcs_python.benchmark import benchmark

print(benchmark(LMQCS_I, repetitions=30))
```
