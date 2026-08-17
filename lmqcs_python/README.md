# LMQCS in Python

Research implementation of the **LMQCS** scheme described in “Lee Metric
Code-based Signature” (Tan and Prabowo, ISITA 2024,
[DOI: 10.34385/proc.86.tu-pm-2-1-3](https://doi.org/10.34385/proc.86.tu-pm-2-1-3)).

## Status

The paper provides no official code and does not fully specify the hash to a
fixed-weight ternary challenge, serialization, or rejection procedure. This
package implements the paper's equations and documents the required choices:

- SHAKE256 with domain separation;
- a ternary challenge `c` with weight `omega_c`, rejected until invertible;
- uniform sampling from `{-ell,...,ell}^n`;
- a new signing attempt whenever any verification condition fails;
- compact serialization using the sizes in Table I.

This is therefore a reproducible implementation of the paper, not an official
implementation validated by its authors.

## Installation

```bash
pip install numpy numba
```

## Complete workflow

```bash
python -m lmqcs_python.run --level 128
python -m lmqcs_python.run --level 192
python -m lmqcs_python.run --level 256
```

## API

```python
from lmqcs_python import LMQCS128, keygen, sign, verify

par = LMQCS128
keys = keygen(par)
sig = sign(b"message", keys.secret_key, keys.public_key, par)
assert verify(b"message", sig, keys.public_key, par)
```

## Benchmark

```python
from lmqcs_python.benchmark import benchmark
from lmqcs_python import LMQCS128

print(benchmark(LMQCS128, repetitions=30))
```

## Tests

```bash
python -m unittest tests.test_lmqcs -v
```
