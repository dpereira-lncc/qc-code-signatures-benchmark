# QC-LDPC CFS — Python research implementation

Implementation based on F. Ren, X. Yang, and D. Zheng, **A QC-LDPC Code Based
Digital Signature Algorithm** (NaNA 2018).

## Specification limitations

The paper does not provide a complete instance: exact QC-LDPC block weights,
concrete constructions for `S` and `Q`, channel probability, normalization,
and the BP iteration limit are missing. This package uses explicit,
reproducible choices and is therefore **not an official implementation**.

## Implemented construction

- `H_secret`: one row of four sparse circulant blocks;
- `S`: a dense invertible circulant matrix;
- `Q`: a four-block diagonal matrix of sparse invertible circulants;
- `H_public = S H_secret Q`;
- decoding: normalized min-sum for syndromes;
- hash: SHAKE256 to an `r`-bit syndrome;
- signature: `(z, counter)`.

## Profiles

### Demo

```bash
python -m qc_ldpc_cfs.run --profile demo
```

### Original paper structure

Uses `n0=4`, `k0=3`, and `q=4096`, resulting in
`(n,k)=(16384,12288)`:

```bash
python -m qc_ldpc_cfs.run --profile original
```

The full profile can consume substantial time and memory during decoding.
The `estimated_128`, `estimated_192`, and `estimated_256` profiles use
experimentally scaled dimensions and do not represent validated security.

## Tests

```bash
python -m unittest tests.test_qc_ldpc_cfs -v
```

## Benchmark

```python
from qc_ldpc_cfs import DEMO
from qc_ldpc_cfs.benchmark import benchmark

print(benchmark(DEMO, repetitions=10))
```

## Public-key representation

For `n0=4`, the public matrix used by this package is:

```text
H_public = [A0 | A1 | A2 | A3]
```

All four circulant blocks are stored explicitly. For the paper profile, with
blocks of `4096` bits:

```text
4 * 4096 bits = 16384 bits = 2048 bytes
```

The 6144-byte expression in the paper corresponds to a count of twelve blocks,
but the algorithmic workflow implemented here uses one row containing four
circulant blocks.

This code is intended exclusively for scientific use. It is not constant-time
and must not be used in production.

## Performance optimizations

This version introduces two improvements without changing the algorithmic
workflow:

1. The Tanner graph is built once during `keygen()` and attached to the secret
   key. Subsequent `sign()` calls reuse the same graph.
2. The normalized min-sum kernel is implemented as a Numba-compiled function
   (`@njit(cache=True)`) using `float32` vectors and CSR adjacency storage.

Decoder warm-up is explicitly requested by the runners and excluded from the
measurements. KeyGen, Sign, and Verify do not maintain internal timers.

Dependencies:

```bash
pip install numpy numba
```

`keygen` receives the shared `ShakeRNG`, and `sign` accepts `max_attempts`.
For the `original` and `estimated_128/192/256` profiles, runners use 100
attempts by default and report `SigningFailure`, avoiding a wait of up to
10,000 attempts. The limit can be explicitly changed for experiments.
