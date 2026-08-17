# Punctured QC-LDPC signature with random insertion

Research implementation based on the paper:

> X. Lin, Y. Fu, Z. Wang, J. Yin,
> "A New Post-Quantum Signature Based on Punctured QC-LDPC Code
> With Random Insertion", IEEE ICC 2025.
> [DOI: 10.1109/ICC52391.2025.11162115](https://doi.org/10.1109/ICC52391.2025.11162115).

## Implemented workflow

The base systematic matrix is:

```text
H = [P^T | I_r]
```

The construction removes `p` rows from `P^T` and the corresponding columns
from the identity part, producing:

```text
H_D = [P_D^T | I_(r-p)]
```

It then adds `p` random rows `R` together with a new identity matrix of size
`p`:

```text
H_DI = [
    H_D   0
    R     I_p
]
```

During signing:

```text
H_D e_(n-p)^T = s*_(r-p)
e_p = s*_p + R e_(n-p)
```

and then:

```text
e_bar = [e_(n-p) | e_p]
```

The error is permuted into the public representation and verified with:

```text
H_pub e^T = SHAKE256(domain || len(m) || m || i)
```

In this encoding, `len(m)` and counter `i` occupy 8 little-endian bytes each,
and the SHAKE256 output is truncated to `r` bits.

## Profiles

### Demo

```bash
python -m qc_ldpc_cfs_punc.run --profile demo
```

### Original

Main structure:

```text
n = 16384
k = 12288
r = 4096
```

The paper does not provide a numeric value for `p`. This implementation uses:

```text
p = 64
```

as an experimental choice. The `estimated_128`, `estimated_192`, and
`estimated_256` profiles use scaled dimensions without claiming validated
security.

## Choices not defined by the authors

The following values are reproduction choices:

- `p = 64` in the full profile;
- weight 13 for the QC-LDPC blocks;
- selection of punctured rows by lowest row weight;
- channel probability 0.003;
- normalized min-sum with factor 0.75;
- at most 80 iterations;
- weight limit 128;
- at most 10,000 attempts;
- scrambling matrix implemented with elementary row operations;
- public key stored as a dense binary matrix.

The paper calls for selection based on minimum-weight codewords but provides
no concrete algorithm for finding them. The lowest-row-weight criterion is an
experimental substitute and is not equivalent to the theoretical procedure.

## Key size

This implementation stores `H_pub` in full, resulting in:

```text
r * ceil(n / 8) bytes
```

When `n` is divisible by 8, this is equivalent to `r*n` bits. For the full
profile, the size is 8 MiB. The 6144-byte value reported in the paper assumes
structural compression that is not specified after random-row insertion. It
has therefore not been reproduced artificially.

## Dependencies

```bash
pip install numpy numba
```

## Tests

```bash
python -m unittest tests.test_qc_ldpc_cfs_punc -v
```

## Standardization and bounded signing

Key generation uses the shared `ShakeRNG`. KeyGen, Sign, and Verify do not
maintain internal timers; runners measure each call externally. The CSR graph
is reused, and normalized min-sum remains compiled with Numba.

`sign` accepts `max_attempts` and raises `SigningFailure` when the limit is
reached. For the `original` and `estimated_128/192/256` profiles, runners use
100 attempts by default. The benchmark records failure to produce a signature
as `signing_incomplete` instead of appearing to hang.

Bounded direct execution:

```bash
python -m qc_ldpc_cfs_punc.run --profile original --max-sign-attempts 100
```

## Warning

This code is intended for scientific reproduction. It is not constant-time
and must not be used in production.
