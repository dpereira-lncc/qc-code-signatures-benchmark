# HQCS-R — Python research implementation

Implementation based on the paper:

> C. H. Tan and T. F. Prabowo,
> "Hamming Metric Code-Based Signature Scheme With Restricted Vectors",
> AsiaJCIS 2025. [DOI: 10.29007/d743](https://doi.org/10.29007/d743).

## Algorithms

### KeyGen

```text
h <- R_q
e1,e2 <- U_{ell_e}
b = e1*h + e2
pk = (h,b)
sk = e1
```

### Sign

```text
u <- R_q
v <- R_q^*
c = H(m,v,[vu]_p,[vuh]_p,pk)
s = u + v^{-1} c e1
```

The signature is accepted when:

```text
[vs]_p = [vu]_p
[vsh-cb]_p = [vuh]_p
```

### Verify

```text
t = vsh-cb
c' = H(m,v,[vs]_p,[t]_p,pk)
```

Verification accepts when `c'=c`.

## Implemented parameter sets

- HQCS-R-1
- HQCS-R-2
- HQCS-R-3

The parameters and sizes reproduce Tables 1 and 2 of the paper.

## Concrete implementation choices

The paper does not specify a hash-to-challenge function or a complete binary
encoding. This implementation uses:

- SHAKE256 with domain separation;
- a ternary challenge with exact weight `omega_c`;
- `h` and `v` expanded from seeds of `2*lambda` bits;
- `s` coefficients packed in `ceil(log2(q))` bits;
- challenge `c` encoded with two bits per coordinate.

These choices exactly reproduce the stated size formulas.

## Arithmetic

Dense multiplication in `F_q[x]/(x^k-1)` uses Kronecker substitution and
Python big-integer multiplication. Inversion of `v` uses an extended Euclidean
algorithm compiled with Numba and modular multiplication without overflow.

## Dependencies

```bash
pip install numpy numba
```

The main workflow uses NumPy and Numba. Reproducible parameter analysis also
requires SymPy:

```bash
pip install sympy
```

## Running

```bash
python -m hqcs_r_signature.run --set 1
python -m hqcs_r_signature.run --set 2
python -m hqcs_r_signature.run --set 3
```

## Tests

```bash
python -m unittest tests.test_hqcs_r_signature -v
```

## Warning

This is an independent implementation based on the paper. No official public
implementation was found. The code is not constant-time and must not be used
in production.

## Note on HQCS-R-3

The public-key size formula in the text produces 7116 bytes for HQCS-R-3,
because `k=1619` and `ceil(log2(q))=35`. However, Table 2 reports 7520 bytes.
To reproduce the paper's table, this implementation pads the HQCS-R-3 public
key with zeros up to 7520 bytes. The padding bytes carry no cryptographic
information.

## Candidate parameter sets for NIST levels 3 and 5

The paper provides only 128-bit parameter sets. This version adds two candidate
sets derived from HQCS-R-1:

| Set | Target security | k | q | p | ell_e | omega_c |
|---|---:|---:|---:|---:|---:|---:|
| HQCS-R-NIST-1 | 128 | 1511 | 2131128193 | 16780537 | 2 | 73 |
| HQCS-R-NIST-3-CANDIDATE | 192 | 2267 | 3777613439 | 29981059 | 2 | 110 |
| HQCS-R-NIST-5-CANDIDATE | 256 | 3022 | 5964784949 | 47339563 | 2 | 146 |

### Derivation method

- `k` and `omega_c` were scaled by `lambda/128`;
- `ell=126` and `ell_e=2` were retained;
- `p` and `q` were chosen as primes;
- `q` was adjusted to preserve the same lower acceptance estimate as HQCS-R-1,
  approximately `0.996862`;
- the conditions `2p < q-1`, `ell=floor(q/p)`, and
  `q-1 < (ell+1)*p` were verified;
- challenge entropy exceeds `2*lambda`;
- the forgery condition `k*log2(q/p) > lambda` is satisfied with a wide margin.

These sets were NOT analyzed by the paper's authors or subjected to a complete
BKZ/HRSDP estimate. They must be treated as experimental candidate parameters,
not cryptographically validated parameter sets.

Run them with:

```bash
python -m hqcs_r_signature.run --set nist1
python -m hqcs_r_signature.run --set nist3
python -m hqcs_r_signature.run --set nist5
```

## Candidate derivation results

| Set | k | q bits | omega_c | Challenge entropy | Lower acceptance | PK | SK | Signature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NIST 1 | 1511 | 31 | 73 | 494.4 bits | 0.996862 | 5888 B | 567 B | 6265 B |
| NIST 3 candidate | 2267 | 32 | 110 | 740.3 bits | 0.996862 | 9116 B | 851 B | 9683 B |
| NIST 5 candidate | 3022 | 33 | 146 | 984.8 bits | 0.996862 | 12530 B | 1134 B | 13286 B |

The challenge entropy values exceed the minimum values of 256, 384, and 512
bits. The public-key forgery condition is also satisfied with a wide margin.

The reproducible script:

```bash
python -m hqcs_r_signature.parameter_analysis
```

calculates primality, algebraic conditions, entropy, the acceptance bound, the
forgery exponent, and sizes.

### Security limitation

The derivation scales the HQCS-R-1 set and preserves the paper's explicit
tests, but it does not reproduce a complete BKZ block-dimension estimate or
the HRSDP algorithms for the new parameters. The NIST 3 and 5 sets therefore
remain experimental candidates. A formal NIST category claim would require
independent cryptanalysis.

### Automated validation

Automated tests check sizes, primality, algebraic conditions, serialization,
and the demonstration workflow. This does not replace independent
cryptanalysis or validate the NIST 3 and 5 candidates.

## Parameter-inequality inconsistency

The paper states `2p < q-1 <= ell*p` while also using `ell=floor(q/p)`. These
two conditions are generally incompatible. In addition, the parameters in
Table 1 do not satisfy `q-1 <= ell*p`.

Package validation uses the conditions consistent with the table:

```text
2p < q-1
ell = floor(q/p)
q-1 < (ell+1)*p
```

This interpretive correction is made explicit in `parameter_analysis.py`.
