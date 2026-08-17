"""
HQCS-R parameter search for 128/192/256-bit security
======================================================
Extends Table 1 / Table 2 of Tan & Prabowo, "Hamming Metric Code-Based
Signature Scheme With Restricted Vectors" (AsiaJCIS 2025), which only
publishes concrete parameters for the 128-bit security level.

WHAT THIS DOES DIFFERENTLY FROM A NAIVE LINEAR SCALING SCRIPT
---------------------------------------------------------------
Sizes must come from an actual joint search over (k, q, p, l_e, omega_c)
satisfying every constraint the paper imposes, not from extrapolating the
single published 128-bit row. This script implements each constraint from
the paper directly:

  1. BKZ key-recovery hardness   -- Section 2.3 / 5.1
  2. Acceptance rate >= 0.99     -- Corollary 1 (verified to reproduce the
                                     paper's own tau values exactly)
  3. Collision resistance        -- Section 5.2.1
  4. Forgery resistance          -- Section 5.2.2 / Proposition 2
  5. p,q relationship            -- q = ell*p + 1 (q,p prime), which is what
                                     the paper's own HQCS-R-2 / HQCS-R-3 rows
                                     actually satisfy exactly (verified below)

IMPORTANT CAVEAT ON THE BKZ MODEL (read before using these numbers)
---------------------------------------------------------------------
The paper states the attack lattice's shortest vector should be set equal
to delta^d * Vol(L)^(1/d) (Section 2.3). Implemented literally, this predicts
that recovering the secret key requires beta close to the full lattice
dimension (d = 2k) at ALL three of the paper's own published 128-bit rows --
i.e. it says the scheme is *far* more secure than 128 bits, which is not a
useful discriminator for choosing k.

This script instead uses the standard "unique-SVP" refinement used
throughout the NTRU/LWE lattice-estimator literature (e.g. the New Hope,
Kyber, Falcon security analyses), which accounts for which projected block
of the BKZ-reduced basis actually needs to contain the short target vector:

    sqrt(beta/d) * ||secret|| <= delta(beta)^(2*beta-d-1) * Vol(L)^(1/d)

This is very likely what the paper's short description is condensing, but
it is a *reconstruction*, not something stated explicitly in the paper.
Applied to the paper's own three published rows it gives self-consistent
classical security estimates of 138.7, 144.8, and 138.7 bits respectively
for a stated 128-bit target -- i.e. an apparent ~1.10-1.13x safety margin
built into their choices. This script reproduces that same margin when
searching for 192- and 256-bit parameters, for a like-for-like comparison.

This script does NOT implement the ISD-based attack complexity of Theorem 2
/ Equation (1). The paper states BKZ is the binding (weaker) attack at its
own 128-bit parameters; that has not been independently re-verified here
for the new 192-/256-bit candidates. Treat the output as a well-justified
ESTIMATE for a comparison table, not as authoritative cryptographic
parameters -- ideally cross-check with a proper lattice estimator (e.g.
https://github.com/malb/lattice-estimator) before using in a submission.
"""

import math
from sympy import nextprime, isprime


# ---------------------------------------------------------------------
# 1) BKZ key-recovery attack (unique-SVP success condition)
# ---------------------------------------------------------------------

def bkz_delta(beta):
    """Root Hermite factor, Chen-Nguyen BKZ 2.0 model (paper's ref [17])."""
    return ((math.pi * beta) ** (1.0 / beta) * beta / (2 * math.pi * math.e)) \
        ** (1.0 / (2 * (beta - 1)))


def usvp_beta_min(k, log2_q, target_norm, beta_lo=50):
    """Smallest BKZ block size solving unique-SVP in the key-recovery lattice
    (secret (e1,e2), dimension d=2k, determinant q^k)."""
    d = 2 * k
    beta_hi = d - 2
    log2_target = math.log2(target_norm)

    def margin(beta):
        log2_delta = math.log2(bkz_delta(beta))
        log2_rhs = (2 * beta - d - 1) * log2_delta + 0.5 * log2_q
        log2_lhs = 0.5 * math.log2(beta / d) + log2_target
        return log2_rhs - log2_lhs

    if margin(beta_hi) < 0:
        return None
    lo, hi = beta_lo, beta_hi
    if margin(lo) >= 0:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if margin(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def bkz_security_bits(k, log2_q, l_e, quantum=False):
    var_e = l_e * (l_e + 1) / 3.0          # Var of a coord uniform on [-l_e,l_e]
    target_norm = math.sqrt(2 * k * var_e)  # ||(e1,e2)||, 2k coords total
    beta = usvp_beta_min(k, log2_q, target_norm)
    if beta is None:
        return float("inf")
    return (0.265 if quantum else 0.292) * beta


def min_k_for_lambda(log2_q, l_e, lam, k_lo=200, k_hi=20000):
    """Binary search smallest k with bkz_security_bits >= lam.
    (Verified numerically monotone increasing in k for fixed log2_q, l_e.)"""
    if bkz_security_bits(k_hi, log2_q, l_e) < lam:
        return None
    lo, hi = k_lo, k_hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bkz_security_bits(mid, log2_q, l_e) >= lam:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------
# 2) Acceptance rate -- paper's Proposition 1 / Corollary 1
#    (verified to reproduce the paper's tau=0.99566 etc. exactly)
# ---------------------------------------------------------------------

def acceptance_rate(k, q, p, l_e, omega_c, ell):
    omega = l_e * omega_c
    var_e = (l_e * (l_e + 1)) / 3.0
    sigma = math.sqrt(omega_c * var_e)
    p_hat = [0.5, 0.15865, 0.02275, 0.00135]
    n0 = math.ceil(sigma * 1) - math.ceil(sigma * 0)
    n1 = math.ceil(sigma * 2) - math.ceil(sigma * 1)
    n2 = math.ceil(sigma * 3) - math.ceil(sigma * 2)
    n = [0, 0, 0, 0]
    n[0] = min(omega, n0)
    n[1] = min(max(0, omega - n[0]), n1)
    n[2] = min(max(0, omega - n[0] - n[1]), n2)
    n[3] = max(0, omega - sum(n[:3]))
    rho = sum(n[j] * p_hat[j] for j in range(4))
    base = 1 - (2 * (ell + 1) * rho / q)
    if base <= 0:
        return 0.0
    return base ** (2 * k)


# ---------------------------------------------------------------------
# 3) Collision resistance -- Section 5.2.1: omega_c + log2 C(k,omega_c) >= 2*lambda
# ---------------------------------------------------------------------

def log2_comb(k, omega_c):
    return (math.lgamma(k + 1) - math.lgamma(omega_c + 1)
            - math.lgamma(k - omega_c + 1)) / math.log(2)


def collision_bits(k, omega_c):
    return omega_c + log2_comb(k, omega_c)


# ---------------------------------------------------------------------
# 4) Forgery resistance -- Section 5.2.2 / Proposition 2: k*log2(q/p) >= lambda
# ---------------------------------------------------------------------

def forgery_bits(k, p, q):
    return k * math.log2(q / p)


# ---------------------------------------------------------------------
# 5) Sizes -- Section 6 (verified to reproduce the paper's byte counts exactly)
# ---------------------------------------------------------------------

def sizes_bytes(k, q, l_e, lam):
    """Uses ceil(log2(q)) per the paper's stated formulas exactly (Section 6):
    PK = ceil((k*ceil(log2 q) + 2*lambda)/8), etc. Passing raw (non-ceiled)
    log2(q) here was a bug in an earlier version of this script -- it silently
    undercounts by up to a full bit/coordinate whenever q sits just above a
    power-of-two boundary. Verified against the paper's own three rows: now
    reproduces PK/SK/Sig exactly for rows 1-2, and SK/Sig exactly for row 3
    (row 3's published PK=7520 looks like a copy-paste typo in the paper's
    Table 2, since it's identical to that row's Sig, which the formula says
    cannot happen -- Sig is always PK plus a positive extra term)."""
    log2_q_ceil = math.ceil(math.log2(q))
    log2_le = math.ceil(math.log2(2 * l_e + 1))
    pk = math.ceil((k * log2_q_ceil + 2 * lam) / 8)
    sk = math.ceil((k * log2_le) / 8)
    sg = math.ceil((k * log2_q_ceil + 2 * lam + k * 2) / 8)
    return pk, sk, sg


# ---------------------------------------------------------------------
# 6) q,p generation: q = ell*p + 1, q & p prime
#    (this is what HQCS-R-2 / HQCS-R-3's published numbers satisfy exactly;
#    HQCS-R-1's published p looks like a transcription/OCR error against
#    this relationship -- see analysis notes)
# ---------------------------------------------------------------------

def find_q_p(log2_q, target_ell, tol=1.5, max_tries=4000):
    p_target = max(3, int(2 ** log2_q / target_ell))
    p = nextprime(p_target)
    for _ in range(max_tries):
        q = target_ell * p + 1
        if abs(math.log2(q) - log2_q) < tol and isprime(q):
            return q, p
        p = nextprime(p)
    return None, None


# ---------------------------------------------------------------------
# 7) Full joint search for a target security level
# ---------------------------------------------------------------------

def find_best(lam, log2_q_range, l_e_range=(2, 3, 4),
              ell_range=(100, 130, 160, 200, 256), margin=1.0):
    """margin>1.0 requires bkz_security_bits >= margin*lam while still
    reporting/sizing for the true target lam (used to mirror the paper's
    own ~1.10-1.13x apparent safety margin under this same estimator)."""
    best = None
    search_lam = lam * margin
    for log2_q in log2_q_range:
        for ell in ell_range:
            q, p = find_q_p(log2_q, ell)
            if q is None:
                continue
            for l_e in l_e_range:
                k = min_k_for_lambda(math.log2(q), l_e, search_lam)
                if k is None:
                    continue

                found_omega = None
                omega_c = max(10, int(k * 0.01))
                omega_c_max = int(k * 0.2)
                while omega_c < omega_c_max:
                    tau = acceptance_rate(k, q, p, l_e, omega_c, ell)
                    coll = collision_bits(k, omega_c)
                    if tau >= 0.99 and coll >= 2 * lam:
                        found_omega = omega_c
                        break
                    omega_c += 1
                if found_omega is None:
                    continue

                fbits = forgery_bits(k, p, q)
                if fbits < lam:
                    continue

                pk, sk, sg = sizes_bytes(k, q, l_e, lam)
                total = pk + sg
                sec_bits = bkz_security_bits(k, math.log2(q), l_e)
                cand = dict(lam=lam, k=k, q=q, p=p, ell=ell, l_e=l_e,
                            omega_c=found_omega,
                            tau=acceptance_rate(k, q, p, l_e, found_omega, ell),
                            coll=collision_bits(k, found_omega), forge=fbits,
                            sec_bits=sec_bits, pk=pk, sk=sk, sg=sg, total=total)
                if best is None or total < best["total"]:
                    best = cand
    return best


def sanity_check():
    """Reproduce the paper's own published HQCS-R-2 row through this pipeline."""
    k, q, p, ell, l_e, omega_c = 1511, 2147446991, 16518823, 130, 3, 71
    print("=== Sanity check against paper's published HQCS-R-2 row ===")
    print(f"acceptance rate: {acceptance_rate(k, q, p, l_e, omega_c, ell):.5f}"
          f"  (paper: 0.99566)")
    print(f"collision bits:  {collision_bits(k, omega_c):.1f}  (need >= 256)")
    print(f"forgery bits:    {forgery_bits(k, p, q):.1f}  (need >= 128)")
    print(f"BKZ security (this estimator): {bkz_security_bits(k, math.log2(q), l_e):.1f} bits"
          f"  (stated target: 128 -> ~1.13x margin)")


if __name__ == "__main__":
    sanity_check()

    print("\n=== Extended parameter table (margin-matched to the paper's own rows) ===")
    header = f"{'lambda':>7} {'k':>6} {'q':>14} {'p':>12} {'ell':>5} {'l_e':>4} " \
             f"{'omega_c':>8} {'tau':>8} {'PK(B)':>7} {'SK(B)':>6} {'Sig(B)':>7} {'sec~bits':>9}"
    print(header)
    for lam, rng in [(128, range(28, 36)), (192, range(28, 42)), (256, range(30, 50))]:
        b = find_best(lam, rng, margin=1.10)
        if b is None:
            print(f"{lam:>7}  -- no candidate found in search range --")
            continue
        print(f"{b['lam']:>7} {b['k']:>6} {b['q']:>14} {b['p']:>12} {b['ell']:>5} "
              f"{b['l_e']:>4} {b['omega_c']:>8} {b['tau']:>8.5f} {b['pk']:>7} "
              f"{b['sk']:>6} {b['sg']:>7} {b['sec_bits']:>9.1f}")