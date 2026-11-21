"""Exact statevector observables for OTOC echo circuits (OpenQASM 2, up to 30 qubits).

Applies gates in place: consecutive single-qubit gates on a qubit are fused into one 2x2
matrix, and cz is a sign flip on a strided view, so a 25-qubit circuit fits in about 1 GB.

Usage:
    python otoc.py CIRCUIT.qasm [CIRCUIT.qasm ...] [--json OUT.json] [--strip-perturbations]

For each circuit it reports P(all zeros), <Z_q> for every qubit, and the most probable
bitstring (Qiskit order, highest qubit leftmost). --strip-perturbations drops the bare
x and z gates, which turns an echo circuit back into the identity (a sanity check).
"""

import argparse
import json
import sys
import time

import numpy as np

import peak

MAX_QUBITS = 30


def _gate_matrix(name, params):
    if name in peak.FIXED_1Q and not params:
        return peak.FIXED_1Q[name]
    if name in peak.PARAM_1Q and len(params) == peak.PARAM_1Q[name][0]:
        return peak.PARAM_1Q[name][1](*params)
    return None


def compile_ops(n, ops, strip_perturbations=False):
    """Fuse runs of 1q gates. Returns a list of ('u', q, mat) and ('cz', a, b)."""
    pending = {}
    out = []

    def flush(q):
        m = pending.pop(q, None)
        if m is not None and not np.allclose(m, np.eye(2)):
            out.append(("u", q, m))

    for name, params, qubits, lineno in ops:
        if strip_perturbations and name in ("x", "z") and not params:
            continue
        if len(qubits) == 1:
            mat = _gate_matrix(name, params)
            if mat is None:
                raise peak.QasmError(f"statement {lineno}: unsupported gate {name!r}")
            q = qubits[0]
            pending[q] = mat @ pending.get(q, np.eye(2, dtype=complex))
        elif name == "cz" and len(qubits) == 2 and not params:
            a, b = qubits
            if a == b:
                raise peak.QasmError(f"statement {lineno}: cz on one qubit")
            flush(a)
            flush(b)
            out.append(("cz", min(a, b), max(a, b)))
        else:
            raise peak.QasmError(
                f"statement {lineno}: unsupported gate {name!r} on {len(qubits)} qubits"
            )
    for q in sorted(pending):
        flush(q)
    return out


def run(n, compiled, dtype=np.complex128):
    if n > MAX_QUBITS:
        raise peak.QasmError(f"{n} qubits exceeds the limit of {MAX_QUBITS}")
    psi = np.zeros(2 ** n, dtype=dtype)
    psi[0] = 1.0
    for op in compiled:
        if op[0] == "u":
            _, q, m = op
            v = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
            x0 = v[:, 0, :].copy()
            x1 = v[:, 1, :]
            new1 = m[1, 0] * x0 + m[1, 1] * x1
            v[:, 0, :] = m[0, 0] * x0 + m[0, 1] * x1
            v[:, 1, :] = new1
        else:
            _, a, b = op
            v = psi.reshape(2 ** (n - 1 - b), 2, 2 ** (b - 1 - a), 2, 2 ** a)
            v[:, 1, :, 1, :] *= -1
    return psi


def observables(n, psi):
    probs = np.abs(psi) ** 2
    z = []
    for q in range(n):
        p = probs.reshape(2 ** (n - 1 - q), 2, 2 ** q).sum(axis=(0, 2))
        z.append(float(p[0] - p[1]))
    top = int(np.argmax(probs))
    return {
        "qubits": n,
        "norm": float(probs.sum()),
        "p_all_zero": float(probs[0]),
        "z": z,
        "top_bitstring": format(top, f"0{n}b"),
        "top_probability": float(probs[top]),
    }


def evaluate(path, strip_perturbations=False):
    with open(path, encoding="utf-8") as fh:
        n, ops = peak.parse(fh.read())
    t0 = time.perf_counter()
    compiled = compile_ops(n, ops, strip_perturbations)
    psi = run(n, compiled)
    res = observables(n, psi)
    res.update(file=path, gates=len(ops), fused_ops=len(compiled),
               seconds=round(time.perf_counter() - t0, 2))
    return res


def prefix_z(path, dtype=np.complex128):
    """Simulate only the statements before the first bare z gate; return <Z_q> for all q.

    For an echo circuit V, U, X, U^dagger, z q[k], U, X, U^dagger, V^dagger the second half is
    the inverse of the first, so P(all zeros) of circuit k equals prefix_z(...)[k] ** 2.
    """
    with open(path, encoding="utf-8") as fh:
        n, ops = peak.parse(fh.read())
    cut = next((i for i, op in enumerate(ops) if op[0] == "z" and not op[1]), None)
    if cut is None:
        raise peak.QasmError(f"{path}: no bare z gate to split on")
    t0 = time.perf_counter()
    psi = run(n, compile_ops(n, ops[:cut]), dtype=dtype)
    res = observables(n, psi)
    res.update(file=path, split_statement=ops[cut][3], z_qubit=ops[cut][2][0],
               seconds=round(time.perf_counter() - t0, 2))
    return res


def prefix_scan(paths, dtype=np.complex128, log=None):
    """P(all zeros) for a family of echo circuits that differ only in one bare z gate.

    Checks that every file has the same non-z statements, then runs that shared sequence
    once, reading <Z_k> at each file's own z position. Returns {path: (k, <Z_k>, p0)}.
    """
    seqs = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            n, ops = peak.parse(fh.read())
        zs = [i for i, op in enumerate(ops) if op[0] == "z" and not op[1]]
        if len(zs) != 1:
            raise peak.QasmError(f"{p}: expected exactly one bare z gate, found {len(zs)}")
        i = zs[0]
        rest = [(op[0], tuple(op[1]), tuple(op[2])) for op in ops[:i] + ops[i + 1:]]
        seqs[p] = (n, ops[:i] + ops[i + 1:], rest, i, ops[i][2][0])
    first = next(iter(seqs.values()))
    for p, s in seqs.items():
        if s[0] != first[0] or s[2] != first[2]:
            raise peak.QasmError(f"{p}: non-z statements differ from {next(iter(seqs))}")
    n, common = first[0], first[1]
    cuts = sorted((s[3], p, s[4]) for p, s in seqs.items())
    psi = np.zeros(2 ** n, dtype=dtype)
    psi[0] = 1.0
    done = 0
    out = {}
    t0 = time.perf_counter()
    for cut, p, k in cuts:
        if cut > done:
            psi = _run_from(n, psi, compile_ops(n, common[done:cut]))
            done = cut
        probs = np.abs(psi.reshape(2 ** (n - 1 - k), 2, 2 ** k)) ** 2
        p0, p1 = probs[:, 0, :].sum(), probs[:, 1, :].sum()
        zk = float(p0 - p1)
        out[p] = (k, zk, zk * zk)
        if log:
            log(f"{p} k={k} cut={cut} Z={zk:+.12f} p0={zk * zk:.12f} "
                f"t={time.perf_counter() - t0:.1f}s")
    return out


def _run_from(n, psi, compiled):
    for op in compiled:
        if op[0] == "u":
            _, q, m = op
            v = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
            x0 = v[:, 0, :].copy()
            x1 = v[:, 1, :]
            new1 = m[1, 0] * x0 + m[1, 1] * x1
            v[:, 0, :] = m[0, 0] * x0 + m[0, 1] * x1
            v[:, 1, :] = new1
        else:
            _, a, b = op
            v = psi.reshape(2 ** (n - 1 - b), 2, 2 ** (b - 1 - a), 2, 2 ** a)
            v[:, 1, :, 1, :] *= -1
    return psi


def main(argv=None):
    ap = argparse.ArgumentParser(description="Exact observables for OTOC echo circuits.")
    ap.add_argument("circuits", nargs="+")
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--strip-perturbations", action="store_true")
    ap.add_argument("--scan", action="store_true",
                    help="one pass over a family that differs only in its z gate (p0 only)")
    args = ap.parse_args(argv)
    if args.scan:
        res = prefix_scan(args.circuits, log=lambda s: print(s, flush=True))
        if args.json:
            rows = [{"file": p, "k": k, "z": z, "p_all_zero": p0}
                    for p, (k, z, p0) in sorted(res.items(), key=lambda kv: kv[1][0])]
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(rows, fh, indent=1)
        return 0
    results = []
    for path in args.circuits:
        r = evaluate(path, args.strip_perturbations)
        results.append(r)
        zs = " ".join(f"{v:+.6f}" for v in r["z"])
        print(f"{path} n={r['qubits']} norm={r['norm']:.9f} p0={r['p_all_zero']:.9f} "
              f"t={r['seconds']}s Z=[{zs}]", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except peak.QasmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
