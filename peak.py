"""Find the peak bitstring of a small OpenQASM 2 circuit by exact statevector simulation.

Usage:
    python peak.py CIRCUIT.qasm [--top K]

Supports circuits up to 28 qubits (memory grows as 2**n complex numbers).
Unknown gates or syntax raise an error that names the offending line.
"""

import argparse
import math
import re
import sys

import numpy as np

MAX_QUBITS = 28

SQ = 1 / math.sqrt(2)


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


def _rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _phase(t):
    return np.array([[1, 0], [0, np.exp(1j * t)]], dtype=complex)


FIXED_1Q = {
    "id": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[SQ, SQ], [SQ, -SQ]], dtype=complex),
    "s": _phase(math.pi / 2),
    "sdg": _phase(-math.pi / 2),
    "t": _phase(math.pi / 4),
    "tdg": _phase(-math.pi / 4),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex),
}

PARAM_1Q = {
    "rx": (1, _rx),
    "ry": (1, _ry),
    "rz": (1, _rz),
    "p": (1, _phase),
    "u1": (1, _phase),
    "u2": (2, lambda phi, lam: _u3(math.pi / 2, phi, lam)),
    "u3": (3, _u3),
    "u": (3, _u3),
}

# Two-qubit controlled gates: name -> (param count, target 1q matrix builder)
CONTROLLED = {
    "cx": (0, lambda: FIXED_1Q["x"]),
    "cy": (0, lambda: FIXED_1Q["y"]),
    "cz": (0, lambda: FIXED_1Q["z"]),
    "ch": (0, lambda: FIXED_1Q["h"]),
    "crx": (1, _rx),
    "cry": (1, _ry),
    "crz": (1, _rz),
    "cp": (1, _phase),
    "cu1": (1, _phase),
    "cu3": (3, _u3),
}

SKIP = ("OPENQASM", "include", "barrier", "creg", "measure")


class QasmError(ValueError):
    pass


def _eval_param(expr, lineno):
    if not re.fullmatch(r"[0-9eE+\-*/(). pi]*", expr):
        raise QasmError(f"line {lineno}: unsupported parameter expression {expr!r}")
    try:
        return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))
    except Exception as exc:
        raise QasmError(f"line {lineno}: cannot evaluate parameter {expr!r}: {exc}") from exc


def parse(text):
    """Return (n_qubits, ops). Each op is (name, params, qubit indices)."""
    text = re.sub(r"//[^\n]*", "", text)
    regs = {}
    n = 0
    ops = []
    for lineno, raw in enumerate(text.split(";"), start=1):
        stmt = " ".join(raw.split())
        if not stmt or stmt.startswith(SKIP):
            continue
        m = re.fullmatch(r"qreg (\w+)\[(\d+)\]", stmt)
        if m:
            regs[m.group(1)] = (n, int(m.group(2)))
            n += int(m.group(2))
            continue
        m = re.fullmatch(r"(\w+)\s*(?:\(([^)]*)\))?\s+(.+)", stmt)
        if not m:
            raise QasmError(f"statement {lineno}: cannot parse {stmt!r}")
        name, params_s, args_s = m.group(1), m.group(2), m.group(3)
        params = [_eval_param(p.strip(), lineno) for p in params_s.split(",")] if params_s else []
        qubits = []
        for arg in args_s.split(","):
            am = re.fullmatch(r"\s*(\w+)\[(\d+)\]\s*", arg)
            if not am or am.group(1) not in regs:
                raise QasmError(f"statement {lineno}: bad qubit argument {arg.strip()!r}")
            base, size = regs[am.group(1)]
            idx = int(am.group(2))
            if idx >= size:
                raise QasmError(f"statement {lineno}: index {idx} out of range for {am.group(1)}")
            qubits.append(base + idx)
        ops.append((name, params, qubits, lineno))
    if n == 0:
        raise QasmError("no qreg declared")
    return n, ops


def _apply_1q(state, n, mat, q):
    # Axis for qubit q: state reshaped to (2,)*n with axis 0 = qubit n-1.
    psi = state.reshape((2,) * n)
    ax = n - 1 - q
    psi = np.tensordot(mat, psi, axes=([1], [ax]))
    return np.moveaxis(psi, 0, ax).reshape(-1)


def _apply_controlled(state, n, mat, c, t):
    psi = state.reshape((2,) * n).copy()
    ac, at = n - 1 - c, n - 1 - t
    idx = [slice(None)] * n
    idx[ac] = 1
    sub = psi[tuple(idx)]
    at_sub = at if at < ac else at - 1
    sub = np.moveaxis(np.tensordot(mat, sub, axes=([1], [at_sub])), 0, at_sub)
    psi[tuple(idx)] = sub
    return psi.reshape(-1)


def _apply_swap(state, n, a, b):
    psi = state.reshape((2,) * n)
    return np.swapaxes(psi, n - 1 - a, n - 1 - b).reshape(-1)


def simulate(n, ops):
    if n > MAX_QUBITS:
        raise QasmError(f"{n} qubits exceeds the exact-simulation limit of {MAX_QUBITS}")
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for name, params, qubits, lineno in ops:
        if name in FIXED_1Q and not params and len(qubits) == 1:
            state = _apply_1q(state, n, FIXED_1Q[name], qubits[0])
        elif name in PARAM_1Q and len(qubits) == 1 and len(params) == PARAM_1Q[name][0]:
            state = _apply_1q(state, n, PARAM_1Q[name][1](*params), qubits[0])
        elif name in CONTROLLED and len(qubits) == 2 and len(params) == CONTROLLED[name][0]:
            if qubits[0] == qubits[1]:
                raise QasmError(f"statement {lineno}: {name} control equals target")
            state = _apply_controlled(state, n, CONTROLLED[name][1](*params), qubits[0], qubits[1])
        elif name == "swap" and len(qubits) == 2 and not params:
            state = _apply_swap(state, n, qubits[0], qubits[1])
        else:
            raise QasmError(
                f"statement {lineno}: unsupported gate {name!r} with "
                f"{len(params)} params on {len(qubits)} qubits"
            )
    return state


def top_bitstrings(state, n, k):
    probs = np.abs(state) ** 2
    order = np.argsort(probs)[::-1][:k]
    # Qiskit order: qubit n-1 leftmost. Reversed: qubit 0 leftmost.
    return [(format(i, f"0{n}b"), format(i, f"0{n}b")[::-1], float(probs[i])) for i in order]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find the peak bitstring of a small OpenQASM 2 circuit.")
    ap.add_argument("circuit")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args(argv)
    with open(args.circuit, encoding="utf-8") as fh:
        n, ops = parse(fh.read())
    state = simulate(n, ops)
    norm = float(np.vdot(state, state).real)
    print(f"qubits={n} gates={len(ops)} norm={norm:.12f}")
    print("rank  qiskit_order(q[n-1]..q[0])  qubit0_first  probability")
    for r, (qk, rev, p) in enumerate(top_bitstrings(state, n, args.top), start=1):
        print(f"{r:<5} {qk:<27} {rev:<13} {p:.6f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QasmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
