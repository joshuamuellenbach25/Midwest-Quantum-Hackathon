import numpy as np
import pytest

import peak


def run(qasm):
    n, ops = peak.parse(qasm)
    return n, peak.simulate(n, ops)


HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_x_on_qubit0_sets_rightmost_bit():
    n, st = run(HEAD + "qreg q[3];\nx q[0];\n")
    qk, rev, p = peak.top_bitstrings(st, n, 1)[0]
    assert (qk, rev) == ("001", "100")
    assert p == pytest.approx(1.0)


def test_bell_state_split():
    n, st = run(HEAD + "qreg q[2];\nh q[0];\ncx q[0],q[1];\n")
    probs = np.abs(st) ** 2
    assert probs[0] == pytest.approx(0.5)
    assert probs[3] == pytest.approx(0.5)


def test_hidden_string_roundtrip():
    # H on all, then phase oracle pattern, then H on all, recovers 1011 (Bernstein-Vazirani).
    secret = "1011"  # qubit 3..0
    lines = [HEAD + "qreg q[5];", "x q[4];", "h q[4];"]
    lines += [f"h q[{i}];" for i in range(4)]
    lines += [f"cx q[{i}],q[4];" for i, b in enumerate(reversed(secret)) if b == "1"]
    lines += [f"h q[{i}];" for i in range(4)]
    n, st = run("\n".join(lines) + "\n")
    probs = (np.abs(st) ** 2).reshape(2, 16).sum(axis=0)
    assert format(int(np.argmax(probs)), "04b") == secret
    assert probs.max() == pytest.approx(1.0)


def test_parametric_and_pi_expressions():
    n, st = run(HEAD + "qreg q[1];\nrx(pi) q[0];\n")
    assert abs(st[1]) ** 2 == pytest.approx(1.0)
    n, st = run(HEAD + "qreg q[1];\nu3(pi/2,0,pi) q[0];\n")
    assert np.abs(st) ** 2 == pytest.approx([0.5, 0.5])


def test_controlled_target_above_control():
    n, st = run(HEAD + "qreg q[3];\nx q[2];\ncx q[2],q[0];\n")
    assert peak.top_bitstrings(st, n, 1)[0][0] == "101"


def test_unknown_gate_is_refused_with_name():
    with pytest.raises(peak.QasmError, match="'foo'"):
        run(HEAD + "qreg q[1];\nfoo q[0];\n")


def test_bad_parameter_is_refused():
    with pytest.raises(peak.QasmError, match="unsupported parameter"):
        run(HEAD + "qreg q[1];\nrz(__import__) q[0];\n")


def test_norm_preserved_random_circuit():
    rng = np.random.default_rng(0)
    lines = [HEAD + "qreg q[6];"]
    for _ in range(80):
        a, b = rng.choice(6, 2, replace=False)
        lines.append(f"u3({rng.uniform(0, 6):.4f},{rng.uniform(0, 6):.4f},{rng.uniform(0, 6):.4f}) q[{a}];")
        lines.append(f"cz q[{a}],q[{b}];")
    n, st = run("\n".join(lines) + "\n")
    assert np.vdot(st, st).real == pytest.approx(1.0)
