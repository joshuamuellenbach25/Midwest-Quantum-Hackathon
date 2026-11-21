import numpy as np
import pytest

import otoc
import peak

HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def random_circuit(n, depth, seed, perturb=False):
    rng = np.random.default_rng(seed)
    lines = [HEAD + f"qreg q[{n}];"]
    for _ in range(depth):
        a, b = rng.choice(n, 2, replace=False)
        for q in (a, b):
            t, p, l = rng.uniform(-3, 3, 3)
            lines.append(f"u3({t:.6f},{p:.6f},{l:.6f}) q[{q}];")
        lines.append(f"cz q[{a}],q[{b}];")
    if perturb:
        lines.append("x q[0];")
        lines.append(f"z q[{n - 1}];")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("n,seed", [(3, 1), (5, 2), (7, 3)])
def test_matches_reference_simulator(n, seed):
    text = random_circuit(n, 40, seed, perturb=True)
    n1, ops = peak.parse(text)
    ref = peak.simulate(n1, ops)
    fast = otoc.run(n1, otoc.compile_ops(n1, ops))
    assert np.allclose(ref, fast, atol=1e-10)


def test_z_expectation_signs():
    n, ops = peak.parse(HEAD + "qreg q[3];\nx q[1];\n")
    r = otoc.observables(n, otoc.run(n, otoc.compile_ops(n, ops)))
    assert r["z"] == pytest.approx([1.0, -1.0, 1.0])
    assert r["p_all_zero"] == pytest.approx(0.0)
    assert r["top_bitstring"] == "010"


def test_strip_perturbations_restores_echo_identity():
    fwd = random_circuit(4, 20, 9)
    n, ops = peak.parse(fwd)
    # Build U, x q0, U^dagger by inverting fused ops: U^dagger reverses order and conjugates.
    comp = otoc.compile_ops(n, ops)
    inv = [("u", q, m.conj().T) if k == "u" else (k, q, m) for k, q, m in reversed(comp)]
    psi = otoc.run(n, comp + inv)
    assert abs(psi[0]) ** 2 == pytest.approx(1.0)


def _echo_family(tmp_path, n=4, seed=5):
    rng = np.random.default_rng(seed)
    u = []
    for _ in range(12):
        a, b = rng.choice(n, 2, replace=False)
        t, p, l = rng.uniform(-3, 3, 3)
        u.append(f"u3({t:.6f},{p:.6f},{l:.6f}) q[{a}];")
        u.append(f"cz q[{a}],q[{b}];")
    u_inv = []
    for line in reversed(u):
        if line.startswith("u3"):
            t, p, l = (float(x) for x in line[3:line.index(")")].split(","))
            q = line[line.index("q["):]
            u_inv.append(f"u3({-t:.6f},{-l:.6f},{-p:.6f}) {q}")
        else:
            u_inv.append(line)
    paths = []
    for k in range(n):
        body = u + ["x q[0];"] + u_inv + [f"z q[{k}];"] + u + ["x q[0];"] + u_inv
        path = tmp_path / f"fam_{k}.qasm"
        path.write_text(HEAD + f"qreg q[{n}];\n" + "\n".join(body) + "\n")
        paths.append(str(path))
    return paths


def test_prefix_scan_matches_full_simulation(tmp_path):
    paths = _echo_family(tmp_path)
    scan = otoc.prefix_scan(paths)
    for p in paths:
        full = otoc.evaluate(p)["p_all_zero"]
        assert scan[p][2] == pytest.approx(full, abs=1e-10)


def test_prefix_scan_refuses_mismatched_family(tmp_path):
    paths = _echo_family(tmp_path)
    text = open(paths[1]).read().replace("x q[0];", "x q[1];", 1)
    open(paths[1], "w").write(text)
    with pytest.raises(peak.QasmError, match="differ"):
        otoc.prefix_scan(paths)


def test_unsupported_gate_refused():
    n, ops = peak.parse(HEAD + "qreg q[2];\ncx q[0],q[1];\n")
    with pytest.raises(peak.QasmError, match="'cx'"):
        otoc.compile_ops(n, ops)
