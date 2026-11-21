# Midwest Quantum Hackathon

Tools for finding the peak bitstring of peaked quantum circuits.

## peak.py

Exact statevector simulation of an OpenQASM 2 circuit (up to 28 qubits) with NumPy. It prints
the most probable bitstrings in two orders: Qiskit order (highest qubit leftmost) and qubit 0
first. Unknown gates or syntax stop the run with an error that names the statement.

```
pip install numpy pytest
python peak.py circuit.qasm --top 5
python -m pytest -q test_peak.py
```

## otoc.py

Exact observables for OTOC echo circuits (prep, U, X, U^dagger, Z on qubit k, U, X, U^dagger,
inverse prep) built from u3, cz, x and z gates, up to 30 qubits. Single-qubit gates are fused
and applied in place, so a 25-qubit state needs about 1 GB.

- `python otoc.py FILES... --json out.json` simulates each circuit fully and reports
  P(all zeros) and <Z_q> for every qubit.
- `python otoc.py --scan FILES... --json out.json` handles a family of circuits that differ only
  in the qubit of their Z gate. P(all zeros) of circuit k equals <Z_k>^2 on the state just
  before its Z, so one pass over the shared gate sequence gives every member. The tool refuses
  a family whose other gates differ.

```
python -m pytest -q test_otoc.py test_peak.py
```
