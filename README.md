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
