import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

from kodeketak import build_feature_map


def fidelity_circuit(x_a, x_b, feature_map, x_a_p=None, x_b_p=None):
    qc_a = build_feature_map(feature_map=feature_map, data=x_a, data_p=x_a_p)
    qc_b = build_feature_map(feature_map=feature_map, data=x_b, data_p=x_b_p)

    num_qubits = qc_a.num_qubits

    qc = QuantumCircuit(num_qubits)
    qc.compose(qc_a, inplace=True)
    qc.compose(qc_b.inverse(), inplace=True)
    return qc


def get_fidelity(x_a, x_b, feature_map, x_a_p=None, x_b_p=None, backend="statevector", shots=1024, seed=42):
    qc = fidelity_circuit(x_a=x_a, x_b=x_b, feature_map=feature_map, x_a_p=x_a_p, x_b_p=x_b_p)
    if backend.lower() == "statevector":
        state = Statevector.from_instruction(qc)
        return np.abs(state.data[0]) ** 2
    else:
        print("Backend egokia aukeratu")

def get_kernel_matrix(X1, feature_map, X2=None, backend="statevector", shots=1024, X1_p=None, X2_p=None):
    # Caso 1: un solo dataset -> matriz simétrica
    if X2 is None:
        n = len(X1)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                value = get_fidelity(
                    x_a=X1[i],
                    x_b=X1[j],
                    feature_map=feature_map,
                    x_a_p=X1_p[i] if X1_p is not None else None,
                    x_b_p=X1_p[j] if X1_p is not None else None,
                    backend=backend,
                    shots=shots
                )

                K[i, j] = value
                K[j, i] = value

    # Caso 2: dos datasets -> matriz rectangular
    else:
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = get_fidelity(
                    x_a=X1[i],
                    x_b=X2[j],
                    feature_map=feature_map,
                    x_a_p=X1_p[i] if X1_p is not None else None,
                    x_b_p=X2_p[j] if X2_p is not None else None,
                    backend=backend,
                    shots=shots
                )

    return K

def save_kernel_heatmap(K, title, filename, cmap="cool"):
    plt.figure(figsize=(7,7))

    vmin = np.min(K)
    vmax = np.max(K)

    im = plt.imshow(
        K,
        cmap=cmap,
        aspect="auto",
        vmin = vmin,
        vmax=vmax
    )

    plt.colorbar(im)

    plt.title(title)
    plt.xlabel("Lagin indizea")
    plt.ylabel("Lagin indizea")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    # K[i,j] = exp(-gamma * ||x_i - x_j||²)
def kernel_alignment(K, y):
    y_pm = np.where(y == 0, -1, 1)      # convierte 0/1 a -1/+1
    K_ideal = np.outer(y_pm, y_pm).astype(float)
    num = np.sum(K * K_ideal)
    den = np.sqrt(np.sum(K ** 2) * np.sum(K_ideal ** 2))
    return num / den
