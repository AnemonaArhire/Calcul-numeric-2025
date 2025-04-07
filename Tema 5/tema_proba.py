import numpy as np
import random
from collections import defaultdict
from scipy.linalg import svd
import matplotlib.pyplot as plt

# ===============================
# Tema 5 - Analiza matricelor
# ===============================

def generate_sparse_symmetric_matrix_list(n, density=0.01):
    matrix = defaultdict(list)
    for i in range(n):
        for j in range(i, n):
            if random.random() < density:
                val = random.uniform(0.1, 10)
                matrix[i].append((j, val))
                if i != j:
                    matrix[j].append((i, val))
    return matrix


def read_sparse_matrix_list_custom(file_path):
    matrix = defaultdict(list)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    for line in lines[1:]:
        parts = line.strip().replace(',', '.').split()
        if len(parts) == 3:
            try:
                val = float(parts[0].strip('.'))
                i = int(float(parts[1]))
                j = int(float(parts[2]))
                matrix[i].append((j, val))
                if i != j:
                    matrix[j].append((i, val))  # Simetrie
            except ValueError:
                continue
    return matrix, n


def is_symmetric(matrix):
    for i, neighbors in matrix.items():
        for j, val in neighbors:
            if not any(i2 == i and abs(v2 - val) < 1e-9 for i2, v2 in matrix.get(j, [])):
                return False
    return True


def power_method_list(matrix, n, epsilon=1e-9, kmax=1000000):
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    for k in range(kmax):
        w = np.zeros(n)
        for i in matrix:
            for j, val in matrix[i]:
                w[i] += val * v[j]

        lambda_ = np.dot(w, v)
        v_new = w / np.linalg.norm(w)

        if np.linalg.norm(w - lambda_ * v) < n * epsilon:
            return lambda_, v_new, k
        v = v_new

    return lambda_, v, k


def compute_norm_list(matrix, v, lambda_max):
    n = len(v)
    Av = np.zeros(n)
    for i in matrix:
        for j, val in matrix[i]:
            Av[i] += val * v[j]
    return np.linalg.norm(Av - lambda_max * v)


def svd_analysis(A, b, epsilon=1e-9):
    U, S, Vt = svd(A, full_matrices=False)
    r = np.sum(S > epsilon)
    cond = S[0] / S[r - 1] if r > 0 else np.inf

    S_inv = np.diag([1/s if s > epsilon else 0 for s in S])
    A_inv = Vt.T @ S_inv @ U.T
    x_I = A_inv @ b
    residual = np.linalg.norm(b - A @ x_I)

    return S, r, cond, A_inv, x_I, residual


# ===============================
# Exemplu de rulare completă
# ===============================
if __name__ == "__main__":
    # Cazul 1: matrice rară simetrică citită din fișier
    matrix_file, n_file = read_sparse_matrix_list_custom("m_rar_sim.txt")

    print("✅ Matrice citită din fișier.")
    print("Dimensiune:", n_file)
    print("Este simetrică?", is_symmetric(matrix_file))

    lambda_max, u_max, iters = power_method_list(matrix_file, n_file)
    residual_norm = compute_norm_list(matrix_file, u_max, lambda_max)

    print(f"Valoare proprie maximă: {lambda_max}")
    print(f"Norma ||Au - λu||: {residual_norm}")
    print(f"Număr de iterații: {iters}")

    # Cazul 2: analiză SVD pentru p > n
    p, n = 600, 400
    A_svd = np.random.rand(p, n)
    b_svd = np.random.rand(p)

    S, r, cond, A_inv, x_I, residual = svd_analysis(A_svd, b_svd)

    print("\n✅ Analiză SVD pentru matrice densă (p > n)")
    print("Valori singulare (primele 10):", S[:10])
    print("Rangul:", r)
    print("Numărul de condiționare:", cond)
    print("Norma ||b - Ax||:", residual)

    # Salvare grafic
    plt.figure(figsize=(10, 4))
    plt.plot(S, marker='o')
    plt.title("Spectrul valorilor singulare (SVD)")
    plt.xlabel("Index")
    plt.ylabel("Valoare singulară")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("spectru_valori_singulare.png")
    plt.close()
