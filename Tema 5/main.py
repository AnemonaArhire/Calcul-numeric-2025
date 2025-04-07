import numpy as np
import random

def generate_sparse_symmetric_matrix(n, density=0.1):
    matrix = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if random.random() < density:
                value = random.uniform(0.1, 10)
                matrix[i].append((j, value))
                if i != j:
                    matrix[j].append((i, value))
    return matrix

def read_sparse_matrix_list_from_file(file_path):
    """
    Citește matricea rară din fișier, prima linie conține dimensiunea n,
    apoi fiecare linie conține: valoare, i, j
    """
    matrix = {}
    with open(file_path, 'r') as f:
        n = int(f.readline().strip())  # prima linie conține dimensiunea matricei
        for line in f:
            if not line.strip():
                continue  # sare peste linii goale
            parts = line.strip().replace(',', '.').split()
            if len(parts) != 3:
                continue  # sărim liniile incorecte
            val, i, j = parts
            val = float(val.strip().replace(',', '.').strip('.'))

            i = int(float(i)) - 1
            j = int(float(j)) - 1
            matrix[(i, j)] = val
            if i != j:
                matrix[(j, i)] = val  # păstrăm simetria
    return matrix, n


def power_method(A, n, epsilon=1e-9, kmax=1000000):
    """
    Implementarea metodei puterii pentru a calcula valoarea proprie de modul maxim
    A: matricea rară simetrică (stocată sub forma unui dicționar)
    n: dimensiunea matricei
    epsilon: precizia calculului
    kmax: numărul maxim de iterații
    """
    # Alegem un vector inițial aleator
    v = np.random.rand(n)
    v /= np.linalg.norm(v)  # normalizăm vectorul

    lambda_k = 0
    for k in range(kmax):
        w = np.zeros(n)
        # Calculăm produsul matricei sparse cu vectorul
        for (i, j), value in A.items():
            w[i] += value * v[j]  # A[i,j] * v[j]
            if i != j:
                w[j] += value * v[i]  # A[j,i] * v[i] pentru simetrie

        # Calculăm valoarea proprie folosind coeficientul Rayleigh
        lambda_new = np.dot(w, v) / np.dot(v, v)
        v_new = w / np.linalg.norm(w)

        # Convergență
        if np.linalg.norm(w - lambda_new * v) < epsilon:
            return lambda_new, v_new

        v = v_new

    # Dacă nu s-a ajuns la convergență, returnăm valoarea aproximativă
    return lambda_k, v


def compute_norm(A, v, lambda_max, n):
    """
    Calcularea normei reziduurilor.
    A: matricea rară stocată sub formă de dicționar
    v: vectorul propriu
    lambda_max: valoarea proprie maximă
    n: dimensiunea matricei
    """
    # Calculăm reziduul
    residual = np.zeros(n)
    for i in range(n):
        # Calculăm produsul liniei i din A cu vectorul v
        for j, value in A.get(i, []):  # A.get(i, []) pentru a evita KeyError
            residual[i] += value * v[j]

    # Calculăm norma reziduurilor
    norma = np.linalg.norm(residual - lambda_max * v)
    return norma


def compute_svd(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(S > 1e-9)
    cond_number = max(S) / min(S[S > 1e-9])
    S_inv = np.diag(1 / S)
    A_inv = Vt.T @ S_inv.T @ U.T
    return U, S, Vt, rank, cond_number, A_inv

def solve_least_squares(A, b):
    U, S, Vt, _, _, A_inv = compute_svd(A)
    return A_inv @ b

def compute_error(A, b, x_I):
    return np.linalg.norm(b - A @ x_I)

def is_symmetric(matrix):
    for (i, j), val in matrix.items():
        if matrix.get((j, i), None) != val:
            return False
    return True


# === Exemplu de utilizare ===

# Pentru a citi din fișier:
file_path = "m_rar_sim"
A, n = read_sparse_matrix_list_from_file(file_path)

if is_symmetric(A):
    print("Matricea este simetrică. \n")
else:
    print("⚠ Matricea nu este simetrică. \n")


# Metoda puterii
lambda_max, v_max = power_method(A, n)
print("Cea mai mare valoare proprie:", lambda_max)
print("Vectorul propriu asociat:", v_max)

# Norma
norma = compute_norm(A, v_max, lambda_max, n)
print("\nNorma ∥A * u_max - λ_max * u_max∥:", norma)

# SVD pe o matrice completă aleatoare A_full (ex: p > n)
A_full = np.random.rand(n, n + 10)
b = np.random.rand(n)

U, S, Vt, rank, cond_number, A_inv = compute_svd(A_full)
print("\nValorile singulare:", S)
print("\nRangul matricei:", rank)
print("\nNumărul de condiționare:", cond_number)
print("\nPseudoinversa A:", A_inv)

# Soluția sistemului și eroare
x_I = solve_least_squares(A_full, b)
print("\nSoluția x_I:", x_I)
print("\nEroarea:", compute_error(A_full, b, x_I))
