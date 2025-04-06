import numpy as np
import random

def generate_sparse_symmetric_matrix(n, density=0.1):
    """
    Generează o matrice simetrică, rară, cu valori nenule pozitive, de dimensiune n x n
    n: dimensiunea matricei
    density: densitatea valorilor nenule
    """
    matrix = {}  # structură de memorie pentru matricea rară

    for i in range(n):
        for j in range(i, n):  # doar partea superioară și diagonală
            if random.random() < density:  # alege dacă elementul va fi nenul
                value = random.uniform(0.1, 10)  # valoare pozitivă
                if i == j:  # valoare pe diagonală
                    matrix[(i, j)] = value
                else:  # valoare pe diagonală superioară
                    matrix[(i, j)] = value
                    matrix[(j, i)] = value  # simetric

    return matrix

def read_sparse_matrix_from_file(file_path, n):
    """
    Citește matricea rară din fișier
    file_path: calea fișierului cu tripletele (val, i, j)
    n: dimensiunea matricei
    """
    matrix = {}  # structură de memorie pentru matricea rară

    with open(file_path, 'r') as f:
        for line in f:
            val, i, j = map(int, line.split())  # citim tripletele
            i, j = i - 1, j - 1  # corectăm pentru indexare de la 0
            if i == j:
                matrix[(i, j)] = val
            else:
                matrix[(i, j)] = val
                matrix[(j, i)] = val  # matrice simetrică

    # Verificarea simetriei
    for (i, j), value in matrix.items():
        if matrix.get((j, i)) != value:
            raise ValueError("Matricea nu este simetrică!")

    return matrix

def power_method(A, n, epsilon=1e-9, kmax=1000000):
    """
    Implementarea metodei puterii pentru a calcula valoarea proprie de modul maxim
    A: matricea rară simetrică
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
        for i in range(n):
            for j in range(i, n):
                if (i, j) in A:
                    w[i] += A[(i, j)] * v[j]
                    if i != j:
                        w[j] += A[(i, j)] * v[i]

        # Calculăm valoarea proprie folosind coeficientul Rayleigh
        lambda_new = np.dot(w, v) / np.dot(v, v)
        v_new = w / np.linalg.norm(w)

        # Convergență
        if np.linalg.norm(w - lambda_new * v) < epsilon:
            return lambda_new, v_new

        v = v_new

    # Dacă nu s-a ajuns la convergență, returnăm valoarea aproximativă
    return lambda_k, v

def compute_svd(A):
    # Descompunerea SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Rangul matricei A
    rank = np.sum(S > 1e-9)

    # Numărul de condiționare al matricei A
    cond_number = max(S) / min(S[S > 1e-9])

    # Pseudoinversa Moore-Penrose
    S_inv = np.diag(1 / S)
    A_inv = Vt.T @ S_inv.T @ U.T

    return U, S, Vt, rank, cond_number, A_inv

def solve_least_squares(A, b):
    # Calculăm soluția sistemului Ax = b folosind pseudoinversa
    U, S, Vt, _, _, A_inv = compute_svd(A)
    x_I = A_inv @ b
    return x_I

def compute_error(A, b, x_I):
    return np.linalg.norm(b - A @ x_I)

def compute_norm(A, u_max, lambda_max, n):
    """
    Calculăm norma ∥A * u_max - λ_max * u_max∥
    A: matricea rară simetrică
    u_max: vectorul propriu asociat celei mai mari valori proprii
    lambda_max: valoarea proprie maximă
    n: dimensiunea matricei
    """
    # Creăm o matrice completă din structura rară pentru produsul A * u_max
    A_full = np.zeros((n, n))
    for (i, j), value in A.items():
        A_full[i, j] = value
        if i != j:
            A_full[j, i] = value  # păstrăm simetria

    # Verificăm dimensiunea matricei complete
    print("Dimensiunea matricei complete:", A_full.shape)

    # Asigurăm că vectorul u_max este un vector coloană
    u_max = np.reshape(u_max, (-1, 1))  # Transformăm u_max într-un vector coloană, dacă nu este deja

    # Calculăm produsul matricei complete cu vectorul
    result = A_full @ u_max

    # Calculăm norma diferenței
    return np.linalg.norm(result - lambda_max * u_max)

# Exemplu de utilizare:
n = 500  # dimensiunea matricei
density = 0.05  # densitatea valorilor nenule

# 1. Generăm matricea rară și simetrică
A = generate_sparse_symmetric_matrix(n, density)

# 2. Calculăm metoda puterii pentru cea mai mare valoare proprie
lambda_max, v_max = power_method(A, n)

# Afișăm rezultatele
print("Cea mai mare valoare proprie:", lambda_max)
print("Vectorul propriu asociat:", v_max)

# Calculăm norma:
norma = compute_norm(A, v_max, lambda_max, n)
print("Norma ∥A * u_max - λ_max * u_max∥:", norma)

# 3. Descompunerea SVD pentru p > n
A_full = np.random.rand(n, n + 10)  # exemplu de matrice p > n
b = np.random.rand(n)  # vector b

U, S, Vt, rank, cond_number, A_inv = compute_svd(A_full)

# Afișăm rezultatele SVD
print("Valorile singulare:", S)
print("Rangul matricei:", rank)
print("Numărul de condiționare:", cond_number)
print("Pseudoinversa A:", A_inv)

# 4. Calculăm soluția sistemului Ax = b folosind pseudoinversa
x_I = solve_least_squares(A_full, b)
print("Solutia x_I:", x_I)

# 5. Calculăm norma erorii
error = compute_error(A_full, b, x_I)
print("Eroarea:", error)
