import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import norm
from scipy.linalg import solve


# 🔹 1. Citirea matricei rare
def citire_matrice_rara(nume_fisier):
    try:
        with open(nume_fisier, 'r') as f:
            n = int(f.readline().strip())
            A = lil_matrix((n, n))
            for line in f:
                valori = line.strip().replace(',', ' ').split()
                if len(valori) != 3:
                    print(f"⚠️ Linie invalidă (ignorat): {line}")
                    continue
                i, j, val = int(float(valori[0])), int(float(valori[1])), float(valori[2])
                A[i - 1, j - 1] = val
        return A.tocsr()
    except FileNotFoundError:
        print(f"❌ Eroare: Fișierul {nume_fisier} nu a fost găsit!")
        exit(1)
    except ValueError as e:
        print(f"❌ Eroare la citirea fișierului {nume_fisier}: {e}")
        exit(1)


# 🔹 2. Citirea vectorului b
def citire_vector_b(nume_fisier):
    try:
        with open(nume_fisier, 'r') as f:
            n = int(f.readline().strip())
            b = np.zeros(n)
            for i, line in enumerate(f):
                if line.strip():
                    b[i] = float(line.strip())
        return b
    except FileNotFoundError:
        print(f"❌ Eroare: Fișierul {nume_fisier} nu a fost găsit!")
        exit(1)
    except ValueError as e:
        print(f"❌ Eroare la citirea fișierului {nume_fisier}: {e}")
        exit(1)


# 🔹 3. Eliminarea rândurilor și coloanelor cu zerouri pe diagonală
def verifica_si_elimina_nule(A, b):
    mask = A.diagonal() != 0
    A_filtrata = A[mask][:, mask]  # Eliminăm atât rândurile, cât și coloanele
    b_filtrat = b[mask]
    eliminari = len(b) - len(b_filtrat)
    print(f"\n✅ Eliminat {eliminari} rânduri și coloane cu zerouri pe diagonală.")
    return A_filtrata, b_filtrat


# 🔹 4. Completarea diagonalei pentru evitarea împărțirii la zero
def completeaza_diagonala(A, epsilon=1e-10):
    for i in range(A.shape[0]):
        if A[i, i] == 0:
            A[i, i] = epsilon
            print(f"🔄 Adăugat {epsilon} la A[{i}, {i}] pentru a evita împărțirea la zero.")
    return A


# 🔹 5. Metoda Gauss-Seidel cu prevenirea overflow-ului
# 🔹 5. Metoda Gauss-Seidel cu prevenirea overflow-ului și stabilizare numerică
def gauss_seidel(A, b, tol=1e-10, max_iter=10000):
    n = A.shape[0]
    x = np.zeros(n)
    max_value = 1e10  # Prag maxim pentru stabilizare numerică

    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    try:
                        product = A[i, j] * x_new[j]
                        if np.abs(product) > max_value:  # Limităm produsul pentru a preveni overflow
                            product = np.sign(product) * max_value
                        suma += product
                    except OverflowError:
                        raise ValueError(f"❌ Overflow detectat la iteratia {k}, element A[{i}, {j}]!")

            if np.isinf(suma) or np.isnan(suma):
                raise ValueError(f"❌ Eroare numerică la iteratia {k}: overflow sau NaN detectat!")

            x_new[i] = (b[i] - suma) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new

    return x, max_iter


# 🔹 6. Eliminare Gaussiana
def eliminare_gaussiana(A, b):
    try:
        return solve(A.toarray(), b)
    except np.linalg.LinAlgError:
        print("❌ Matricea este singulară sau sistemul nu are soluție unică!")
        return None


# 🔹 7. Afișarea matricei într-un format lizibil
def afiseaza_matrice_formatata(A):
    np.set_printoptions(precision=7, suppress=True, linewidth=100, threshold=10_000)
    print("\nMatricea A după completarea diagonalei:")
    print(A.toarray())


# 🔹 8. MAIN - Program principal
if __name__ == "__main__":
    nume_fisier_A = "a_1.txt"
    nume_fisier_B = "b_1.txt"

    A = citire_matrice_rara(nume_fisier_A)
    b = citire_vector_b(nume_fisier_B)
    A, b = verifica_si_elimina_nule(A, b)
    A = completeaza_diagonala(A)
    afiseaza_matrice_formatata(A)

    try:
        x_GS, iteratii = gauss_seidel(A, b)
        print(f"\n✅ Soluția aproximativă x_GS: {x_GS}")
        print(f"🔄 Număr de iterații: {iteratii}")
    except ValueError as e:
        print(f"\n❌ Eroare Gauss-Seidel: {e}")
        print("➡️ Încercăm eliminarea Gaussiana...")
        x_LU = eliminare_gaussiana(A, b)
        if x_LU is not None:
            print(f"\n✅ Soluția folosind eliminarea Gaussiana: {x_LU}")
