import numpy as np


def citire_date(nume_fisier="matrice.txt"):
    with open(nume_fisier, 'r') as f:
        # Citim dimensiunea matricei (n)
        n = int(f.readline().strip())

        # Citim matricea A
        A = np.zeros((n, n))
        for i in range(n):
            linie = f.readline().strip()
            A[i] = list(map(float, linie.split()))

    # Citim precizia (ε) și numărul maxim de iterații (kmax) de la utilizator
    epsilon = float(input("Introduceți precizia (ε): "))
    kmax = int(input("Introduceți numărul maxim de iterații (kmax): "))

    return n, epsilon, kmax, A


def alegere_V0(A):
    # Calculăm norma 1 și norma infinit a matricei A
    norm1_A = np.max(np.sum(np.abs(A), axis=0))  # Norma 1: maximul sumei pe coloane
    norm_inf_A = np.max(np.sum(np.abs(A), axis=1))  # Norma infinit: maximul sumei pe linii

    # Calculăm V0 folosind formula (5)
    V0 = A.T / (norm1_A * norm_inf_A)
    return V0


def metoda_Schultz(A, V0, epsilon, kmax):
    Vk = V0
    k = 0
    while k < kmax:
        Vk1 = Vk @ (2 * np.eye(A.shape[0]) - A @ Vk)  # Paranteza închisă aici
        delta_V = np.linalg.norm(Vk1 - Vk, ord=np.inf)

        if delta_V < epsilon:
            return Vk1, k + 1

        Vk = Vk1
        k += 1

    return Vk, k


def metoda_Li_Li_2(A, V0, epsilon, kmax):
    Vk = V0
    k = 0
    while k < kmax:
        AVk = A @ Vk
        Vk1 = Vk @ (3 * np.eye(A.shape[0]) - AVk @ (3 * np.eye(A.shape[0]) - AVk))  # Paranteze corecte
        delta_V = np.linalg.norm(Vk1 - Vk, ord=np.inf)

        if delta_V < epsilon:
            return Vk1, k + 1

        Vk = Vk1
        k += 1

    return Vk, k


def metoda_Li_Li_3(A, V0, epsilon, kmax):
    Vk = V0
    k = 0
    while k < kmax:
        I = np.eye(A.shape[0])
        AVk = A @ Vk
        term = I - AVk
        Vk1 = (1 / 4) * Vk @ (I + term @ (I + term @ term))
        delta_V = np.linalg.norm(Vk1 - Vk, ord=np.inf)

        if delta_V < epsilon:
            return Vk1, k + 1

        Vk = Vk1
        k += 1

    return Vk, k


def calcul_norme(A, Vk, A_inv_exact):
    # Norma ||A * Vk - I||
    norma_A_Vk_I = np.linalg.norm(A @ Vk - np.eye(A.shape[0]), ord=np.inf)

    # Norma ||Vk - A_inv_exact||
    norma_Vk_A_inv = np.linalg.norm(Vk - A_inv_exact, ord=np.inf)

    return norma_A_Vk_I, norma_Vk_A_inv


def forma_generala_inversa(n):
    # Deducem forma generală a inversei pentru matricea A dată
    A_inv = np.zeros((n, n))
    for i in range(n):
        A_inv[i, i] = 1  # Diagonala principală
        if i < n - 1:
            A_inv[i, i + 1] = -2  # Elementele de deasupra diagonalei principale
    return A_inv


def main():
    # Citim datele din fișier
    n, epsilon, kmax, A = citire_date("matrice.txt")

    # Alegem V0
    V0 = alegere_V0(A)

    # Aplicăm metodele iterative
    print("\nMetoda Schultz:")
    Vk_Schultz, iter_Schultz = metoda_Schultz(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Schultz}")

    print("\nMetoda Li și Li (varianta 2):")
    Vk_Li_Li_2, iter_Li_Li_2 = metoda_Li_Li_2(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Li_Li_2}")

    print("\nMetoda Li și Li (varianta 3):")
    Vk_Li_Li_3, iter_Li_Li_3 = metoda_Li_Li_3(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Li_Li_3}")

    # Calculăm inversa exactă (pentru verificare)
    A_inv_exact = forma_generala_inversa(n)

    # Calculăm normele pentru fiecare metodă
    norma_A_Vk_I_Schultz, norma_Vk_A_inv_Schultz = calcul_norme(A, Vk_Schultz, A_inv_exact)
    norma_A_Vk_I_Li_Li_2, norma_Vk_A_inv_Li_Li_2 = calcul_norme(A, Vk_Li_Li_2, A_inv_exact)
    norma_A_Vk_I_Li_Li_3, norma_Vk_A_inv_Li_Li_3 = calcul_norme(A, Vk_Li_Li_3, A_inv_exact)

    print("\nNorme pentru metoda Schultz:")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Schultz}")
    print(f"||Vk - A_inv_exact||: {norma_Vk_A_inv_Schultz}")

    print("\nNorme pentru metoda Li și Li (varianta 2):")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Li_Li_2}")
    print(f"||Vk - A_inv_exact||: {norma_Vk_A_inv_Li_Li_2}")

    print("\nNorme pentru metoda Li și Li (varianta 3):")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Li_Li_3}")
    print(f"||Vk - A_inv_exact||: {norma_Vk_A_inv_Li_Li_3}")


if __name__ == "__main__":
    main()
