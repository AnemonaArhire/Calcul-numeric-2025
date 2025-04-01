import numpy as np


def generare_matrice_A(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 1)
    for i in range(n - 1):
        A[i, i + 1] = 2
    return A


def alegere_V0(A):
    norm1_A = np.max(np.sum(np.abs(A), axis=0))  # norma A1 (max sum pe coloane)
    norm_inf_A = np.max(np.sum(np.abs(A), axis=1))  # norma A∞ (max suma pe linii)
    V0 = A.T / (norm1_A * norm_inf_A)
    return V0


def metoda_Schultz(A, V0, epsilon, kmax):
    Vk = V0
    for k in range(kmax):
        Vk1 = Vk @ (2 * np.eye(A.shape[0]) - A @ Vk)

        # Condiția 1: ||V_k - V_{k-1}|| < epsilon
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) < epsilon:
            return Vk1, k + 1

        # Condiția 3: ||V_k - V_{k-1}|| > 10^10 (divergență)
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) > 1e10:
            print(f"Divergență detectată la iterația {k + 1}!")
            return None, k + 1

        Vk = Vk1

    return Vk, kmax  # Dacă nu s-a atins precizia, returnează ultima aproximare


def metoda_Li_Li_2(A, V0, epsilon, kmax):
    Vk = V0
    for k in range(kmax):
        AVk = A @ Vk
        Vk1 = Vk @ (3 * np.eye(A.shape[0]) - AVk @ (3 * np.eye(A.shape[0]) - AVk))

        # Condiția 1: ||V_k - V_{k-1}|| < epsilon
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) < epsilon:
            return Vk1, k + 1

        # Condiția 3: ||V_k - V_{k-1}|| > 10^10 (divergență)
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) > 1e10:
            print(f"Divergență detectată la iterația {k + 1}!")
            return None, k + 1

        Vk = Vk1

    return Vk, kmax


def metoda_Li_Li_3(A, V0, epsilon, kmax):
    Vk = V0
    I = np.eye(A.shape[0])  # Matricea identitate

    for k in range(kmax):
        AVk = Vk @ A  # V_k * A
        term = I - AVk  # (I_n - V_k A)
        Vk1 = (I + (1 / 4) * term @ (3 * I - AVk) @ (3 * I - AVk)) @ Vk  # Formula corectată

        # Condiția 1: ||V_k - V_{k-1}|| < epsilon
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) < epsilon:
            return Vk1, k + 1

        # Condiția 3: ||V_k - V_{k-1}|| > 10^10 (divergență)
        if np.linalg.norm(Vk1 - Vk, ord=np.inf) > 1e10:
            print(f"Divergență detectată la iterația {k + 1}!")
            return None, k + 1

        Vk = Vk1

    return Vk, kmax  # Returnăm ultima aproximare dacă nu s-a atins precizia


# calcul norme eroare
def calcul_norme(A, Vk, A_inv_exact):
    norma_A_Vk_I = np.linalg.norm(A @ Vk - np.eye(A.shape[0]),
                                  ord=np.inf)  # ||A * Vk - I|| - Cat de apropiata este Vk de inversa reala
    norma_A_inv_exact_A_inv_aprox = np.linalg.norm(Vk - A_inv_exact,
                                                   ord=np.inf)  # ||A^-1_exact - A^-1_aprox|| - Diferenta dintre aproximare si inversa exacta
    return norma_A_Vk_I, norma_A_inv_exact_A_inv_aprox


# calcul invers exact
def forma_generala_inversa(n):
    A_inv = np.zeros((n, n))
    for i in range(n):
        A_inv[i, i] = 1
        if i < n - 1:
            A_inv[i, i + 1] = -2
    return A_inv


def main():
    n = int(input("Introduceți dimensiunea matricei n: "))
    epsilon = float(input("Introduceți precizia (ε): "))
    kmax = int(input("Introduceți numărul maxim de iterații (kmax): "))

    A = generare_matrice_A(n)
    V0 = alegere_V0(A)

    print("\nMetoda Schultz:")
    Vk_Schultz, iter_Schultz = metoda_Schultz(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Schultz}")

    print("\nMetoda Li și Li (varianta 2):")
    Vk_Li_Li_2, iter_Li_Li_2 = metoda_Li_Li_2(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Li_Li_2}")

    print("\nMetoda Li și Li (varianta 3):")
    Vk_Li_Li_3, iter_Li_Li_3 = metoda_Li_Li_3(A, V0, epsilon, kmax)
    print(f"Număr de iterații: {iter_Li_Li_3}")

    A_inv_exact = forma_generala_inversa(n)
    norma_A_Vk_I_Schultz, norma_A_inv_exact_A_inv_aprox_Schultz = calcul_norme(A, Vk_Schultz, A_inv_exact)
    norma_A_Vk_I_Li_Li_2, norma_A_inv_exact_A_inv_aprox_Li_Li_2 = calcul_norme(A, Vk_Li_Li_2, A_inv_exact)
    norma_A_Vk_I_Li_Li_3, norma_A_inv_exact_A_inv_aprox_Li_Li_3 = calcul_norme(A, Vk_Li_Li_3, A_inv_exact)

    print("\nNorme pentru metoda Schultz:")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Schultz}")
    print(f"||A^-1_exact - A^-1_aprox||: {norma_A_inv_exact_A_inv_aprox_Schultz}")

    print("\nNorme pentru metoda Li și Li (varianta 2):")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Li_Li_2}")
    print(f"||A^-1_exact - A^-1_aprox||: {norma_A_inv_exact_A_inv_aprox_Li_Li_2}")

    print("\nNorme pentru metoda Li și Li (varianta 3):")
    print(f"||A * Vk - I||: {norma_A_Vk_I_Li_Li_3}")
    print(f"||A^-1_exact - A^-1_aprox||: {norma_A_inv_exact_A_inv_aprox_Li_Li_3}")


if __name__ == "__main__":
    main()
