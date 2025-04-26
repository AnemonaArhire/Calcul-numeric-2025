import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import solve

# ----------------- DEFINIRE FUNCTII ----------------- #
# Exemplu f(x) (poți schimba aici)
def f(x):
    return x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12
    # return np.sin(x) - np.cos(x)
    # return np.sin(2*x) + np.sin(x) + np.cos(3*x)
    # return np.sin(x)**2 - np.cos(x)**2

# Generare noduri x si valori y pentru polinoame
def genereaza_noduri_si_valori(x0, xn, n):
    xs = np.linspace(x0, xn, n + 1)
    return np.array(xs), np.array([f(x) for x in xs])

# Generare noduri pentru interpolare trigonometrică (in [0, 2pi))
def genereaza_noduri_trigonometrice(n):
    assert n % 2 == 0, "Pentru interpolare trigonometrică, n trebuie par (n=2m)!"
    xs = np.linspace(0, 2 * np.pi, n + 1, endpoint=False)
    return xs, np.array([f(x) for x in xs])

# Metoda celor mai mici pătrate: gaseste coeficientii polinomului
def cei_mai_mici_patrate(xs, ys, m):
    B = np.zeros((m + 1, m + 1))
    f_vec = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            B[i, j] = np.sum(xs ** (i + j))
        f_vec[i] = np.sum(ys * xs ** i)
    coeficienti = solve(B, f_vec)
    return coeficienti

# Schema Horner pentru evaluarea polinomului
def horner(coeficienti, x):
    rezultat = coeficienti[-1]
    for c in reversed(coeficienti[:-1]):
        rezultat = rezultat * x + c
    return rezultat

# Interpolare trigonometrică
def interpolare_trigonometrica(xs, ys):
    n = len(xs) - 1  # n = 2m
    m = n // 2
    T = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        T[i, 0] = 1
        for k in range(1, m + 1):
            T[i, 2 * k - 1] = np.sin(k * xs[i])
            T[i, 2 * k] = np.cos(k * xs[i])
    coeficienti = solve(T, ys)
    return coeficienti

# Evaluare interpolare trigonometrică în punct
def eval_trigonometrica(coeficienti, x):
    n = len(coeficienti) - 1
    m = n // 2
    rezultat = coeficienti[0]
    for k in range(1, m + 1):
        rezultat += coeficienti[2 * k - 1] * np.sin(k * x) + coeficienti[2 * k] * np.cos(k * x)
    return rezultat

# ----------------- PARTEA PRINCIPALA ----------------- #
def main():
    random.seed(42)
    np.random.seed(42)

    # Date inițiale
    x0 = float(input("Introduceti x0: "))  # ex: 1
    xn = float(input("Introduceti xn: "))  # ex: 5
    n = int(input("Introduceti numarul n (n+1 puncte): "))  # ex: 6 -> 7 puncte

    xs, ys = genereaza_noduri_si_valori(x0, xn, n)
    print("\nNoduri xs:", xs)
    print("Valori ys:", ys)

    xbar = float(input("\nIntroduceti xbar (punctul de aproximat): "))  # ex: 2.5

    # Aproximare polinomială pentru m = 1...5
    for m in range(1, 6):
        print(f"\n--- Aproximare polinomiala cu m = {m} ---")
        coef_poli = cei_mai_mici_patrate(xs, ys, m)
        Pm_xbar = horner(coef_poli, xbar)
        eroare_xbar = abs(Pm_xbar - f(xbar))
        eroare_noduri = np.sum(np.abs([horner(coef_poli, xi) - yi for xi, yi in zip(xs, ys)]))

        print(f"Pm({xbar}) = {Pm_xbar}")
        print(f"|Pm({xbar}) - f({xbar})| = {eroare_xbar}")
        print(f"Suma |Pm(xi) - yi| = {eroare_noduri}")

        # Grafic polinom
        x_grafic = np.linspace(x0, xn, 500)
        y_grafic = [horner(coef_poli, xx) for xx in x_grafic]
        plt.plot(x_grafic, y_grafic, label=f'Pm grad {m}')

    # Interpolare trigonometrică
    if n % 2 != 0:
        print("\n!!! Pentru interpolare trigonometrica, n trebuie par. Marim n cu 1.")
        n += 1

    xs_trig, ys_trig = genereaza_noduri_trigonometrice(n)
    coef_trig = interpolare_trigonometrica(xs_trig, ys_trig)
    Tn_xbar = eval_trigonometrica(coef_trig, xbar)
    eroare_trig = abs(Tn_xbar - f(xbar))

    print("\n--- Interpolare Trigonometrica ---")
    print(f"Tn({xbar}) = {Tn_xbar}")
    print(f"|Tn({xbar}) - f({xbar})| = {eroare_trig}")

    # Grafic interpolare trigonometrică
    x_grafic_trig = np.linspace(0, 2 * np.pi, 500)
    y_grafic_trig = [eval_trigonometrica(coef_trig, xx) for xx in x_grafic_trig]
    plt.plot(x_grafic_trig, y_grafic_trig, label='Interpolare Trigonometrică', linestyle='--')

    # Grafic funcția originală
    x_fct = np.linspace(min(x0, 0), max(xn, 2 * np.pi), 500)
    y_fct = [f(xx) for xx in x_fct]
    plt.plot(x_fct, y_fct, label='f(x)', color='black', linewidth=2)

    plt.scatter(xs, ys, color='red', zorder=5, label='Noduri polinoame')
    plt.scatter(xs_trig, ys_trig, color='green', zorder=5, label='Noduri trigonometrice')
    plt.title('Aproximare functii')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
