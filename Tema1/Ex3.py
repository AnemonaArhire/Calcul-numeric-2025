import numpy as np
import matplotlib.pyplot as plt

def find_machine_precision():
    u = 1.0  # Start with 10^0
    while (1.0 + u) != 1.0:  # Check when addition is not equal to 1
        u /= 10  # Decrease u by a factor of 10
    return u * 10  # Return the last working value before it failed

def check_addition_associativity():
    u = find_machine_precision()
    x = 1.0
    y = u / 10
    z = u / 10
    left = (x + y) + z
    right = x + (y + z)
    return left, right, left != right

def check_multiplication_associativity():
    a = 1.0e30
    b = 1.0e-30
    c = 1.0e30
    left = (a * b) * c
    right = a * (b * c)
    return left, right, left != right

def polynomial_sin_approx(x, order):
    coefficients = {
        3: -0.16666666666666666,
        5: 0.008333333333333333,
        7: -0.0001984126984126984,
        9: 2.7557319223985893e-06,
        11: -2.505210838544172e-08,
        13: 1.6059043836821613e-10
    }
    result = x
    for power, coef in coefficients.items():
        if power <= order:
            result += coef * x**power
    return result

x_values = np.linspace(-np.pi, np.pi, 400)
sin_values = np.sin(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, sin_values, label="sin(x)", linewidth=2)

for order in [3, 5, 7, 9, 11, 13]:
    approx_values = polynomial_sin_approx(x_values, order)
    plt.plot(x_values, approx_values, '--', label=f"P{order}(x)")

plt.xlabel("x")
plt.ylabel("Function values")
plt.legend()
plt.title("Polynomial Approximations of sin(x)")
plt.grid()
plt.show()

# Compute and print machine precision
machine_precision = find_machine_precision()
print(f"Machine precision (10^m form): {machine_precision}")
print(f"m = {-int(np.log10(machine_precision))}")

# Check and print addition associativity
add_left, add_right, add_neassoc = check_addition_associativity()
print(f"Addition associativity check: ({add_left}) != ({add_right}) -> {add_neassoc}")

# Check and print multiplication associativity
mul_left, mul_right, mul_neassoc = check_multiplication_associativity()
print(f"Multiplication associativity check: ({mul_left}) != ({mul_right}) -> {mul_neassoc}")
