import numpy as np

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
