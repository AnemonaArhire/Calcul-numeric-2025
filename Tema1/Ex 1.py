import numpy as np

def find_machine_precision():
    u = 1.0  # Start with 10^0
    while (1.0 + u) != 1.0:  # Check when addition is not equal to 1
        u /= 10  # Decrease u by a factor of 10
    return u * 10  # Return the last working value before it failed

# Compute and print machine precision
machine_precision = find_machine_precision()
print(f"Machine precision (10^m form): {machine_precision}")
print(f"m = {-int(np.log10(machine_precision))}")
