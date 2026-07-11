"""
log.py: A standalone, high-precision mathematical extension module.
Implements the natural logarithm (ln) and common bases from scratch.
"""

# Highly accurate mathematical constant e
E = 2.7182818284590452354


def ln(x):
    """
    Computes the natural logarithm (base-e) of x using range reduction 
    and accelerated Taylor series expansion (Inverse Hyperbolic Tangent).

    Args:
        x (float/int): The positive numeric input.

    Returns:
        float: The natural logarithm of x.
    """
    # 1. Handle Edge Cases
    if x <= 0:
        raise ValueError(
            "Math Domain Error: Logarithm undefined for values <= 0.")

    # 2. Fast Range Reduction
    # Brings x into the high-convergence range [0.5, 1.5]
    exponent_shift = 0

    while x > 1.5:
        x /= E
        exponent_shift += 1

    while x < 0.5:
        x *= E
        exponent_shift -= 1

    # 3. Taylor Series Expansion (Artanh Method)
    # ln(x) = 2 * [ ((x-1)/(x+1)) + (1/3)*((x-1)/(x+1))^3 + ... ]
    y = (x - 1) / (x + 1)
    y_squared = y * y

    total_sum = 0.0
    current_term = y
    denominator = 1

    iterations = 0
    # High-precision convergence threshold
    while abs(current_term / denominator) > 1e-16 and iterations < 1000:
        total_sum += current_term / denominator
        current_term *= y_squared
        denominator += 2
        iterations += 1

    # 4. Reconstruct and return the exact value
    return exponent_shift + (2.0 * total_sum)


def log10(x):
    """Computes the common logarithm (base-10) of x using change of base formula."""
    return ln(x) / ln(10.0)


def log2(x):
    """Computes the binary logarithm (base-2) of x using change of base formula."""
    return ln(x) / ln(2.0)


def log_base(x, base):
    """Computes the logarithm of x to an arbitrary custom base."""
    if base <= 0 or base == 1:
        raise ValueError(
            "Math Domain Error: Logarithmic base must be > 0 and not equal to 1.")
    return ln(x) / ln(base)


# --- Self-Testing Block ---
if __name__ == "__main__":
    print("--- Running Verification Suite for log.py ---")

    test_cases = [E, 1, 10, 100, 0.5]
    print(f"{'Value':<10} | {'ln(x)':<20} | {'log10(x)':<20} | {'log2(x)':<20}")
    print("-" * 78)
    for value in test_cases:
        print(
            f"{value:<10.4f} | {ln(value):<20.14f} | {log10(value):<20.14f} | {log2(value):<20.14f}")
