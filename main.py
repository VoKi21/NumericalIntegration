import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, integrate, sin


def f_sympy(x):
    return x ** 2 * sin(x)


def f_numpy(x):
    return x ** 2 * np.sin(x)


def left_rectangle_integration(a, b, f, eps):
    n = 3
    integral_old = 0
    while True:
        h = (b - a) / n
        x = np.linspace(a, b - h, n)
        integral_new = sum(f(x)) * h
        if abs(integral_new - integral_old) < eps:
            break
        integral_old = integral_new
        n *= 2
    return integral_new, n


def midpoint_rectangle_integration(a, b, f, eps):
    n = 1
    integral_old = 0
    while True:
        h = (b - a) / n
        x = np.linspace(a + h / 2, b - h / 2, n)
        integral_new = sum(f(x)) * h
        if abs(integral_new - integral_old) < eps:
            break
        integral_old = integral_new
        n *= 2
    return integral_new, n


def trapezoid_integration(a, b, f, eps):
    n = 1
    integral_old = 0
    while True:
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        integral_new = (h / 2) * (f(a) + 2 * sum(f(x[1:-1])) + f(b))
        if abs(integral_new - integral_old) < eps:
            break
        integral_old = integral_new
        n *= 2
    return integral_new, n


def simpsons_integration(a, b, f, eps):
    n = 1
    integral_old = 0
    while True:
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        integral_new = (h / 3) * (f(a) + 4 * sum(f(x[1:-1:2])) + 2 * sum(f(x[2:-2:2])) + f(b))
        if abs(integral_new - integral_old) < eps:
            break
        integral_old = integral_new
        n *= 2
    return integral_new, n


def monte_carlo_integration(a, b, f, n):
    x = np.random.uniform(a, b, n)
    integral = (b - a) * np.mean(f(x))
    return integral


def deviation(result, expected):
    abs_error = abs(result - expected)
    rel_error = abs_error / abs(expected)
    return abs_error, rel_error


if __name__ == "__main__":
    # Task 1: Exact integral analytically
    x = symbols('x')
    exact_integral = float(integrate(f_sympy(x), (x, 0, 5)))
    print(f"Sympy: {exact_integral}")

    # Task 2: Numerical Integration using Different Methods
    methods = {
        "Left Rectangles": left_rectangle_integration,
        "Midpoint Rectangles": midpoint_rectangle_integration,
        "Trapezoids": trapezoid_integration,
        "Simpson's": simpsons_integration
    }

    results = {}
    eps = 1e-6
    for method_name, method in methods.items():
        integral, steps = method(0, 5, f_numpy, eps)
        abs_error, rel_error = deviation(integral, exact_integral)
        results[method_name] = {
            "integral": integral,
            "steps": steps,
            "abs_error": abs_error,
            "rel_error": rel_error
        }
        print(f"{method_name}:")
        print(f"Integral: {integral}")
        print(f"Absolute Error: {abs_error}")
        print(f"Relative Error: {rel_error}")
        print(f"Number of Steps: {steps}\n")

    # Task 3: Monte Carlo Method Comparison
    monte_carlo_integral = monte_carlo_integration(0, 5, f_numpy, 15000)
    monte_carlo_abs_error, monte_carlo_rel_error = deviation(monte_carlo_integral, exact_integral)

    print("Monte Carlo Method:")
    print(f"Integral: {monte_carlo_integral}")
    print(f"Absolute Error: {monte_carlo_abs_error}")
    print(f"Relative Error: {monte_carlo_rel_error}\n")

    # Task 4: Plotting Error vs. Number of Intervals
    eps_range = np.arange(1e-5, 1e-2, 25 * 10 ** (-6))
    N_left = []
    N_midpoint = []
    N_trapezoid = []
    N_simpsons = []

    for eps in eps_range:
        _, n_left = left_rectangle_integration(0, 5, f_numpy, eps)
        _, n_midpoint = midpoint_rectangle_integration(0, 5, f_numpy, eps)
        _, n_trapezoid = trapezoid_integration(0, 5, f_numpy, eps)
        _, n_simpsons = simpsons_integration(0, 5, f_numpy, eps)
        N_left.append(n_left)
        N_midpoint.append(n_midpoint)
        N_trapezoid.append(n_trapezoid)
        N_simpsons.append(n_simpsons)

    plt.figure(figsize=(10, 6))
    plt.plot(eps_range, N_left, label='Left Rectangles')
    plt.plot(eps_range, N_midpoint, label='Midpoint Rectangles')
    plt.plot(eps_range, N_trapezoid, label='Trapezoids')
    plt.xlabel('Error')
    plt.ylabel('Number of Intervals (N)')
    plt.title('N vs. Error for Rectangles and Trapezoids')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(eps_range, N_simpsons, label="Simpson's")
    plt.xlabel('Error')
    plt.ylabel('Number of Intervals (N)')
    plt.title('N vs. Error for Simpson\'s Method')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
