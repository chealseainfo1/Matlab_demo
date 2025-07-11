import numpy as np
import matplotlib.pyplot as plt
import time


# Define your functions and constants
def f(x, t):
    return 0.5 * x


def P_C(x, c):
    return x - np.dot(c, x) / np.linalg.norm(c) ** 2


def compute_theta_n(x_n, x_n_minus_1, n):
    if np.allclose(x_n, x_n_minus_1):
        return 1
    else:
        return min(1, 1 / (n * (n - 1) * np.linalg.norm(x_n - x_n_minus_1)))


# Parameters
T = 100  # Max number of iterations
t_values = np.linspace(0, 1, 100)
c = np.ones_like(t_values)  # Example constant
lambdas_mus = [(0.01, 0.01), (0.19, 0.19), (0.19, 0.01), (0.01, 0.19)]


# Function to run the original algorithm
def run_original_algorithm(lam, mu):
    x_n = 3 * np.exp(2 * t_values)
    x_n_minus_1 = x_n
    errors = []

    for n in range(1, T):
        tilde_theta_n = compute_theta_n(x_n, x_n_minus_1, n)
        theta_n = np.random.uniform(0, tilde_theta_n)

        y_n = x_n + theta_n * (x_n - x_n_minus_1)
        z_n = (1 - 1 / (2 ** n)) * P_C(1 - mu * f(y_n, t_values), c)
        x_n_plus_1 = ((1 - 1 / (n + 3)) * z_n * f(x_n, t_values) +
                      (0.5 + 1 / (n + 1)) * ((P_C((1 - lam) * f((1 - 1 / (n + 3)) * z_n, t_values), c) -
                                              (1 - 1 / (n + 3)) * z_n)) +
                      1 / (3 ** n))

        error = np.linalg.norm(x_n_plus_1 - x_n) ** 2
        errors.append(error)

        x_n_minus_1 = x_n
        x_n = x_n_plus_1

        if error < 1e-3:
            break

    return errors, n


# Function to run the new algorithm
def run_new_algorithm(lam, mu):
    x_n = 3 * np.exp(2 * t_values)
    errors = []

    for n in range(1, T):
        z_n = P_C((1 - mu) * f(x_n, t_values), c)
        y_n = P_C((1 - lam) * z_n, c)
        x_n_plus_1 = P_C((1 / (2 * (n + 1))) * x_n + (1 - 1 / (n + 1)) * (0.5 ** n) * y_n, c)

        error = np.linalg.norm(x_n_plus_1 - x_n) ** 2
        errors.append(error)

        x_n = x_n_plus_1

        if error < 1e-3:
            break

    return errors, n


# Run both algorithms and plot results for each (lambda, mu)
for i, (lam, mu) in enumerate(lambdas_mus):
    start_time = time.time()
    original_errors, original_iter = run_original_algorithm(lam, mu)
    original_time = time.time() - start_time

    start_time = time.time()
    new_errors, new_iter = run_new_algorithm(lam, mu)
    new_time = time.time() - start_time

    # Plotting
    plt.figure(i)
    plt.plot(original_errors, label=f'Algorithm (15)')
    plt.plot(new_errors, label=f'Algorithm (45) of G. Cai et al.', linestyle='--')
    plt.xlabel('Number of Iteration')
    plt.ylabel('$\|x_{n-1}-x_n\|_2$')
    plt.yscale('log')
    plt.title(f'')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"λ={lam}, μ={mu} - Original: iterations = {original_iter}, CPU time = {original_time:.4f} s")
    print(f"λ={lam}, μ={mu} - New: iterations = {new_iter}, CPU time = {new_time:.4f} s\n")