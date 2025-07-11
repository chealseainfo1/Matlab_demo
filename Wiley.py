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


T = 100  # Max number of iterations
t_values = np.linspace(0, 1, 100)
c = np.ones_like(t_values)  # Example constant

# Initialization
x_n = 3 * np.exp(2 * t_values)
x_n_minus_1 = x_n
errors = []

lambdas_mus = [(0.01, 0.01), (0.19, 0.19), (0.19, 0.01), (0.01, 0.19)]
results = []

for lam, mu in lambdas_mus:
    start_time = time.time()
    for n in range(1, T):
        tilde_theta_n = compute_theta_n(x_n, x_n_minus_1, n)
        theta_n = np.random.uniform(0, tilde_theta_n)  # Choose theta_n from [0, tilde_theta_n]

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

    cpu_time = time.time() - start_time
    results.append((n, cpu_time))

    # Plotting the Cauchy errors
    plt.plot(errors, label=f'λ={lam}, μ={mu}')
    errors.clear()  # Clear errors for next iteration

plt.xlabel('Number of Iteration')
plt.ylabel('$\|x_{n-1}-x_n\|_2$')
plt.yscale('log')
plt.title('Cauchy Error vs Iterations')
plt.legend()
plt.show()

# Output the results
for i, (lam, mu) in enumerate(lambdas_mus):
    print(f"λ={lam}, μ={mu}: iterations = {results[i][0]}, CPU time = {results[i][1]:.4f} s")