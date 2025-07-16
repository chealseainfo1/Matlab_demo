import numpy as np
import matplotlib.pyplot as plt
import time

# Problem Definition
def f(vec):
    x, y = vec
    return np.array([x + y + np.sin(x), -x + y + np.sin(y)])

# Projection onto box constraint
def projection_C(vec, bounds):
    return np.clip(vec, [b[0] for b in bounds], [b[1] for b in bounds])

# Norm difference
def norm_diff(x1, x2):
    return np.linalg.norm(x1 - x2)

# Parameters
C_bounds = [(-10, 100), (10, 100)]
x0 = np.random.uniform([-10, 10], [10, 20])
x1 = np.random.uniform([-10, 10], [10, 20])
tol = 1e-4
max_iters = 1000

# Parameter sets for experimentation
mu_values = [0.01, 0.1, 0.5, 0.9]
lambda_values = [0.5, 1.0, 2.0]

# Run experiments
for mu in mu_values:
    for lambda_ in lambda_values:
        errors_3_3 = []
        errors_eq_3_1 = []

        # Algorithm 3.3
        x_prev = x0.copy()
        x_curr = x1.copy()
        d_prev = (x_curr - x_prev) / lambda_

        # Start timing
        start_time_3_3 = time.time()
        iterations_3_3 = 0

        for _ in range(max_iters):
            iterations_3_3 += 1
            w = x_curr
            u = projection_C(w - mu * f(w), C_bounds)
            t = projection_C(w - mu * f(u), C_bounds)
            z = t
            d = (z - w) / lambda_ + 0.1 * d_prev
            y = w + lambda_ * d
            x_next = 0.9 * w + 0.1 * y

            err = norm_diff(x_next, x_curr)
            errors_3_3.append(err)
            if err / norm_diff(x1, x0) < tol:
                break

            x_prev = x_curr
            x_curr = x_next
            d_prev = d

        runtime_3_3 = time.time() - start_time_3_3

        # Algorithm in Equation (3.1)
        x_curr = x1.copy()
        iterations_eq_3_1 = 0
        start_time_eq_3_1 = time.time()

        alpha_n = lambda n: 1 / (4 * n)
        beta_n = lambda n: (2 * n + 1) / (4 * n)
        gamma_n = lambda n: (2 * n - 3) / (4 * n)
        lambda_n = lambda_

        for n in range(1, max_iters):
            iterations_eq_3_1 += 1
            y = projection_C(x_curr - lambda_n * f(x_curr), C_bounds)
            t = projection_C(x_curr - lambda_n * f(y), C_bounds)
            T_t = t
            x_next = alpha_n(n) * x0 + beta_n(n) * x_curr + gamma_n(n) * T_t

            err = norm_diff(x_next, x_curr)
            errors_eq_3_1.append(err)
            if err / norm_diff(x1, x0) < tol:
                break

            x_curr = x_next

        runtime_eq_3_1 = time.time() - start_time_eq_3_1

        # Print iterations and runtime
        print(f"mu={mu}, lambda={lambda_}")
        print(f"Algorithm 3.3: Iterations={iterations_3_3}, Runtime={runtime_3_3:.4f} seconds")
        print(f"Algorithm Eq (3.1): Iterations={iterations_eq_3_1}, Runtime={runtime_eq_3_1:.4f} seconds")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(errors_3_3, label='Algorithm 3.3', marker='o')
        plt.plot(errors_eq_3_1, label='Algorithm in Eq (3.1)', marker='x')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Error Norm (log scale)')
        plt.title(f'Comparison with mu={mu}, lambda={lambda_}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()