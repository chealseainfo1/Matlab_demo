% Problem Definition
f = @(vec) [vec(1) + vec(2) + sin(vec(1)); -vec(1) + vec(2) + sin(vec(2))];

% Projection onto box constraint
projection_C = @(vec, bounds) min(max(vec, bounds(:, 1)), bounds(:, 2));

% Norm difference
norm_diff = @(x1, x2) norm(x1 - x2);

% Parameters
C_bounds = [-10, 100; 10, 100];
x0 = [-10; 10] + rand(2, 1) .* [20; 10];
x1 = [-10; 10] + rand(2, 1) .* [20; 10];
tol = 1e-4;
max_iters = 1000;

% Parameter sets for experimentation
mu_values = [0.01, 0.1, 0.5];
lambda_values = [0.5, 1.0, 2.0];

% Run experiments
for mu = mu_values
    for lambda_ = lambda_values
        % Store errors
        errors_3_3 = [];
        errors_eq_3_1 = [];

        % Algorithm 3.3
        x_prev = x0;
        x_curr = x1;
        d_prev = (x_curr - x_prev) / lambda_;

        for iter = 1:max_iters
            w = x_curr;
            u = projection_C(w - mu * f(w), C_bounds);
            t = projection_C(w - mu * f(u), C_bounds);
            z = t;
            d = (z - w) / lambda_ + 0.1 * d_prev;
            y = w + lambda_ * d;
            x_next = 0.9 * w + 0.1 * y;

            err = norm_diff(x_next, x_curr);
            errors_3_3(end+1) = err;
            if err / norm_diff(x1, x0) < tol
                break;
            end

            x_prev = x_curr;
            x_curr = x_next;
            d_prev = d;
        end

        % Algorithm in Equation (3.1)
        x_curr = x1;

        for n = 1:max_iters
            alpha_n = 1 / (4 * n);
            beta_n = (2 * n + 1) / (4 * n);
            gamma_n = (2 * n - 3) / (4 * n);
            lambda_n = lambda_;

            y = projection_C(x_curr - lambda_n * f(x_curr), C_bounds);
            t = projection_C(x_curr - lambda_n * f(y), C_bounds);
            T_t = t;
            x_next = alpha_n * x0 + beta_n * x_curr + gamma_n * T_t;

            err = norm_diff(x_next, x_curr);
            errors_eq_3_1(end+1) = err;
            if err / norm_diff(x1, x0) < tol
                break;
            end

            x_curr = x_next;
        end

        % Plot
        figure;
        semilogy(errors_3_3, '-o');
        hold on;
        semilogy(errors_eq_3_1, '-x');
        xlabel('Number of Iterations');
        ylabel('||x_{n+1}-x_n||_2');
        title(['Comparison with mu=', num2str(mu), ', lambda=', num2str(lambda_)]);
        legend('Algorithm 3.3 of Yekini et al.', 'Proposed Algorithm');
        grid on;
        hold off;
    end
end