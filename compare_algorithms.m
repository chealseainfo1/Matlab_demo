function compare_algorithms
    % Constants and parameters
    T = 100; % Max number of iterations
    t_values = linspace(0, 1, 100);
    c = ones(size(t_values));
    lambdas_mus = [0.01, 0.01; 0.19, 0.19; 0.19, 0.01; 0.01, 0.19];
    
    for i = 1:size(lambdas_mus, 1)
        lam = lambdas_mus(i, 1);
        mu = lambdas_mus(i, 2);
        
        % Run both algorithms
        [original_errors, original_iter] = run_original_algorithm(lam, mu, T, t_values, c);
        [new_errors, new_iter] = run_new_algorithm(lam, mu, T, t_values, c);

        % Plotting
        figure;
        semilogy(original_errors, '-o', 'DisplayName', sprintf('Algorithm (15)', lam, mu));
        hold on;
        semilogy(new_errors, '-*', 'DisplayName', sprintf('Algorithm (45) of G. Cai et al.', lam, mu));
        xlabel('Number of Iteration');
        ylabel('$\|x_{n-1}-x_n\|_2$');
        title(sprintf('', lam, mu));
        legend show;
        grid on;
        hold off;

        % Print results
        fprintf('Algorithm (15)', lam, mu, original_iter);
        fprintf('Algorithm (45) of G. Cai et al.', lam, mu, new_iter);
    end
end

function [errors, iter] = run_original_algorithm(lam, mu, T, t_values, c)
    x_n = 3 * exp(2 * t_values);
    x_n_minus_1 = x_n;
    errors = [];

    for n = 1:T
        tilde_theta_n = compute_theta_n(x_n, x_n_minus_1, n);
        theta_n = rand * tilde_theta_n;

        y_n = x_n + theta_n * (x_n - x_n_minus_1);
        z_n = (1 - 1 / (2 ^ n)) * P_C(1 - mu * f(y_n), c);
        x_n_plus_1 = ((1 - 1 / (n + 3)) * z_n .* f(x_n) + ...
                      (0.5 + 1 / (n + 1)) * (P_C((1 - lam) * f((1 - 1 / (n + 3)) * z_n), c) - ...
                                               (1 - 1 / (n + 3)) * z_n) + ...
                      1 / (3 ^ n));

        error = norm(x_n_plus_1 - x_n) ^ 2;
        errors = [errors, error];

        x_n_minus_1 = x_n;
        x_n = x_n_plus_1;

        if error < 1e-3
            break;
        end
    end

    iter = n;
end

function [errors, iter] = run_new_algorithm(lam, mu, T, t_values, c)
    x_n = 3 * exp(2 * t_values);
    errors = [];

    for n = 1:T
        z_n = P_C((1 - mu) * f(x_n), c);
        y_n = P_C((1 - lam) * z_n, c);
        x_n_plus_1 = P_C((1 / (2 * (n + 1))) * x_n + (1 - 1 / (n + 1)) * (0.5 ^ n) * y_n, c);

        error = norm(x_n_plus_1 - x_n) ^ 2;
        errors = [errors, error];

        x_n = x_n_plus_1;

        if error < 1e-3
            break;
        end
    end

    iter = n;
end

function x_proj = P_C(x, c)
    x_proj = x - (dot(c, x) / norm(c)^2) * c;
end

function y = f(x)
    y = 0.5 * x;
end

function theta = compute_theta_n(x_n, x_n_minus_1, n)
    if norm(x_n - x_n_minus_1) < 1e-10
        theta = 1;
    else
        theta = min(1, 1 / (n * (n - 1) * norm(x_n - x_n_minus_1)));
    end
end