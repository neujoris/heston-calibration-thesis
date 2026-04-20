# ===============================
# LOAD PACKAGES
# ===============================

using Random
using Statistics
using LinearAlgebra
using ForwardDiff

# ===============================
# SETTINGS
# ===============================

seed = 2026

S0 = 100.0
r  = 0.02
T  = 1.0

nr_steps = 252
nr_paths = 10000

strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

# ===============================
# PARAMETERS 
# theta = [kappa, theta, sigma, rho, v0]
# ===============================

theta_true = [2.0, 0.04, 0.30, -0.70, 0.04] 
theta_null = [1.5, 0.05, 0.25, -0.40, 0.05] 

println("True parameters: ", theta_true)
println("Initial guess:   ", theta_null)

# ===============================
# PARAMETERS CHECK
# ===============================

function check_heston_params(theta)
    kappa, theta_bar, sigma, rho, v0 = theta

    if kappa <= 0 error("kappa must be positive") end
    if theta_bar <= 0 error("theta must be positive") end
    if sigma <= 0 error("sigma must be positive") end
    if v0 <= 0 error("v0 must be positive") end
    if rho <= -1 || rho >= 1 error("rho must be in (-1,1)") end

    feller_lhs = 2 * kappa * theta_bar
    feller_rhs = sigma^2

    if feller_lhs <= feller_rhs
        println("Feller condition not satisfied.")
    end
end

check_heston_params(theta_true)
check_heston_params(theta_null)

# ===============================
# FIXED SHOCKS
# ===============================

Random.seed!(seed)
Z1_mat = randn(nr_steps, nr_paths) 
Z2_mat = randn(nr_steps, nr_paths)

# ===============================
# FULL PATH SIMULATION (FOR DATA)
# ===============================

function simulate_heston_paths(S0, r, T, theta, Z1_mat, Z2_mat)
    kappa, theta_bar, sigma, rho, v0 = theta

    nsteps, npaths = size(Z1_mat)
    dt = T / nsteps

    S = Matrix{Float64}(undef, nsteps + 1, npaths)
    v = Matrix{Float64}(undef, nsteps + 1, npaths)

    S[1, :] .= S0
    v[1, :] .= v0

    for i in 1:nsteps
        Z1 = Z1_mat[i, :]
        Z2 = Z2_mat[i, :]

        dW_v = sqrt(dt) .* Z1
        dW_s = sqrt(dt) .* (rho .* Z1 .+ sqrt(1 - rho^2) .* Z2)

        v_prev = v[i, :]
        v_pos  = max.(v_prev, 0)

        v_next = v_prev .+
                 kappa .* (theta_bar .- v_pos) .* dt .+
                 sigma .* sqrt.(v_pos) .* dW_v

        v_next = max.(v_next, 0)

        S_prev = S[i, :]
        S_next = S_prev .* exp.((r .- 0.5 .* v_pos) .* dt .+ sqrt.(v_pos) .* dW_s)

        S[i+1, :] .= S_next
        v[i+1, :] .= v_next
    end

    return S, v
end

S_paths, v_paths = simulate_heston_paths(S0, r, T, theta_true, Z1_mat, Z2_mat)

println("Simulation complete.")
println("Min variance: ", minimum(v_paths))
println("Max variance: ", maximum(v_paths))
println("Mean terminal price: ", mean(S_paths[end, :]))
println("Mean terminal variance: ", mean(v_paths[end, :]))

# ===============================
# MARKET DATA
# ===============================

ST = S_paths[end, :]

function mc_call_prices(ST, strikes, r, T)
    prices = zeros(length(strikes))
    for (j, K) in enumerate(strikes)
        payoffs = max.(ST .- K, 0.0)
        prices[j] = exp(-r * T) * mean(payoffs)
    end
    return prices
end

market_option_prices = mc_call_prices(ST, strikes, r, T)

println("True market option prices at given strikes:")
for (K, C) in zip(strikes, market_option_prices)
    println("K=",K," -> ",round(C,digits=4))
end

# ===============================
# MOMENTS 
# ===============================

function compute_moments(log_returns, terminal_variance)
    m1 = mean(log_returns)
    centered = log_returns .- m1
    m2 = mean(centered .^ 2)
    m3 = mean(centered .^ 3) / (m2^(3/2))
    m4 = mean(centered .^ 4) / (m2^2) - 3.0
    m5 = mean(terminal_variance)

    return [m1, m2, m3, m4, m5]
end

log_returns = log.(ST ./ S0)
terminal_variance = v_paths[end, :]

m_data = compute_moments(log_returns, terminal_variance)

println("Target moments:")
println(m_data)

# ===============================
# TERMINAL SIMULATION (FOR LOSS)
# doesn't save entire paths 
# only terminal values (faster)
# better for AAD
# ===============================

function simulate_terminal(S0, r, T, theta, Z1_mat, Z2_mat)
    kappa, theta_bar, sigma, rho, v0 = theta

    nsteps, npaths = size(Z1_mat)
    dt = T / nsteps
    eps_v = 1e-8

    S = fill(S0, npaths)
    v = fill(v0, npaths)

    for i in 1:nsteps
        Z1 = Z1_mat[i, :]
        Z2 = Z2_mat[i, :]

        dW_v = sqrt(dt) .* Z1
        dW_s = sqrt(dt) .* (rho .* Z1 .+ sqrt(1 - rho^2) .* Z2)

        v_pos = max.(v, eps_v)

        v = max.(
            v .+
            kappa .* (theta_bar .- v_pos) .* dt .+
            sigma .* sqrt.(v_pos) .* dW_v,
            eps_v
        )

        S = S .* exp.((r .- 0.5 .* v_pos) .* dt .+ sqrt.(v_pos) .* dW_s)
    end

    return S, v
end

# ===============================
# MSM LOSS
# ===============================

function model_moments(S0, r, T, theta, Z1_mat, Z2_mat)
    ST_sim, vT_sim = simulate_terminal(S0, r, T, theta, Z1_mat, Z2_mat)
    log_returns_sim = log.(ST_sim ./ S0)
    return compute_moments(log_returns_sim, vT_sim)
end

function msm_loss(S0, r, T, theta, Z1_mat, Z2_mat, m_data, W)
    m_model = model_moments(S0, r, T, theta, Z1_mat, Z2_mat)
    diff = m_data .- m_model
    return dot(diff, W * diff)
end

W = Matrix{Float64}(I, 5, 5)

function loss(theta)
    return msm_loss(S0, r, T, theta, Z1_mat, Z2_mat, m_data, W)
end

println("Loss true:", loss(theta_true))
println("Loss null:", loss(theta_null))

# ===============================
# FD GRADIENT
# ===============================

function finite_difference_gradient(f, theta; h=1e-4)
    grad = zeros(length(theta))

    for i in 1:length(theta)
        theta_plus = copy(theta)
        theta_minus = copy(theta)

        theta_plus[i] += h
        theta_minus[i] -= h

        grad[i] = (f(theta_plus) - f(theta_minus)) / (2h)
    end

    return grad
end

grad_fd = finite_difference_gradient(loss, theta_null)

println("FD gradient:")
println(grad_fd)

# ===============================
# AAD GRADIENT
# ===============================

grad_aad = ForwardDiff.gradient(loss, theta_null)

println("AAD gradient:")
println(grad_aad)

println("Difference FD - AAD:")
println(grad_fd .- grad_aad)

# ===============================
# GRADIENT DESCENT UPDATE
# ===============================

function update_theta(theta, grad, step_size)
    return theta .- step_size .* grad
end

# ===============================
# OPTIMIZATION ALGORITHM
# ===============================

function calibrate(theta_init; method, step_size, max_iter, tol, h)
    theta = copy(theta_init)

    loss_history = Float64[]
    theta_history = Vector{Float64}[]

    for iter in 1:max_iter
        current_loss = loss(theta)

        if method == :FD
            grad = finite_difference_gradient(loss, theta; h=h)
        else method == :AAD
            grad = ForwardDiff.gradient(loss, theta)
        end

        theta_new = update_theta(theta, grad, step_size)

        push!(loss_history, current_loss)
        push!(theta_history, copy(theta))

        println("Iteration $iter")
        println("theta = ", theta)
        println("loss  = ", current_loss)
        println("grad  = ", grad)
        println()

        # convergence check 1: parameter update becomes very small
        if norm(theta_new .- theta) < tol
            println("Converged: parameter change below tolerance.")
            theta = theta_new
            break
        end

        # convergence check 2: loss change becomes very small
        if iter > 1 && abs(loss_history[end] - loss_history[end-1]) < tol
            println("Converged: loss change below tolerance.")
            theta = theta_new
            break
        end

        theta = theta_new
    end

    final_loss = loss(theta)

    return theta, final_loss, loss_history, theta_history
end

# ===============================
# RUN FULL CALIBRATION
# ===============================

println("===================================")
println("CALIBRATION WITH FINITE DIFFERENCES")
println("===================================")

theta_fd, final_loss_fd, loss_hist_fd, theta_hist_fd = calibrate(
    theta_null;
    method=:FD,
    step_size=1e-4,
    max_iter=50,
    tol=1e-10,
    h=1e-4
)

println("Final FD parameters: ", theta_fd)
println("Final FD loss: ", final_loss_fd)
println()

println("===================================")
println("CALIBRATION WITH AAD")
println("===================================")

theta_aad, final_loss_aad, loss_hist_aad, theta_hist_aad = calibrate(
    theta_null;
    method=:AAD,
    step_size=1e-4,
    max_iter=50,
    tol=1e-10,
    h=1e-4
)

println("Final AAD parameters: ", theta_aad)
println("Final AAD loss: ", final_loss_aad)
println()

println("===================================")
println("COMPARISON")
println("===================================")
println("True theta: ", theta_true)
println("Initial theta: ", theta_null)
println("FD final theta: ", theta_fd)
println("AAD final theta: ", theta_aad)
println("FD final loss: ", final_loss_fd)
println("AAD final loss: ", final_loss_aad)