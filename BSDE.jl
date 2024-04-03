using HighDimPDE
using Flux, Zygote, LinearAlgebra, Statistics
using Test, StochasticDiffEq

begin # solving at one unique point
    # one-dimensional heat equation
    x0 = [11.0f0]  # initial points
    tspan = (0.0f0, 5.0f0)
    dt = 0.5   # time step
    time_steps = div(tspan[2] - tspan[1], dt)
    d = 1      # number of dimensions
    m = 10     # number of trajectories (batch size)

    g(X) = sum(X .^ 2)   # terminal condition
    f(X, u, σᵀ∇u, p, t) = 0.0f0  # function from solved equation
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = 1.0
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hidden layer size
    opt = Flux.Optimise.Adam(0.005)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))
    # sub-neural network approximating the spatial gradients at time point
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 200,
        dt = dt,
        trajectories = m)

    u_analytical(x, t) = sum(x .^ 2) .+ d * t
    analytical_sol = u_analytical(x0, tspan[end])

    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
end