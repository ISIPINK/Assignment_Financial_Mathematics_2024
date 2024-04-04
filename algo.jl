using StochasticDiffEq
using Flux
using StochasticDiffEq
using Statistics
using Flux, Zygote, LinearAlgebra
using Functors
# using ProgressMeter: @showprogress
using Tracker
using CUDA, cuDNN
using Random
using SparseArrays

abstract type HighDimPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
abstract type AbstractPDEProblem <: SciMLBase.AbstractSciMLProblem end

abstract type AbstractSampling{T} end
Base.eltype(::AbstractSampling{T}) where {T} = eltype(T)

# Monte Carlo AbstractSampling
abstract type MCSampling{T} <: AbstractSampling{T} end

"""
    UniformSampling(a, b)

Uniform sampling for the Monte Carlo integration, in the hypercube `[a, b]^2`.
"""
struct UniformSampling{A} <: MCSampling{A}
    a::A
    b::A
end
@functor UniformSampling

function (mc_sample::UniformSampling{T})(x_mc, kwargs...) where {T}
    Tel = eltype(T)
    rand!(x_mc)
    m = (mc_sample.b + mc_sample.a) ./ convert(Tel, 2)
    x_mc .= (x_mc .- convert(Tel, 0.5)) .* (mc_sample.b - mc_sample.a) .+ m
end

"""
    NormalSampling(σ)
    NormalSampling(σ, shifted)

Normal sampling method for the Monte Carlo integration.

# Arguments
* `σ`: the standard deviation of the sampling
* `shifted` : if true, the integration is shifted by `x`. Defaults to false.
"""
struct NormalSampling{T} <: MCSampling{T}
    σ::T
    shifted::Bool # if true, we shift integration by x when invoking mc_sample::MCSampling(x)
end
@functor NormalSampling

NormalSampling(σ) = NormalSampling(σ, false)

function (mc_sample::NormalSampling)(x_mc)
    randn!(x_mc)
    x_mc .*= mc_sample.σ
end

function (mc_sample::NormalSampling)(x_mc, x)
    mc_sample(x_mc)
    mc_sample.shifted ? x_mc .+= x : nothing
end

struct NoSampling <: AbstractSampling{Nothing} end

(mc_sample::NoSampling)(x...) = nothing

function _integrate(::MCS) where {MCS<:AbstractSampling}
    if MCS <: NoSampling
        return false
    else
        return true
    end
end





struct PIDEProblem{uType,G,F,Mu,Sigma,xType,tType,P,UD,NBC,K} <:
       DiffEqBase.AbstractODEProblem{uType,tType,false}
    u0::uType
    g::G # initial condition
    f::F # nonlinear part
    μ::Mu
    σ::Sigma
    x::xType
    tspan::Tuple{tType,tType}
    p::P
    x0_sample::UD # the domain of u to be solved
    neumann_bc::NBC # neumann boundary conditions
    kwargs::K
end

"""

Defines a Partial Integro Differential Problem, of the form
```math
\\begin{aligned}
    \\frac{du}{dt} &= \\tfrac{1}{2} \\text{Tr}(\\sigma \\sigma^T) \\Delta u(x, t) + \\mu \\nabla u(x, t) \\\\
    &\\quad + \\int f(x, y, u(x, t), u(y, t), ( \\nabla_x u )(x, t), ( \\nabla_x u )(y, t), p, t) dy,
\\end{aligned}
```
with `` u(x,0) = g(x)``.

## Arguments

* `g` : initial condition, of the form `g(x, p, t)`.
* `f` : nonlinear function, of the form `f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(y, t), p, t)`.
* `μ` : drift function, of the form `μ(x, p, t)`.
* `σ` : diffusion function `σ(x, p, t)`.
* `x`: point where `u(x,t)` is approximated. Is required even in the case where `x0_sample` is provided. Determines the dimensionality of the PDE.
* `tspan`: timespan of the problem.
* `p`: the parameter vector.
* `x0_sample` : sampling method for `x0`. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default). If `NoSampling`, only solution at the single point `x` is evaluated.
* `neumann_bc`: if provided, Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.
"""
function PIDEProblem(μ,
    σ,
    x0::Union{Nothing,AbstractArray},
    tspan::TF,
    g,
    f;
    p::Union{Nothing,AbstractVector}=nothing,
    x0_sample::Union{Nothing,AbstractSampling}=NoSampling(),
    neumann_bc::Union{Nothing,AbstractVector}=nothing,
    kw...) where {TF<:Tuple{AbstractFloat,AbstractFloat}}


    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert(eltype(f(x0, x0, g(x0), g(x0), x0, x0, p, tspan[1])) == eltype(x0),
        "Type returned by non linear function `f` must match the type of `x0`")

    @assert eltype(g(x0)) == eltype(x0) "Type of `g(x)` must match the Type of x"

    PIDEProblem{typeof(g(x0)),
        typeof(g),
        typeof(f),
        typeof(μ),
        typeof(σ),
        typeof(x0),
        eltype(tspan),
        typeof(p),
        typeof(x0_sample),
        typeof(neumann_bc),
        typeof(kw)}(g(x0),
        g,
        f,
        μ,
        σ,
        x0,
        tspan,
        p,
        x0_sample,
        neumann_bc,
        kw)

end



struct ParabolicPDEProblem{uType,G,F,Mu,Sigma,xType,tType,P,UD,NBC,K} <:
       DiffEqBase.AbstractODEProblem{uType,tType,false}
    u0::uType
    g::G # initial condition
    f::F # nonlinear part
    μ::Mu
    σ::Sigma
    x::xType
    tspan::Tuple{tType,tType}
    p::P
    x0_sample::UD # the domain of u to be solved
    neumann_bc::NBC # neumann boundary conditions
    kwargs::K
end

"""

Defines a Parabolic Partial Differential Equation of the form:
```math
\\begin{aligned}
    \\frac{du}{dt} &= \\tfrac{1}{2} \\text{Tr}(\\sigma \\sigma^T) \\Delta u(x, t) + \\mu \\nabla u(x, t) \\\\
    &\\quad +  f(x, u(x, t), ( \\nabla_x u )(x, t), p, t)
\\end{aligned}
```

- Semilinear Parabolic Partial Differential Equation 
    * f -> f(X, u, σᵀ∇u, p, t)
- Kolmogorov Differential Equation
    * f -> `nothing`
    * x0 -> nothing, xspan must be provided.
- Obstacle Partial Differential Equation 
    * f -> `nothing`
    * g -> `nothing`
    * discounted payoff function provided.

## Arguments

* `μ` : drift function, of the form `μ(x, p, t)`.
* `σ` : diffusion function `σ(x, p, t)`.
* `x`: point where `u(x,t)` is approximated. Is required even in the case where `x0_sample` is provided. Determines the dimensionality of the PDE.
* `tspan`: timespan of the problem.
* `g` : initial condition, of the form `g(x, p, t)`.
* `f` : nonlinear function, of the form  `f(X, u, σᵀ∇u, p, t)`

## Optional Arguments 
* `p`: the parameter vector.
* `x0_sample` : sampling method for `x0`. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default). If `NoSampling`, only solution at the single point `x` is evaluated.
* `neumann_bc`: if provided, Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.
* `xspan`: The domain of the independent variable `x`
* `payoff`: The discounted payoff function. Required when solving for optimal stopping problem (Obstacle PDEs).
"""
function ParabolicPDEProblem(μ,
    σ,
    x0::Union{Nothing,AbstractArray},
    tspan;
    g=nothing,
    f=nothing,
    p::Union{Nothing,AbstractVector}=nothing,
    xspan::Union{Nothing,TF,AbstractVector{<:TF}}=nothing,
    x0_sample::Union{Nothing,AbstractSampling}=NoSampling(),
    neumann_bc::Union{Nothing,AbstractVector}=nothing,
    payoff=nothing,
    kw...) where {TF<:Tuple{AbstractFloat,AbstractFloat}}

    # Check the Initial Condition Function returns correct types.
    isnothing(g) && @assert !isnothing(payoff) "Either of `g` or `payoff` must be provided."

    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert !isnothing(x0) || !isnothing(xspan) "Either of `x0` or `xspan` must be provided."

    !isnothing(f) && @assert(eltype(f(x0, eltype(x0)(0.0), x0, p, tspan[1])) == eltype(x0),
        "Type of non linear function `f(x)` must type of x")

    # Wrap kwargs : 
    kw = NamedTuple(kw)
    prob_kw = (xspan=xspan, payoff=payoff)
    kwargs = merge(prob_kw, kw)

    # If xspan isa Tuple, then convert it as a Vector{Tuple} with single element
    xspan = isa(xspan, Tuple) ? [xspan] : xspan

    # if `x0` is not provided, pick up the lower-bound of `xspan`.
    x0 = isnothing(x0) ? first.(xspan) : x0

    # Initial Condition 
    u0 = if haskey(kw, :p_prototype)
        u0 = g(x0, kw.p_prototype.p_phi)
    else
        !isnothing(g) ? g(x0) : payoff(x0, 0.0)
    end
    @assert eltype(u0) == eltype(x0) "Type of `g(x)` must match the Type of x"

    ParabolicPDEProblem{typeof(u0),
        typeof(g),
        typeof(f),
        typeof(μ),
        typeof(σ),
        typeof(x0),
        eltype(tspan),
        typeof(p),
        typeof(x0_sample),
        typeof(neumann_bc),
        typeof(kwargs)}(u0,
        g,
        f,
        μ,
        σ,
        x0,
        tspan,
        p,
        x0_sample,
        neumann_bc,
        kwargs)
end



struct PIDESolution{X0,Ts,L,Us,NNs,Ls}
    x0::X0
    ts::Ts
    losses::L
    us::Us # array of solution evaluated at x0, ts[i]
    ufuns::NNs # array of parametric functions
    limits::Ls
    function PIDESolution(x0, ts, losses, usols, ufuns, limits=nothing)
        new{typeof(x0),
            typeof(ts),
            typeof(losses),
            typeof(usols),
            typeof(ufuns),
            typeof(limits)}(x0,
            ts,
            losses,
            usols,
            ufuns,
            limits)
    end
end

Base.summary(prob::PIDESolution) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::PIDESolution)
    println(io, summary(A))
    print(io, "timespan: ")
    show(io, A.ts)
    print(io, "\nu(x,t): ")
    show(io, A.us)
end

"""
```julia
DeepBSDE(u0,σᵀ∇u;opt=Flux.Optimise.Adam(0.1))
```

[DeepBSDE algorithm](https://arxiv.org/abs/1707.02568), from J. Han, A. Jentzen and Weinan E. 

## Arguments
- `u0`: a Flux.jl `Chain` with a d-dimensional input and a 1-dimensional output for the solytion guess.
- `σᵀ∇u`: a Flux.jl `Chain` for the BSDE value guess.
- `opt`: the optimization algorithm to be used to optimize the neural networks. Defaults to `Flux.Optimise.Adam(0.1)`.

## Example
Black-Scholes-Barenblatt equation

```julia
d = 30 # number of dimensions
x0 = repeat([1.0f0, 0.5f0], div(d,2))
tspan = (0.0f0,1.0f0)
dt = 0.2
m = 30 # number of trajectories (batch size)

r = 0.05f0
sigma = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u - sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(sigma*X) #Matrix d x d
prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f)

hls  = 10 + d #hidden layer size
opt = Flux.Optimise.Adam(0.001)
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = DeepBSDE(u0, σᵀ∇u, opt=opt)

solve(prob, 
    pdealg, 
    EM(), 
    verbose=true, 
    maxiters=150, 
    trajectories=m, 
    sdealg=StochasticDiffEq., 
    dt=dt, 
    pabstol = 1f-6)
```
"""
struct DeepBSDE{C1,C2,O} <: HighDimPDEAlgorithm
    u0::C1
    σᵀ∇u::C2
    opt::O
end

DeepBSDE(u0, σᵀ∇u; opt=Flux.Optimise.Adam(0.1)) = DeepBSDE(u0, σᵀ∇u, opt)

"""

Returns a `PIDESolution` object.

# Arguments

- `sdealg`: a SDE solver from [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/solvers/sde_solve/). 
    If not provided, the plain vanilla [DeepBSDE](https://arxiv.org/abs/1707.02568) method will be applied.
    If provided, the SDE associated with the PDE problem will be solved relying on 
    methods from DifferentialEquations.jl, using [Ensemble solves](https://diffeq.sciml.ai/stable/features/ensemble/) 
    via `sdealg`. Check the available `sdealg` on the 
    [DifferentialEquations.jl doc](https://diffeq.sciml.ai/stable/solvers/sde_solve/).
- `limits`: if `true`, upper and lower limits will be calculated, based on 
    [Deep Primal-Dual algorithm for BSDEs](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071506).
- `maxiters`: The number of training epochs. Defaults to `300`
- `trajectories`: The number of trajectories simulated for training. Defaults to `100`
- Extra keyword arguments passed to `solve` will be further passed to the SDE solver.
"""
function DiffEqBase.solve(prob::ParabolicPDEProblem,
    pdealg::DeepBSDE,
    sdealg;
    verbose=false,
    maxiters=300,
    trajectories=100,
    dt=eltype(prob.tspan)(0),
    pabstol=1.0f-6,
    save_everystep=false,
    limits=false,
    ensemblealg=EnsembleThreads(),
    trajectories_upper=1000,
    trajectories_lower=1000,
    maxiters_limits=10,
    kwargs...)
    x0 = prob.x
    tspan = prob.tspan
    d = length(x0)
    g, f, μ, σ = prob.g, prob.f, prob.μ, prob.σ
    p = prob.p isa AbstractArray ? prob.p : Float32[]
    A = haskey(kwargs, :A) ? prob.A : nothing
    u_domain = prob.x0_sample
    data = Iterators.repeated((), maxiters)

    #hidden layer
    opt = pdealg.opt
    u0 = pdealg.u0
    σᵀ∇u = pdealg.σᵀ∇u
    p1, _re1 = Flux.destructure(u0)
    p2, _re2 = Flux.destructure(σᵀ∇u)
    p3 = [p1; p2; p]
    ps = Flux.params(p3)

    re1 = p -> _re1(p[1:length(p1)])
    re2 = p -> _re2(p[(length(p1)+1):(length(p1)+length(p2))])
    re3 = p -> p[(length(p1)+length(p2)+1):end]

    function F(h, p, t)
        u = h[end]
        X = h[1:(end-1)]
        t_ = eltype(X)(t)
        _σᵀ∇u = re2(p)([X; t_])'
        _p = re3(p)
        _f = -f(X, u, _σᵀ∇u, _p, t_)
        vcat(μ(X, _p, t), [_f])
    end

    function G(h, p, t)
        X = h[1:(end-1)]
        _p = re3(p)
        t_ = eltype(X)(t)
        _σᵀ∇u = re2(p)([X; t_])'
        vcat(σ(X, _p, t), _σᵀ∇u)
    end

    # used for AD
    function F(h::Tracker.TrackedArray, p, t)
        u = h[end]
        X = h[1:(end-1)].data
        t_ = eltype(X)(t)
        _σᵀ∇u = σᵀ∇u([X; t_])' |> collect
        _f = -f(X, u.data, _σᵀ∇u, p, t_)

        Tracker.TrackedArray(vcat(μ(X, p, t), [_f]))
    end

    function G(h::Tracker.TrackedArray, p, t)
        X = h[1:(end-1)].data
        t_ = eltype(X)(t)
        _σᵀ∇u = σᵀ∇u([X; t_])' |> collect
        Tracker.TrackedArray(vcat(σ(X, p, t), _σᵀ∇u))
    end

    noise = zeros(Float32, d + 1, d)
    prob = SDEProblem{false}(F, G, [x0; 0.0f0], tspan, p3, noise_rate_prototype=noise)

    function neural_sde(init_cond)
        map(1:trajectories) do j #TODO add Ensemble Simulation
            predict_ans = Array(solve(prob, sdealg;
                dt=dt,
                u0=init_cond,
                p=p3,
                save_everystep=false,
                sensealg=SciMLSensitivity.TrackerAdjoint(),
                kwargs...))[:, end]
            (X, u) = (predict_ans[1:(end-1)], predict_ans[end])
        end
    end

    function predict_n_sde()
        _u0 = re1(p3)(x0)
        init_cond = [x0; _u0]
        neural_sde(init_cond)
    end

    function loss_n_sde()
        mean(sum(abs2, g(X) - u) for (X, u) in predict_n_sde())
    end

    iters = eltype(x0)[]
    losses = eltype(x0)[]
    cb = function ()
        save_everystep && push!(iters, u0(x0)[1])
        l = loss_n_sde()
        push!(losses, l)
        verbose && println("Current loss is: $l")
        l < pabstol && Flux.stop()
    end

    verbose && println("DeepBSDE")
    Flux.train!(loss_n_sde, ps, data, opt; cb=cb)

    if !limits
        # Returning iters or simply u0(x0) and the trained neural network approximation u0
        if save_everystep
            sol = PIDESolution(x0, tspan[1]:dt:tspan[2], losses, iters, re1(p3))
        else
            sol = PIDESolution(x0, tspan[1]:dt:tspan[2], losses, re1(p3)(x0)[1], re1(p3))
        end
        save_everystep ? iters : re1(p3)(x0)[1]
        return sol
    else
        verbose && println("Upper limit")
        if iszero(dt) == true
            error("dt choice is required for upper and lower bound calculation ")
        end
        sdeProb = SDEProblem(μ, σ, x0, tspan, noise_rate_prototype=zeros(Float32, d, d))
        output_func(sol, i) = (sol[end], false)
        ensembleprob = EnsembleProblem(sdeProb, output_func=output_func)
        sim_f = solve(ensembleprob,
            sdealg,
            ensemblealg,
            dt=dt,
            trajectories=trajectories_upper)
        Xn = reduce(vcat, sim_f.u)
        Un = collect(g(X) for X in Xn)

        tspan_rev = (tspan[2], tspan[1])
        sdeProb2 = SDEProblem(F,
            G,
            [Xn[1]; Un[1]],
            tspan_rev,
            p3,
            noise_rate_prototype=noise)
        function prob_func(prob, i, repeat)
            SDEProblem(prob.f,
                prob.g,
                [Xn[i]; Un[i]],
                prob.tspan,
                prob.p,
                noise_rate_prototype=prob.noise_rate_prototype)
        end

        ensembleprob2 = EnsembleProblem(sdeProb2,
            prob_func=prob_func,
            output_func=output_func)
        sim = solve(ensembleprob2,
            sdealg,
            ensemblealg,
            dt=dt,
            trajectories=trajectories_upper,
            output_func=output_func,
            save_everystep=false,
            sensealg=TrackerAdjoint())

        function sol_high()
            map(sim.u) do u
                u[2]
            end
        end

        loss_() = sum(sol_high()) / trajectories_upper

        ps = Flux.params(u0, σᵀ∇u...)
        cb = function ()
            l = loss_()
            true && println("Current loss is: $l")
            l < 1e-6 && Flux.stop()
        end
        dataS = Iterators.repeated((), maxiters_upper)
        Flux.train!(loss_, ps, dataS, Flux.Optimise.Adam(0.01); cb=cb)
        u_high = loss_()

        verbose && println("Lower limit")
        # Function to precalculate the f values over the domain
        function give_f_matrix(X, urange, σᵀ∇u, p, t)
            map(urange.urange) do u
                f(X, u, σᵀ∇u, p, t)
            end
        end

        #The Legendre transform that uses the precalculated f values.
        function legendre_transform(f_matrix, a, urange)
            le = a .* (collect(urange.urange)) .- f_matrix
            return maximum(le)
        end

        # lowe
        ts = tspan[1]:dt:tspan[2]
        function sol_low()
            map(1:trajectories_lower) do j
                u = u0(x0)[1]
                X = x0
                I = zero(eltype(u))
                Q = zero(eltype(u))
                for i in 1:(length(ts)-1)
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u([X; 0.0f0])
                    dW = sqrt(dt) * randn(d)
                    u = u - f(X, u, _σᵀ∇u, p, t) * dt + _σᵀ∇u' * dW
                    X = X .+ μ(X, p, t) * dt .+ σ(X, p, t) * dW
                    f_matrix = give_f_matrix(X, u_domain, _σᵀ∇u, p, ts[i])
                    a_ = A[findmax(collect(A) .* u .-
                                   collect(legendre_transform(f_matrix, a, u_domain)
                                           for a in A))[2]]
                    I = I + a_ * dt
                    Q = Q + exp(I) * legendre_transform(f_matrix, a_, u_domain)
                end
                I, Q, X
            end
        end
        u_low = sum(exp(I) * g(X) - Q for (I, Q, X) in sol_low()) / (trajectories_lower)
        if save_everystep
            sol = PIDESolution(x0,
                tspan[1]:dt:tspan[2],
                losses,
                iters,
                re1(p3),
                (u_low, u_high))
        else
            sol = PIDESolution(x0,
                tspan[1]:dt:tspan[2],
                losses,
                re1(p3)(x0)[1],
                re1(p3),
                (u_low, u_high))
        end
        return sol
    end
end #pde_solve_ns