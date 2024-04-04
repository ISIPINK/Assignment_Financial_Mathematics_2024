using HighDimPDE
using Flux # needed to define the neural network

K = 110.0
x0 = [100.0]
dim = 1
d = dim
r = 0.02
mu = 0.07
t0 = 0.0
t1 = 1.0
sigma = 0.3


## Definition of the problem
tspan = (t0, t1) # time horizon
g(x) = max(0.0, maximum(K .- x)) # initial condition
μ(x, p, t) = mu * x # advection coefficients
σ(x, p, t) = sigma * x # diffusion coefficients
# x0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
q(x, y) = r * K * (y <= g(x))
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = -r * v_x + q(x, v_x)
prob = PIDEProblem(g, f, μ, σ, x0, tspan)



alg = MLP(M=20, L=4, K=10)

@time sol = solve(prob, alg, multithreading=true, verbose=false)
print(sol)
## Definition of the neural network to use

hls = dim + 50 #hidden layer size
nn = Flux.Chain(Dense(dim, hls, tanh),
    Dense(hls, hls, tanh),
    Dense(hls, 1)) # neural network used by the scheme

opt = ADAM(1e-2)

## Definition of the algorithm
alg = DeepSplitting(nn, opt=opt)

x0_sample = UniformSampling(fill(98.0, d), fill(102.0, d))

prob = PIDEProblem(g, f, μ, σ, x0, tspan, x0_sample=x0_sample)

# throws a mapfoldl error
sol = solve(prob,
    alg,
    0.1,
    verbose=true,
    abstol=2e-3,
    maxiters=1000,
    batch_size=1000)
print(sol)