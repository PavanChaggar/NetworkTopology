using DifferentialEquations, LightGraphs, Random, Turing
using Turing: Variational
using LinearAlgebra
using Bijectors
using Bijectors: Scale, Shift
using Plots
using StatsPlots
Turing.setadbackend(:forwarddiff)

Random.seed!(1)

function graph_laplacian(N::Int64, P::Float64)
    L = laplacian_matrix(erdos_renyi(N, P))
end

function makeFKPP(L, u=nothing, p=nothing, t=nothing)
    function ode(u, p, t)
        κ, α = p 
        du = -κ * L * u .+ α .* u .* (1 .- u)
    end 
    return ode
end

function simulate(func, u0, t_span, p)
    problem = ODEProblem(func, u0, t_span, p)
    sol = solve(problem, Tsit5(), saveat=0.05)
    data = Array(sol)
    return data, problem
end

@model function fit(data, problem)
    σ ~ InverseGamma(2, 3)
    k ~ truncated(Normal(5,10.0),0.0,10)
    a ~ truncated(Normal(5,10.0),0.0,10)
    u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), 5)

    p = [k, a] 

    prob = remake(problem, u0=u, p=p)

    predicted = solve(prob, Tsit5(), saveat=0.05)

    for i ∈ 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end 
end 

function getq(θ)
    offset = 0
    L = LowerTriangular(reshape(@inbounds(θ[offset + 1: offset + d^2]), (d, d)))
    offset += d^2
    b = @inbounds θ[offset + 1: offset + d]
    
    # For this to represent a covariance matrix we need to ensure that the diagonal is positive.
    # We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
    D = Diagonal(diag(L))
    A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries
    
    b = to_constrained ∘ Shift(b; dim = Val(1)) ∘ Scale(A; dim = Val(1))
    
    return transformed(base_dist, b)
end

N = 5
Lap = graph_laplacian(N, 0.7)
u0 = rand(N)
p = [2.0, 3.0]
t_span = (0.0, 2.0)
d = N + 3

func = makeFKPP(Lap)

data, prob = simulate(func, u0, t_span, p) 
model =  fit(data, prob)

base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))

to_constrained = inv(bijector(model));

advi = ADVI(10, 5_000)
q_full_normal = vi(model, advi, getq, randn(d^2 + d); optimizer = Variational.DecayedADAGrad(1e-2));

A = q_full_normal.transform.ts[1].a

heatmap(cov(A * A'))

samples = rand(q_full_normal, 5_000)
avg = vec(mean(samples; dims=2))