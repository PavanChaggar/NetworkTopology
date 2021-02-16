using DifferentialEquations, LightGraphs, Random, Turing
using Turing: Variational
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
    u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), size(data)[1])

    prob = remake(problem, u0=u, p=[k, a])

    predicted = solve(prob, Tsit5(), saveat=0.05)

    for i ∈ 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end 
end 

N = 5
L = graph_laplacian(N, 0.7)
u0 = rand(N)
p = [2.0, 3.0]
t_span = (0.0, 2.0)

func = makeFKPP(L)

data, prob = simulate(func, u0, t_span, p) 
model =  fit(data, prob)

nuts = sample(model, NUTS(.65), 1_000)
