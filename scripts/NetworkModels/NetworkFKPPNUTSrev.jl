using DifferentialEquations, LightGraphs, Random, Turing
using Turing: Variational
using Plots
using StatsPlots
using ReverseDiff
using DiffEqSensitivity
using Zygote

Random.seed!(1)

const N = 5
const P = 1.0

G = erdos_renyi(N, P)
L = laplacian_matrix(G)


function NetworkFKPP(u, p, t)
    κ, α = p 
    du = -κ * L * u .+ α .* u .* (1 .- u)
end

u0 = rand(N)
p = [2, 3.0]
t_span = (0.0, 2.0)

problem = ODEProblem(NetworkFKPP, u0, t_span, p)
sol = solve(problem, Tsit5(), saveat=0.05)

plot(sol)

data = Array(sol)

Turing.setadbackend(:reversediff)
@model function fit(data, problem)
    σ ~ InverseGamma(2, 3)
    k ~ truncated(Normal(5,10.0),0.0,10)
    a ~ truncated(Normal(5,10.0),0.0,10)
    u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), 5)

    p = [k, a] 

    prob = remake(problem, u0=u, p=p)
    #prob = ODEProblem(func, u, (0.0,2.0), p)

    predicted = solve(prob, Tsit5(), saveat=0.05)

    for i ∈ 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end 
end 

model = fit(data, problem)

nuts = sample(model, NUTS(.65), 1_000)

plot(nuts)

