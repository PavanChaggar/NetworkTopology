using DifferentialEquations, LightGraphs, Random, Turing
using Turing: Variational
#using Plots
#using StatsPlots
using LinearAlgebra
#=
This solved for initial values of FKPP model on network using forward diff and 
advi. 
=#

Random.seed!(1)

const N = 10
const P = 1.0

G = erdos_renyi(N, P)
L = laplacian_matrix(G)


function NetworkFKPP(u, p, t)
    κ, α = p 
    du = -κ * L * u .+ α .* u .* (1 .- u)
end

u0 = rand(N)
p = [2.0, 3.0]
t_span = (0.0, 0.5)

problem = ODEProblem(NetworkFKPP, u0, t_span, p)
sol = solve(problem, AutoTsit5(Rosenbrock23()), saveat=0.005)

plot(sol)

const data = Array(sol)

Turing.setadbackend(:forwarddiff)
@model function fit(data, func, ::Type{T} = Float64) where {T}
    σ ~ InverseGamma(2, 3)
    k ~ truncated(Normal(5,10.0),0.0,10)
    a ~ truncated(Normal(5,10.0),0.0,10)

    u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), N)
    #u ~ MvNormal(0.5*ones(N),I)

    p = [k, a] 

    prob = remake(problem, u0=u, p=p)

    predicted = solve(prob, AutoTsit5(Rosenbrock23()), saveat=0.005)

    for i ∈ 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ) 
    end 
end 

model = fit(data, NetworkFKPP)
advi = ADVI(10, 1000)
opt = Variational.DecayedADAGrad(1e-3, 1.1, 0.9)
q = vi(model, advi)

samples = rand(q, 10000)
avg = vec(mean(samples; dims=2))
