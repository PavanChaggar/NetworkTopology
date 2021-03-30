using DifferentialEquations, LightGraphs, Plots
using SimpleWeightedGraphs
using LinearAlgebra
using Turing
using Base.Threads
Turing.setadbackend(:forwarddiff)

gr()

#G = erdos_renyi(10, 0.5)
#GW = SimpleWeightedGraph(G)
#for e in edges(GW)
#    add_edge!(GW, src(e), dst(e), rand())
#end

function MakeSimpleWeightedGraph(n::Int64, p::Float64)
    GW = SimpleWeightedGraph(erdos_renyi(n, p))
    for e in edges(GW)
        add_edge!(GW, src(e), dst(e), rand())
    end
    return GW
end

GW = MakeSimpleWeightedGraph(10,0.5)

L = laplacian_matrix(GW)

heatmap(Array(adjacency_matrix(GW)))

function NetworkAtrophy(du, u, p, t; L=L)
    n = Int(length(u)/2)

    x = u[1:n]
    y = u[n+1:2n]

    ρ, α, β = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

p = [0.1, 2.0, 1.0]

protein = zeros(10) 
protein[rand(1:10)] = 0.1

u0 = [protein; zeros(10)]
t_span = (0.0,10.0)
prob = ODEProblem(NetworkAtrophy, u0, t_span, p)

sol = solve(prob, Tsit5(), saveat=0.1)
data = Array(sol)

plot(sol, vars=(1:10))
scatter(data')
plot(sol, vars=(11:20))

@model function NetworkAtrophy(data, problem)
    σ ~ InverseGamma(2, 3)
    r ~ truncated(Normal(0, 1), 0, Inf)
    a ~ truncated(Normal(0, 2), 0, Inf)
    b ~ truncated(Normal(0, 2), 0, Inf)

    u ~ filldist(truncated(Normal(0, 0.1), 0, 1), 20)

    p = [r, a, b]

    prob = remake(problem, u0=u, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.1)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end

    return r, a, b, u
end


prior_chain = sample(NetworkAtrophy(data,prob), Prior(), 10_000)

chain_array = Array(prior_chain)

plot(Array(sol)[10,:], w=1, legend = false)
for k in 1:500
    par = chain_array[rand(1:10_000), 1:23]
    resol = solve(remake(prob,u0=par[4:23], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
    plot!(Array(resol)[10,:], alpha=0.5, color = "#BBBBBB", legend = false)
end
scatter!(data[10,:], legend = false)

model = NetworkAtrophy(data,prob)

chain = sample(model, NUTS(0.65), 1_000)

chain_array = Array(chain)

avg = mean(chain_array, dims=1)

noden = 10
plot(Array(sol)[noden,:], w=1, legend = false)
for k in 1:100
    par = chain_array[rand(1:500), 1:23]
    resol = solve(remake(prob,u0=par[4:23], p=[par[3],par[1],par[2]]),AutoTsit5(Rosenbrock23()),saveat=0.1)
    plot!(Array(resol)[noden,:], alpha=0.5, color = "#BBBBBB", legend = false)
end
scatter!(data[noden,:], legend = false)

using StatsPlots, MCMCChains
plot(chain)