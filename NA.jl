using DifferentialEquations, LightGraphs, Plots 
using SimpleWeightedGraphs
using LinearAlgebra
using Turing
Turing.setadbackend(:forwarddiff)

plotly()

G = erdos_renyi(10, 0.5)
GW = SimpleWeightedGraph(G)
for e in edges(GW)
    add_edge!(GW, src(e), dst(e), rand())
end

L = laplacian_matrix(GW) 

heatmap(Matrix(L))

function NetworkAtrophy(du, u, p, t; L=L)
    n = Int(length(u0)/2)

    x = u[1:n]
    y = u[n+1:2n]

    ρ, α, β = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

p = [0.1, 2.0, 1.0]

protein = zeros(10) 
protein[rand(1:10)] = 0.1

u0 = [protein; zeros(10)]
t_span = (0.0,10.0)
prob = ODEProblem(NetworkAtrophy, u0, t_span, p)

sol = solve(prob, Tsit5())


plot(sol, vars=(1:10))
plot(sol, vars=(11:20))

@model function NetworkAtrophy(data, problem)
    sigma ~ InverseGamma(2, 3)
    r ~ truncated(Normal(0, 10), 0, inf)
    a ~ truncated(Normal(0, 10), 0, inf)
    b ~ truncated(Normal(0, 10), 0, inf)

    u0 = filldist(truncated(Normal(0, 10), 0, 1), 10)

    p = [r, a, b]
    prob(remake(problem, u0=u, p=p))
    
    predicted = solve(prob, Tsit5(), saveat=0.05)

    for i ∈ 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
    return r, a, b, u
end

