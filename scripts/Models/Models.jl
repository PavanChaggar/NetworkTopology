using DifferentialEquations, LightGraphs, Plots, SimpleWeightedGraphs, LinearAlgebra, Turing, Base.Threads
Turing.setadbackend(:forwarddiff)

"""
    MakeSimpleWeightedGraph(n, p)

Create a simple weighted random graph of `n` nodes with connection probability `p`.
"""
function MakeSimpleWeightedGraph(n::Int64, p::Float64)
    GW = SimpleWeightedGraph(erdos_renyi(n, p))
    for e in edges(GW)
        add_edge!(GW, src(e), dst(e), rand())
    end
    return GW
end

function NetworkAtrophy(du, u, p, t; L=L)
    n = Int(length(u)/2)

    x = u[1:n]
    y = u[n+1:2n]

    ρ, α, β = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

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

function plot_predictive(chain, sol, data, node)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:500
        par = chain[rand(1:10_000), 1:8]
        resol = solve(remake(problem,u0=par[3:7], p=par[1:2]),AutoTsit5(Rosenbrock23()),saveat=0.05)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    return scatter!(data[node,:], legend = false)
end