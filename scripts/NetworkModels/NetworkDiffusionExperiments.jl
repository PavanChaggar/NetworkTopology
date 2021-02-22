using Random
using DifferentialEquations
using Turing 
using LightGraphs
using Base.Threads
using Plots, StatsPlots, MCMCChains
Turing.setadbackend(:forwarddiff)
Random.seed!(1)

function make_graph(N::Int64, P::Float64)
    G = erdos_renyi(N, P)
    L = laplacian_matrix(G)
    A = adjacency_matrix(G)
    return L, A
end

function NetworkDiffusion(u, p, t)
    du = -p * L * u
end

@model function fitode(data, problem)
    u_n = size(data)[1]
    σ ~ InverseGamma(2, 3) # ~ is the tilde character
    ρ ~ truncated(Normal(5,10.0),0.0,10)
    u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), u_n)

    prob = remake(problem, u0=u, p=ρ)
    predicted = solve(prob, AutoTsit5(Rosenbrock23()),saveat=0.05)

    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

L, A = make_graph(5,0.5)

u0 = rand(5)
p = 2

problem = ODEProblem(NetworkDiffusion, u0, (0.0,1.0), p);
sol = solve(problem, AutoTsit5(Rosenbrock23()), saveat=0.05)
data = Array(sol)

model = fitode(data,problem)
chain = sample(model, NUTS(0.65), MCMCThreads(), 1_000, 10, progress=true)

plot(chain)


pl = scatter(sol.t, data');
chain_array = Array(chain)
for k in 1:300 
    resol = solve(remake(problem,p=chain_array[rand(1:3000), 1:4]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
plot!(sol1, w=1, legend = false)