using DelimitedFiles
using SparseArrays
using Statistics
using SimpleWeightedGraphs
using LightGraphs
using DifferentialEquations
using Turing
using Plots
using Base.Threads
Turing.setadbackend(:forwarddiff)

include("../Models/Models.jl")
include("../Models/Matrices.jl")

const csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"
const subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"
const subjects = read_subjects(csv_path);

const An = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, false));
const Al = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, true));

const A = diffusive_weight(An, Al);
const L = max_norm(laplacian_matrix(A));

function NetworkAtrophyODE(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]
	
    α, β, ρ = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

@model function NetworkAtrophyPM(data, problem)
	n = Int(size(data)[1])

    σ ~ InverseGamma(2, 3)	
    a ~ truncated(Normal(1, 3), 0, 10)
    b ~ truncated(Normal(1, 5), 0, 10)
	r ~ truncated(Normal(0, 1), 0, 10)

    p = [a, b, r]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=2.0)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[n+1:end,i], σ)
    end

    return a, b, r
end


p = [2.5, 1.0, 0.4]
protein = zeros(83)
protein[[27,68]] .= 0.1;
u0 = [protein; zeros(83)];
t_span = (0.0, 10.0);

problem = ODEProblem(NetworkAtrophyODE, u0, t_span, p)
sol = solve(problem, Tsit5(), saveat=2.0)
fullsol = solve(problem, Tsit5(), saveat=0.1)

data = clamp.(Array(sol) + 0.02 * randn(size(Array(sol))), 0.0,1.0)

atrophy_data = data[84:end,:]

model = NetworkAtrophyPM(atrophy_data, problem)

#prior_chain = sample(model, Prior(), 1_000)

chain = sample(model, NUTS(0.65), MCMCThreads(), 1_000, 10, progress=true)
#chain = sample(model, NUTS(0.65), 5_00, progress=true)
#using Serialization
#serialize("/home/chaggar/Projects/NetworkTopology/Chains/NetworkAtrophy_83_3params.jls", chain)

serialisedchain = deserialize("/home/chaggar/Projects/NetworkTopology/Chains/NetworkAtrophy_83_3params.jls")
chain_array = Array(serialisedchain)

nodes = rand(1:83, 5)

fig = plot(xlims=(0,10))
for k in 1:300
    par = chain_array[rand(1:10_000), 1:3]
    resol = solve(remake(problem, p=[par[1],par[2],par[3]]),Tsit5())
    plot!(fig, resol, vars=(nodes), alpha=0.5, color = "#BBBBBB", legend = false)
end
plot!(fig, fullsol, vars=(nodes), legend = false, palette=:Dark2_5)
scatter!(fig, 0:2:10, data[nodes,:]', legend =false, palette=:Dark2_5)

chainplot = plot(serialisedchain)

savefig(chainplot, "/home/chaggar/Projects/NetworkTopology/ChainSummary.png")