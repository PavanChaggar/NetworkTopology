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

csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"
subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"
subjects = read_subjects(csv_path);

An = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, false));
Al = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, true));

A = diffusive_weight(An, Al);
L = max_norm(laplacian_matrix(A));

function NetworkAtrophyODE(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]
	
    α, β, ρ = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

@model function NetworkAtrophyPM(data, problem)
    σ ~ InverseGamma(2, 3)	
    a ~ truncated(Normal(1, 3), 0, 10)
    b ~ truncated(Normal(1, 5), 0, 10)
	r ~ truncated(Normal(0, 1), 0, 10)

    #u ~ filldist(truncated(Normal(0, 0.1), 0, 1), 166)

    p = [a, b, r]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.5)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end

    return a, b, r
end

@model function NetworkAtrophyOnly(data, problem, ::Type{T} = Float64) where {T} 
    n = Int(size(data)[1])

    σ ~ InverseGamma(2, 3)
    r ~ truncated(Normal(0, 1), 0, Inf)
    a ~ truncated(Normal(0, 2), 0, Inf)
    b ~ truncated(Normal(0, 2), 0, Inf)

    p = [r, a, b]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.1)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[n+1:end,i], σ)
    end

    return r, a, b
end

p = [2.5, 1.0, 0.3]
protein = zeros(83)
protein[[27,68]] .= 0.1;
u0 = [protein; zeros(83)];
t_span = (0.0, 10.0);

problem = ODEProblem(NetworkAtrophyODE, u0, t_span, p)
sol = solve(problem, Tsit5(), saveat=0.1)

data = Array(sol)
atrophy_data = data[84:end,:]

model = NetworkAtrophyPM(data,problem)

@time sample(model, Prior(), 1)

model = NetworkAtrophyOnly(atrophy_data, problem)

prior_chain = sample(model, Prior(), 1_000)
