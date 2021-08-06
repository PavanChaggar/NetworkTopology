using DelimitedFiles
using SparseArrays
using Statistics
using SimpleWeightedGraphs
using LightGraphs
using DifferentialEquations
using Turing
using Plots
using Base.Threads
using BenchmarkTools
using ReverseDiff

include("../Models/Models.jl")
include("../Models/Matrices.jl")

const csv_path = "/Users/pavanchaggar/Projects/Connectomes/all_subjects"
const subject_dir = "/Users/pavanchaggar/Projects/Connectomes/standard_connectome/scale1/subjects/"
const subjects = read_subjects(csv_path);

const An = load_connectome(subjects, subject_dir, 100, 83, false) |> mean_connectome
const Al = load_connectome(subjects, subject_dir, 100, 83, true) |> mean_connectome

const A = diffusive_weight(An, Al) |> max_norm
const L = laplacian(A)

function NetworkAtrophyODE(du, u0, p, t)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]

    κ, α, β = p

    du[1:n] .= -κ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

@model function NetworkAtrophyPM(data, problem)
    σ ~ InverseGamma(2, 3)	
	
    k ~ truncated(Normal(0, 1), 0, 5)
	a ~ truncated(Normal(0, 5), 0, 10)
    b ~ truncated(Normal(0, 5), 0, 10)

    p = [k, a, b]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=2.0)

    data ~ arraydist(Normal.(predicted[84:end,:], σ))
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

model()

function plot_predictive(chain, prob, sol, data; N=200, node=27) 
	plot(sol, vars=(node), w=2, legend = false)
    for i in 1:N
        resol = solve(remake(prob, 
							 p=[chain[:k][i], chain[:a][i], chain[:b][i]]),
							 Tsit5(),
			                 saveat=0.5)
        plot!(resol, vars=(node), alpha=0.1, color=:grey,legend=false)
	end
    scatter!(sol.t, data[node,:], legend = false)
end

prior_chain = sample(model, Prior(), 1_000)

plot_predictive(prior_chain, problem, sol, data)

chain = sample(model, NUTS(0.65), 1_000, progress=true)

plot_predictive(chain, problem, sol, data; node=27)