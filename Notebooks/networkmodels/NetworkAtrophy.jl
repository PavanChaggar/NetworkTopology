### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ dadc259e-9214-11eb-2b38-33b4cdb9e26b
begin
	using DelimitedFiles
	using SparseArrays
	using Statistics
	using PlutoUI
	using SimpleWeightedGraphs
	using LightGraphs
	using DifferentialEquations
	using Turing
	using Plots
	using PlutoUI
	using Base.Threads
	using LinearAlgebra
	Turing.setadbackend(:forwarddiff)
	gr()
end;

# ╔═╡ 20b75f23-fa3f-440c-8092-4843ed7a5397
begin
	using MCMCChains
	using StatsPlots
end

# ╔═╡ 77e34aac-b734-49b1-8041-ee42d5139fa1
using Serialization

# ╔═╡ af9c3535-d247-47a5-a8c3-a80e00af01bf
begin
	include("../scripts/Models/Models.jl");
	include("../scripts/Models/Matrices.jl");
end

# ╔═╡ b442e1b4-920c-11eb-0a77-411787683d43
md"# Network FKPP & Atrophy

Here, we will look at performing inference on a coupled network model of protein diffusion giveb by FKPP dynamics and strutural ROI atrophy given as a functiono protein concentration. 
Where as before we considered random networks, here we will consider networks produced using tractography."

# ╔═╡ 04360dc6-920e-11eb-3450-ff37ab783adc
md"## Load the Brain Network 

We will load the brain network from 100 subjects who have undergone tractography using FSL. Each subject contains two networks: number of connected streamlines and avergae length of streamlines. These are directed networks and both will be symmetrised as: 

$A_{sym} = \frac{1}{2}(A + A^{T})$

I will use weighted networks given by: 

$A_{i,j} = \frac{n_{i,j}}{l_{i,j}^2}$ 

However, options can be provided to switch the weighting of the network." 

# ╔═╡ 432a42e4-920f-11eb-1d98-87b08b70318d
#=begin
	# Functions to Load graph
	
	read_subjects(csv_path::String) = Int.(readdlm(csv_path))
	
	symmetrise(M) = 0.5 * (M + transpose(M))
	
	max_norm(M) = M ./ maximum(M)
	
	adjacency_matrix(file::String) = sparse(readdlm(file))
	
	laplacian_matrix(A::Array{Float64,2}) = SimpleWeightedGraphs.laplacian_matrix(SimpleWeightedGraph(A))
		
	function load_connectome(subjects, subject_dir, N, size; length::Bool)
		
		M = Array{Float64}(undef, size, size, N)
		
		if length == true
			connectome_type = "/fdt_network_matrix_lengths"
		else
			connectome_type = "/fdt_network_matrix"
		end
		
		for i ∈ 1:N
			file = subject_dir * string(subjects[i]) * connectome_type
			M[:,:,i] = symmetrise(adjacency_matrix(file))
		end
		
		return M
	end
	
	mean_connectome(M) = mean(M, dims=3)[:,:]

	
	function diffusive_weight(An, Al)
		A = An ./ Al.^2
		[A[i,i] = 0.0 for i in 1:size(A)[1]]
		return A
	end	
	
	function plot_predictive(chain_array, prob, sol, data, node::Int)
		plot(Array(sol)[node,:], w=2, legend = false)
		for k in 1:300
			par = chain_array[rand(1:1_000), 1:23]
			resol = solve(remake(prob,u0=par[4:23], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
			plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
		end
		scatter!(data[node,:], legend = false)
	end
 
md"##### Some functions are hidden here"
end
=#

# ╔═╡ f92933dc-9211-11eb-091c-11c31d59df21
csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"

# ╔═╡ 6c77c6b6-9215-11eb-1289-5558d371050e
subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"

# ╔═╡ cbae4f5c-9214-11eb-0807-cd37295f37ba
subjects = read_subjects(csv_path);

# ╔═╡ f54ae6e0-9214-11eb-2baa-fba9babd502e
An = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, false));

# ╔═╡ 1e560938-9216-11eb-15fa-4d0e02bff32d
Al = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, true));

# ╔═╡ 7c9b1dfe-9217-11eb-068b-d17b2a1192aa
A = diffusive_weight(An, Al);

# ╔═╡ a79f01a8-9217-11eb-2466-79e51dd4fd98
L = max_norm(laplacian_matrix(A));

# ╔═╡ 05e312f6-9216-11eb-2a63-31aa25bbdd62
md"Now we have an adjacency matrix and a Laplacian matrix and can start to run simulations on networks!"

# ╔═╡ 262a69b8-921d-11eb-30dd-b3f8bcbabeef
md"## Network Atrophy Model 

The model we'll be working with is the network atrophy model, given by the coupled ODEs: 

$\frac{d\mathbf{p}_i}{dt} = -\rho \sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i\left(1-\mathbf{p}_i\right)$ 

$\frac{d\mathbf{a}_i}{dt} = \beta \mathbf{p}_i (1 - \mathbf{a}_i)$ 

where $p$ is the toxic protein vector and $\mathbf{a}$ is the atrophy level vector."

# ╔═╡ 06f74d20-921f-11eb-3911-e774241dcbd5
function NetworkAtrophyODE(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]
	
    α, β, ρ = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

# ╔═╡ 537c2a70-9220-11eb-1991-079d887c9277
md"""
ρ = $(@bind ρ Slider(0:0.05:3, show_value=true, default=0.1))

α = $(@bind α Slider(0.1:0.1:5, show_value=true, default=1.0))

β = $(@bind β Slider(0.1:0.1:3, show_value=true, default=1.0))

tf = $(@bind tf Slider(1:1:20, show_value=true, default=10.0))
"""

# ╔═╡ 5ef83788-9228-11eb-32fe-8b092a2e7f90
log(ρ/α)

# ╔═╡ 60eb89f0-921e-11eb-2978-1f90d4769c1f
begin 
	p = [α, β, ρ]
	protein = zeros(83)
	protein[[27,68]] .= 0.1;
	u0 = [protein; zeros(83)];
	t_span = (0.0,tf);
	
	prob = ODEProblem(NetworkAtrophyODE, u0, t_span, p)
	sol = solve(prob, Tsit5(), saveat=tf/20)
end;

# ╔═╡ 20b51026-921f-11eb-2fd0-8f7b3dbd8069
plot(sol, vars=(1:83), legend=false)

# ╔═╡ 6395656d-f336-4087-a2a8-8b5e4d9aadb7
plot(sol, vars=(84:166), legend=false)

# ╔═╡ 31212946-9223-11eb-0593-87378deb5fc4
md"## Inference

Now that we have the ODE set up and we can write a probablist model for this using Turing to perform inference"

# ╔═╡ 9437f3b0-9224-11eb-1f20-5ba9384a8f73
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

# ╔═╡ 76f1dea6-9242-11eb-37a8-350ddacaadf5
#data = clamp.(Array(sol) + 0.02 * randn(size(Array(sol))), 0.0,1.0);

# ╔═╡ 4d49615b-4ac4-4edf-8f05-975966f8559a
data = Array(sol);

# ╔═╡ 34389c9a-9243-11eb-2979-4b1807ffdce5
model = NetworkAtrophyPM(data,prob);

# ╔═╡ 20164994-b115-49c4-8763-2886b121c4d3
prior_chain = sample(model, Prior(), 1_000);

# ╔═╡ c2942b0e-f9ba-47c4-b68e-ff6f64b0a22d
function plot_predictivenew(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:4]
        resol = solve(remake(prob, p=[par[1],par[2],par[3]]),Tsit5(),saveat=0.5)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end

# ╔═╡ e4ae1a45-05c5-435a-9a76-8fb9cd9a5a71
plot_predictivenew(Array(prior_chain), prob, sol, data, 5)

# ╔═╡ c9def5db-293f-4f7c-8a3f-c2d86e52c882
chain = sample(model, NUTS(0.65), 1_000)

# ╔═╡ 6517f6d6-2f80-48ff-9138-1890b04f645e
plot(chain)

# ╔═╡ ce419582-35c0-45a7-9e21-c8d145c25805
serialize("/home/chaggar/Projects/NetworkTopology/Chains/NetworkFKPPAtrophyChain.jls", chain)

# ╔═╡ 6e06d987-7f3d-468a-b5af-1f89343a92cd
atrophy_data = data[84:end, :]

# ╔═╡ 99c2e529-bf25-47a4-857e-ef03a156a19a
@model function NetworkAtrophyOnlyPM(data, problem)
	n = Int(size(data)[1])

    σ ~ InverseGamma(2, 3)	
    a ~ truncated(Normal(1, 3), 0, 10)
    b ~ truncated(Normal(1, 5), 0, 10)
	r ~ truncated(Normal(0, 1), 0, 10)

    #u ~ filldist(truncated(Normal(0, 0.1), 0, 1), 166)

    p = [a, b, r]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=0.5)
    @threads for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[n+1:end,i], σ)
    end

    return a, b, r
end

# ╔═╡ 30a4495d-a87d-4dc1-bd60-d0ae3dfd3159
atrophy_model = NetworkAtrophyOnlyPM(atrophy_data,prob);

# ╔═╡ d0ca10c0-8d6c-4c9d-92d6-98fb48e710a6
atrophy_prior_chain = sample(atrophy_model, Prior(), 1_000);

# ╔═╡ 7be0dd16-f367-439a-99af-a0da44208fd8
atrophy_chain = sample(atrophy_model, NUTS(0.65), 1_000)

# ╔═╡ 84410e38-889a-422e-829f-8d97df1ca274
plot(atrophy_chain)

# ╔═╡ d5801d27-5257-4de3-9465-a21986723d79
serialize("/home/chaggar/Projects/NetworkTopology/Chains/NetworkAtrophyChain2.jls", atrophy_chain)

# ╔═╡ Cell order:
# ╟─b442e1b4-920c-11eb-0a77-411787683d43
# ╠═dadc259e-9214-11eb-2b38-33b4cdb9e26b
# ╠═af9c3535-d247-47a5-a8c3-a80e00af01bf
# ╟─04360dc6-920e-11eb-3450-ff37ab783adc
# ╠═432a42e4-920f-11eb-1d98-87b08b70318d
# ╠═f92933dc-9211-11eb-091c-11c31d59df21
# ╠═6c77c6b6-9215-11eb-1289-5558d371050e
# ╠═cbae4f5c-9214-11eb-0807-cd37295f37ba
# ╠═f54ae6e0-9214-11eb-2baa-fba9babd502e
# ╠═1e560938-9216-11eb-15fa-4d0e02bff32d
# ╠═7c9b1dfe-9217-11eb-068b-d17b2a1192aa
# ╠═a79f01a8-9217-11eb-2466-79e51dd4fd98
# ╟─05e312f6-9216-11eb-2a63-31aa25bbdd62
# ╟─262a69b8-921d-11eb-30dd-b3f8bcbabeef
# ╠═06f74d20-921f-11eb-3911-e774241dcbd5
# ╟─537c2a70-9220-11eb-1991-079d887c9277
# ╠═5ef83788-9228-11eb-32fe-8b092a2e7f90
# ╠═60eb89f0-921e-11eb-2978-1f90d4769c1f
# ╠═20b51026-921f-11eb-2fd0-8f7b3dbd8069
# ╠═6395656d-f336-4087-a2a8-8b5e4d9aadb7
# ╟─31212946-9223-11eb-0593-87378deb5fc4
# ╠═9437f3b0-9224-11eb-1f20-5ba9384a8f73
# ╠═76f1dea6-9242-11eb-37a8-350ddacaadf5
# ╠═4d49615b-4ac4-4edf-8f05-975966f8559a
# ╠═34389c9a-9243-11eb-2979-4b1807ffdce5
# ╠═20164994-b115-49c4-8763-2886b121c4d3
# ╠═20b75f23-fa3f-440c-8092-4843ed7a5397
# ╠═c2942b0e-f9ba-47c4-b68e-ff6f64b0a22d
# ╠═e4ae1a45-05c5-435a-9a76-8fb9cd9a5a71
# ╠═c9def5db-293f-4f7c-8a3f-c2d86e52c882
# ╠═6517f6d6-2f80-48ff-9138-1890b04f645e
# ╠═77e34aac-b734-49b1-8041-ee42d5139fa1
# ╠═ce419582-35c0-45a7-9e21-c8d145c25805
# ╠═6e06d987-7f3d-468a-b5af-1f89343a92cd
# ╠═99c2e529-bf25-47a4-857e-ef03a156a19a
# ╠═30a4495d-a87d-4dc1-bd60-d0ae3dfd3159
# ╠═d0ca10c0-8d6c-4c9d-92d6-98fb48e710a6
# ╠═7be0dd16-f367-439a-99af-a0da44208fd8
# ╠═84410e38-889a-422e-829f-8d97df1ca274
# ╠═d5801d27-5257-4de3-9465-a21986723d79
