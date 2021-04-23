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

# ╔═╡ c04ab0ba-738d-411d-9b6a-8419c8f4e5a7
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
	using MCMCChains
	using StatsPlots
	using Serialization
	Turing.setadbackend(:forwarddiff)
	gr()
	#include("../../../Projects/NetworkTopology/scripts/Models/Models.jl");
	include("../../../Projects/NetworkTopology/scripts/Models/Matrices.jl");
	md" ## Setting up modules here"
end

# ╔═╡ 387616a6-9ae9-11eb-1d2d-03a63f575d5f
md"# Inference on Network Models of Neurodegeneration 

To validate models, we plan to use Bayesian inference methods to estimate parameters and quantify uncertainty. This is easy to do using `Turing`, a probablistic programming language in Julia. As before, we need to load in the necessary parts of the model and write down our generative models of (simulated) data."

# ╔═╡ 1c3282b9-3588-4868-87a0-efe596f554a4
csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"

# ╔═╡ 0ca3b82b-ba3e-49b9-904e-8e9264bf4182
subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"

# ╔═╡ 142f7dbb-bd82-493e-863b-06da525adedf
subjects = read_subjects(csv_path);

# ╔═╡ ceb4c14f-c843-4ce9-8084-7830f5f9e4c0
An = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, false));

# ╔═╡ 85f60156-0191-4c2e-a0fc-88fe5ab166d2
Al = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, true));

# ╔═╡ b5015ea0-3a11-4514-8e99-0b4d259c32b0
A = diffusive_weight(An, Al);

# ╔═╡ 5f7f3402-43f9-4ca3-ad7a-6e018e66a86e
L = laplacian_matrix(max_norm(A));

# ╔═╡ 88770502-c951-47c2-8e26-8217424fcf17
function NetworkAtrophyODE(du, u0, p, t; L=L)
    n = Int(length(u0)/2)

    x = u0[1:n]
    y = u0[n+1:2n]
	
    α, β, ρ = p

    du[1:n] .= -ρ * L * x .+ α .* x .* (1.0 .- x)
    du[n+1:2n] .= β * x .* (1.0 .- y)
end

# ╔═╡ 90e507e9-b4b3-464a-94d1-cedfb77af3cb
begin	
	p = [2.5, 1.0, 0.4]
	protein = zeros(83)
	protein[[27,68]] .= 0.1;
	u0 = [protein; zeros(83)];
	t_span = (0.0,10);
end

# ╔═╡ 988fc805-035f-4f81-b208-eee784178e29
begin	
	prob = ODEProblem(NetworkAtrophyODE, u0, t_span, p)
	sol = solve(prob, Tsit5())
	datasol = solve(prob, Tsit5(), saveat=2)
end;

# ╔═╡ a5e7fc40-1d4d-4261-802f-50a3f309f83f
plot(sol, vars=(1:83), title="Protein Concentration", legend=false)

# ╔═╡ 5c3ed86b-0dad-41fb-9aec-76af58f83768
plot(sol, vars=(84:166), title="Atrophy", legend=false)

# ╔═╡ 19a1e3af-bb79-42c9-ae75-b199895a0283
data = clamp.(Array(datasol) + 0.02 * randn(size(Array(datasol))), 0.0,1.0);

# ╔═╡ 3e5b4fba-4018-438d-b650-c322c2fffa77
atrophy_data = data[84:end, :];

# ╔═╡ 52c40bad-e14e-4677-881e-22ec18cc923f
scatter(0:2:10, atrophy_data', title="Synthetic Atrophy Data", legend=false)

# ╔═╡ eb992c16-7105-4f03-871a-1c95264e9e97
md"## Inference using `Turing`

Now we have generated some synthetic data, we want to set up an inference procedure and check that it works. 

We can do this using `Turing`, a probablistic programming lanauge that allows for the easy model definition and fast sampling. 

Our model will take the following form: 

$y ≈ \mathcal{N}(f(α, β, ρ), σ)$

$α ≈ \mathcal{N}(1, 3, [0,10])$

$β ≈ \mathcal{N}(1, 5, [0,10])$

$ρ ≈ \mathcal{N}(0, 3, [0,10])$

$\sigma ≈ Γ^{-1}(1, 3, [0,10])$

We can write this up in `Turing` in the following way.
"

# ╔═╡ d9184495-87ed-403c-bc84-082266a031b3
@model function NetworkAtrophyPM(data, problem)
	n = Int(size(data)[1])

    σ ~ InverseGamma(2, 3)	
    a ~ truncated(Normal(1, 3), 0, Inf)
    b ~ truncated(Normal(1, 5), 0, Inf)
	r ~ truncated(Normal(0, 1), 0, Inf)

    p = [a, b, r]

    prob = remake(problem, p=p)
    
    predicted = solve(prob, Tsit5(), saveat=2)
    @threads for i = 1:length(predicted)
		data[:,i] ~ MvNormal(predicted[n+1:end,i], σ)
    end

    return a, b, r
end

# ╔═╡ a5751477-7c41-4f4b-a33a-304290cbd867
md"### Sampling from the prior
To check that the priors of the model are suitable, we can sample from the prior and run several forwad simulations. Ideally, we would hope for a wide output domain from the forward model that overlaps with the data. 
"

# ╔═╡ c0bdde32-c534-4fae-a50b-55c016dbd539
model = NetworkAtrophyPM(atrophy_data, prob);

# ╔═╡ 4449c9b3-b470-45b3-8716-911cd95bd292
prior_chain = sample(model, Prior(), 1_000);

# ╔═╡ af05bb27-2b7f-4e9b-9e8a-3e9be9380292
function plot_predictive(chain_array, prob, sol, node, runs)
	fig = plot(ylims=(0,1), xlims=(0,10))
	for k ∈ 1:runs
		par = Array(chain_array)[rand(1:1_000), 1:4]
		resol = solve(remake(prob, p=[par[1],par[2],par[3]]), Tsit5())
		plot!(fig, resol, vars=(node), alpha=0.5, color = "#BBBBBB", legend = false)
	end
	plot!(fig, sol, vars=(node), w=2, legend = false)
	scatter!(fig, 0:2:10, data[node,:], legend = false)
end

# ╔═╡ 9fa8ccd2-9152-41b3-a81f-81ce563d6aa0
plot_predictive(Array(prior_chain), prob, sol, 1, 300)

# ╔═╡ 03412e1c-1ac8-4338-affc-132cda37bbd4
md"### Sampling from the posterior

We can sample from the posterior using a variety of methods available in `Turing`, including adapative covariance MCMC, Random Walk MH, HMC, NUTS, ADVI. 

For the purpose of this demo, I've loaded sample made previously. This was run using NUTS with an target acceptance ratio of 0.65. 10 chains of 1000 samples were run in paralel."

# ╔═╡ 0fa23d14-4ceb-4e8a-bb4e-af1cee9b113c
chain = deserialize("/home/chaggar/Projects/NetworkTopology/Chains/NetworkAtrophy_83_3params.jls");

# ╔═╡ 4936c47b-aee6-450a-b052-5b6e9eb7ac89
plot(chain)

# ╔═╡ ea5bd477-222d-46c7-81ec-144c0831fb5e
summarize(chain)

# ╔═╡ ce0de31c-6975-4524-b5a2-a1124a1439d6
md"### Posterior predictive check 
Similarly to how we checked the the prior distributions, we can check the posterior distributions by drawing samples from the empirical posteior and running forward simulations."

# ╔═╡ fe8ded39-9733-4709-af15-2acdcba63221
@bind node Slider(0:1:10, show_value=true, default=1)

# ╔═╡ d8e51336-4189-429d-a8ee-fad6f30665e0
plotly()

# ╔═╡ 0643ff12-d688-45b2-8bfa-688ae64f2e29
plot_predictive(Array(chain), prob, sol, node, 50)

# ╔═╡ 55fa30f2-3035-41c1-a600-69c96225c257
md"## Next Steps: 

Use actual data!"


# ╔═╡ Cell order:
# ╟─387616a6-9ae9-11eb-1d2d-03a63f575d5f
# ╟─c04ab0ba-738d-411d-9b6a-8419c8f4e5a7
# ╠═1c3282b9-3588-4868-87a0-efe596f554a4
# ╠═0ca3b82b-ba3e-49b9-904e-8e9264bf4182
# ╠═142f7dbb-bd82-493e-863b-06da525adedf
# ╠═ceb4c14f-c843-4ce9-8084-7830f5f9e4c0
# ╠═85f60156-0191-4c2e-a0fc-88fe5ab166d2
# ╠═b5015ea0-3a11-4514-8e99-0b4d259c32b0
# ╠═5f7f3402-43f9-4ca3-ad7a-6e018e66a86e
# ╠═88770502-c951-47c2-8e26-8217424fcf17
# ╠═90e507e9-b4b3-464a-94d1-cedfb77af3cb
# ╠═988fc805-035f-4f81-b208-eee784178e29
# ╠═a5e7fc40-1d4d-4261-802f-50a3f309f83f
# ╠═5c3ed86b-0dad-41fb-9aec-76af58f83768
# ╠═19a1e3af-bb79-42c9-ae75-b199895a0283
# ╠═3e5b4fba-4018-438d-b650-c322c2fffa77
# ╠═52c40bad-e14e-4677-881e-22ec18cc923f
# ╟─eb992c16-7105-4f03-871a-1c95264e9e97
# ╠═d9184495-87ed-403c-bc84-082266a031b3
# ╟─a5751477-7c41-4f4b-a33a-304290cbd867
# ╠═c0bdde32-c534-4fae-a50b-55c016dbd539
# ╠═4449c9b3-b470-45b3-8716-911cd95bd292
# ╟─af05bb27-2b7f-4e9b-9e8a-3e9be9380292
# ╠═9fa8ccd2-9152-41b3-a81f-81ce563d6aa0
# ╟─03412e1c-1ac8-4338-affc-132cda37bbd4
# ╠═0fa23d14-4ceb-4e8a-bb4e-af1cee9b113c
# ╠═4936c47b-aee6-450a-b052-5b6e9eb7ac89
# ╠═ea5bd477-222d-46c7-81ec-144c0831fb5e
# ╟─ce0de31c-6975-4524-b5a2-a1124a1439d6
# ╠═fe8ded39-9733-4709-af15-2acdcba63221
# ╠═d8e51336-4189-429d-a8ee-fad6f30665e0
# ╠═0643ff12-d688-45b2-8bfa-688ae64f2e29
# ╟─55fa30f2-3035-41c1-a600-69c96225c257
