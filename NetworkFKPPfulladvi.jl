### A Pluto.jl notebook ###
# v0.11.9

using Markdown
using InteractiveUtils

# ╔═╡ 16e8d2ae-6c80-11eb-3d72-d731d9e61759
begin 
	using LightGraphs
	using Random 
	using DifferentialEquations
	using LinearAlgebra
	using Turing
	using Turing: Variational
	using StatsPlots
	using Bijectors
	using Bijectors 
	using Bijectors: Scale, Shift
	Random.seed!(1)
end;

# ╔═╡ 0877e864-6c7f-11eb-1c62-7be614f5c3fb
md"
# Inference on Network FKPP using ADVI

In this notebook, I will demonstrate the use of a full advi to infer parameters from an FKPP model posed on a network. 

Here, full advi means a version of variational inference that does not make a mean-field assumption. 

The Network FKPP equation is given by 

$\frac{du}{dt} = \kappa \mathbf{L} \mathbf{u} + \alpha \mathbf{u} (1 - \mathbf{u})$

Where $\mathbf{L}$ corresponds to the graph Laplacian of the network. 

We will seek to infer the parametesr $\kappa$ and $\alpha$, as well as initial conditions on $\mathbf{u}$.

#### Setting up a random network

Firstly, we initialise an Erdos-Renyi random graph using LightGraphs.
"

# ╔═╡ e0307a4a-6c80-11eb-1d8d-a1e7f4caf8d5
begin
	N = 5
	G = erdos_renyi(N, 0.5)
end

# ╔═╡ f5c03742-6c80-11eb-1c84-a9e61c2a907b
begin 
	using Plots
	A = adjacency_matrix(G)
	L = laplacian_matrix(G)
	heatmap(Matrix(A))
end

# ╔═╡ f06f72f8-6c80-11eb-0c34-4f31d4c3ebb6
md"
We can plot the negative Laplacian matrix of this network as a heatmap to visualise how the network is connected. 
"

# ╔═╡ 85374226-6c81-11eb-1a89-43d427b09252
md"
#### Defining and solving an ODE Problem 

The next step is to define our ODE and solving it. For demonstration purposes, we will use the solution of the ODE as out simulated data. 
"

# ╔═╡ e14c4c46-6c81-11eb-36d4-ad741d3c3fe1
begin 
	function NetworkFKPP(u, p, t)
			κ, α = p 
			du = -κ * L * u .+ α .* u .* (1 .- u)
	end 
end

# ╔═╡ 0c2699c6-6c82-11eb-34d7-d1414600ecaa
begin 
	u0 = rand(N) # random vector for inittial conditions
	p = [1.5,3.0] # initial parameters for κ and α
	t_span = (0.0,2.0) # time span to integrate over
	
	problem = ODEProblem(NetworkFKPP, u0, t_span, p)
    sol = solve(problem, Tsit5(), saveat=0.05)
    data = Array(sol)
end;

# ╔═╡ ca9c31c6-6c88-11eb-16ac-f1c85604081f
scatter(data')

# ╔═╡ f24c5d2c-6c88-11eb-3a7a-c17339d1c586
md"
#### Inference 

Now that we have a model and some data, we need to construct a probalistic model. 

We define our model with the following priors: 

$σ ≈ Γ^{-1}(2, 3)$ 
$κ ≈ \mathcal{N}(5,10,[0,10])$ 
$α ≈ \mathcal{N}(5,10,[0,10])$ 
$u_{i} ≈ \mathcal{N}(0.5,2,[0,1])$


and our data is given by: 

$y_{i} ≈ \mathcal{N}(x_{i},σ)$

We can do this using Turing.
"

# ╔═╡ 74045bf8-6c89-11eb-0041-6f398c1700d4
begin
	Turing.setadbackend(:forwarddiff)	
	@model function fit(data, problem)
		σ ~ InverseGamma(2, 3)
		k ~ truncated(Normal(5,10.0),0.0,10)
		a ~ truncated(Normal(5,10.0),0.0,10)
		u ~ filldist(truncated(Normal(0.5,2.0),0.0,1.0), 5)

		p = [k, a] 

		prob = remake(problem, u0=u, p=p)

		predicted = solve(prob, Tsit5(), saveat=0.05)

		for i ∈ 1:length(predicted)
			data[:,i] ~ MvNormal(predicted[i], σ)
		end 
	end
	model = fit(data, problem)
end 

# ╔═╡ 29b014a2-6c8e-11eb-2080-4f2f9f4e0bba
md"
##### MCMC 
In the first instance, since we have few parameters, we can estimate them using MCMC. 
"

# ╔═╡ e0e26442-6c8c-11eb-06fd-1ff7dc80642e
begin
	#nuts = sample(model, NUTS(.65), 10_000)
	#plot(nuts)
end

# ╔═╡ 4c731320-6c8e-11eb-15a6-33b1a05aeb66
md"
##### Variational Inference
Instead of using MCMC, we can also use variational inference. In particular, automatic differentiation variational inference, using bijectors to map from a simplified normal distribution to a full multivariate normal distribution containing all parameters. 
"

# ╔═╡ ae3bc8f6-6c8e-11eb-2909-55c7450d90eb
begin 
	d = N + 2 + 1 # Number of nodes plus number of parameters + noise
	base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))
	to_constrained = inv(bijector(model));	
end;

# ╔═╡ 9a51c3f2-6c90-11eb-3120-19785b4ea352
begin 
	function getq(θ)
		offset = 0
		L = LowerTriangular(reshape(@inbounds(θ[offset + 1: offset + d^2]), (d, d)))
		offset += d^2
		b = @inbounds θ[offset + 1: offset + d]

		# For this to represent a covariance matrix we need to ensure that the diagonal is positive.
		# We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
		D = Diagonal(diag(L))
		A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries

		b = to_constrained ∘ Shift(b; dim = Val(1)) ∘ Scale(A; dim = Val(1))

		return transformed(base_dist, b)
	end
end

# ╔═╡ b0f614d2-6c90-11eb-1534-ad8d9c27a879
begin 
	advi = ADVI(10, 2_000)
	q_full_normal = vi(model, advi, getq, randn(d^2 + d); optimizer = Variational.DecayedADAGrad(1e-2))
end;

# ╔═╡ c99b308a-6c90-11eb-3d1a-ede74fba4381
begin 
	samples = rand(q_full_normal, 5_000)
	avg = vec(mean(samples; dims=2))
end

# ╔═╡ 4143c158-6c94-11eb-060f-7f7b59bf8e09
begin  	
	Σ = q_full_normal.transform.ts[1].a
	heatmap(cov(Σ * Σ'))
end

# ╔═╡ 8accd5f2-6c9a-11eb-22a1-115f3f00879c
heatmap(cov(Σ[1:5,1:5]))

# ╔═╡ Cell order:
# ╟─0877e864-6c7f-11eb-1c62-7be614f5c3fb
# ╠═16e8d2ae-6c80-11eb-3d72-d731d9e61759
# ╠═e0307a4a-6c80-11eb-1d8d-a1e7f4caf8d5
# ╟─f06f72f8-6c80-11eb-0c34-4f31d4c3ebb6
# ╟─f5c03742-6c80-11eb-1c84-a9e61c2a907b
# ╟─85374226-6c81-11eb-1a89-43d427b09252
# ╠═e14c4c46-6c81-11eb-36d4-ad741d3c3fe1
# ╠═0c2699c6-6c82-11eb-34d7-d1414600ecaa
# ╠═ca9c31c6-6c88-11eb-16ac-f1c85604081f
# ╟─f24c5d2c-6c88-11eb-3a7a-c17339d1c586
# ╠═74045bf8-6c89-11eb-0041-6f398c1700d4
# ╟─29b014a2-6c8e-11eb-2080-4f2f9f4e0bba
# ╠═e0e26442-6c8c-11eb-06fd-1ff7dc80642e
# ╟─4c731320-6c8e-11eb-15a6-33b1a05aeb66
# ╠═ae3bc8f6-6c8e-11eb-2909-55c7450d90eb
# ╠═9a51c3f2-6c90-11eb-3120-19785b4ea352
# ╠═b0f614d2-6c90-11eb-1534-ad8d9c27a879
# ╠═c99b308a-6c90-11eb-3d1a-ede74fba4381
# ╠═4143c158-6c94-11eb-060f-7f7b59bf8e09
# ╠═8accd5f2-6c9a-11eb-22a1-115f3f00879c
