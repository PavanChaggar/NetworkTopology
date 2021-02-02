### A Pluto.jl notebook ###
# v0.11.9

using Markdown
using InteractiveUtils

# ╔═╡ 105b4cf4-64bb-11eb-07ef-2127bbaa69d9
using DifferentialEquations, LightGraphs, PlutoUI, Random, Plots, StatsPlots

# ╔═╡ ff1da1fc-64bb-11eb-12fc-9f7519373a3d
using Turing

# ╔═╡ ab19d95e-64bb-11eb-1613-576a2508b36f
using Turing: Variational

# ╔═╡ 474bf842-64bb-11eb-3678-93fd8b1e9aa5
Random.seed!(1)

# ╔═╡ 79a3153e-64c0-11eb-126f-bff2b909f67c
plotly()

# ╔═╡ 2b35a7c4-64be-11eb-2785-03c7cab059fe
n = 100

# ╔═╡ 5c626174-64bf-11eb-22b0-b3509b9899e3
p = 0.7

# ╔═╡ 401c3fcc-64be-11eb-3b48-cfb9749db66a
begin 
	G = erdos_renyi(n, p);
	L = laplacian_matrix(G);
end

# ╔═╡ 5bf1ef5e-64bb-11eb-297a-c971721eda48
NetworkDiffusion(u, p, t) = -p * L * u

# ╔═╡ 6ab4cec8-64bb-11eb-0cb5-9583308fd650
u0 = rand(n);

# ╔═╡ 086dcd3e-64be-11eb-0b13-c7371c6b367f
 ρ = 0.5

# ╔═╡ 6fca1080-64bb-11eb-37c0-77b51e264c31
begin 
	problem = ODEProblem(NetworkDiffusion, eltype(p).(u0), (0.0,1.0), ρ);
	sol = solve(problem, Tsit5(), saveat=0.005);
	data = Array(sol);
end

# ╔═╡ a130f152-64bb-11eb-17a1-152c296d98db
plot(sol)

# ╔═╡ edf0ca08-64bb-11eb-0637-bf759d891583
begin 
	Turing.setadbackend(:forwarddiff)
	@model function fit(data, func)
		σ ~ InverseGamma(2, 3) # ~ is the tilde character
		ρ ~ truncated(Normal(5,10.0),0.0,10)

		#prob = remake(problem, p=p)
		prob = ODEProblem(func,eltype(ρ).(u0),(0.0,1.0),ρ)
		predicted = solve(prob, Tsit5(),saveat=0.005)

		for i = 1:length(predicted)
			data[:,i] ~ MvNormal(predicted[i], σ)
		end
	end
end

# ╔═╡ ea970ade-64bb-11eb-3975-6b73229f2779
begin
	model = fit(data, NetworkDiffusion)
	advi = ADVI(10, 1000)
	q = vi(model, advi)
	samples = rand(q, 10000);
end

# ╔═╡ 3cf0f646-64bc-11eb-0a91-974808128e23
histogram(samples[2,:])

# ╔═╡ 61a86fee-64be-11eb-0d9f-1d57f870b70b
histogram(samples[1,:])

# ╔═╡ a312fd94-656f-11eb-1a0d-45c5a48872ea
q.dist.m

# ╔═╡ Cell order:
# ╠═105b4cf4-64bb-11eb-07ef-2127bbaa69d9
# ╠═474bf842-64bb-11eb-3678-93fd8b1e9aa5
# ╠═79a3153e-64c0-11eb-126f-bff2b909f67c
# ╠═2b35a7c4-64be-11eb-2785-03c7cab059fe
# ╠═5c626174-64bf-11eb-22b0-b3509b9899e3
# ╠═401c3fcc-64be-11eb-3b48-cfb9749db66a
# ╠═5bf1ef5e-64bb-11eb-297a-c971721eda48
# ╠═6ab4cec8-64bb-11eb-0cb5-9583308fd650
# ╠═086dcd3e-64be-11eb-0b13-c7371c6b367f
# ╠═6fca1080-64bb-11eb-37c0-77b51e264c31
# ╠═a130f152-64bb-11eb-17a1-152c296d98db
# ╠═ff1da1fc-64bb-11eb-12fc-9f7519373a3d
# ╠═edf0ca08-64bb-11eb-0637-bf759d891583
# ╠═ab19d95e-64bb-11eb-1613-576a2508b36f
# ╠═ea970ade-64bb-11eb-3975-6b73229f2779
# ╠═3cf0f646-64bc-11eb-0a91-974808128e23
# ╠═61a86fee-64be-11eb-0d9f-1d57f870b70b
# ╠═a312fd94-656f-11eb-1a0d-45c5a48872ea
