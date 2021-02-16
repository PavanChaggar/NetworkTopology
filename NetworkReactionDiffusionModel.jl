### A Pluto.jl notebook ###
# v0.11.9

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

# ╔═╡ 3c0cf6a8-6476-11eb-09d1-999ee7b2b2a7
using DifferentialEquations, LightGraphs, PlutoUI, Random, Plots, StatsPlots

# ╔═╡ c80d981e-6483-11eb-0fe1-c177d735e7c1
using Turing, Distributions, MCMCChains

# ╔═╡ af2bd910-6476-11eb-370f-b3fa8bbbe1ea
md"""

# Reaction-Diffusion Models on Network

This notebook demonstrates how to use Julia to construct ODE models for the simulation of reaction-diffusion processes on graphs. 

In particular, we will focus on a pure diffusion model and a FKPP model. 

"""

# ╔═╡ 72047f28-64a6-11eb-22e5-a9110360f74f
Random.seed!(1)

# ╔═╡ f5e54040-6481-11eb-01b0-9d79e87d998f
md"""
### Generating a graph
We begin by using a random Erdos-Renyi graph with only five nodes. We then obtain the Laplacian matrix for this graph, which will be needed for the ODE.  
"""

# ╔═╡ a79abb08-6476-11eb-05cb-8f91beb6062b
G = erdos_renyi(5,1.0)

# ╔═╡ 41c66bc0-647a-11eb-3d82-839abf75159d
L = laplacian_matrix(G);

# ╔═╡ 57fc2dd8-647a-11eb-0ee1-1f8c31cd939c
NetworkDiffusion(u, p, t) = -p * L * u

# ╔═╡ da3b96c8-647a-11eb-067d-4d3b966a912a
# intial concentration for protein set as a 1,5 vector with random entries between 0,1
u0 = [0.9,0.1,0.1,0.1,0.1];

# ╔═╡ 25a78c46-64b7-11eb-20b9-f3f7946a71fd
@bind p Slider(0.0:0.25:5.0, default=1.0, show_value=true)

# ╔═╡ 20fc52a2-6483-11eb-26c1-695ca621c6b9
problem = ODEProblem(NetworkDiffusion, eltype(p).(u0), (0.0,2.0), p);

# ╔═╡ 3d071e36-647c-11eb-1cf6-f5556a8e8f6a
sol = solve(problem, Tsit5(), saveat=0.1);

# ╔═╡ 3d174390-649a-11eb-1d75-2ffc2dcb1a21
data = Array(sol)

# ╔═╡ b6dd655a-647d-11eb-2c2e-71024baf1dda
plot(sol)

# ╔═╡ 5c0782c2-6483-11eb-302a-576fad5f79cc
begin 
	Turing.setadbackend(:forwarddiff)
	@model function fit(data, func)
		σ ~ InverseGamma(2, 3) # ~ is the tilde character
		p ~ truncated(Normal(1.25,1.0),0.0,2.5)

		#prob = remake(problem, p=p)
		prob = ODEProblem(func,eltype(p).(u0),(0.0,2.0),p)
		predicted = solve(prob, Tsit5(),saveat=0.1)

		for i = 1:length(predicted)
			data[:,i] ~ MvNormal(predicted[i], σ)
		end
	end
end

# ╔═╡ d921d762-6483-11eb-2d46-275d428e13ca
model = fit(data, NetworkDiffusion);

# ╔═╡ ee2a090e-6483-11eb-2de9-05fac790ac70
chain = sample(model, NUTS(0.65), 1000)

# ╔═╡ 802f0488-64ba-11eb-21ea-2d3fdefb438e
plot(chain)

# ╔═╡ Cell order:
# ╟─af2bd910-6476-11eb-370f-b3fa8bbbe1ea
# ╠═3c0cf6a8-6476-11eb-09d1-999ee7b2b2a7
# ╠═72047f28-64a6-11eb-22e5-a9110360f74f
# ╟─f5e54040-6481-11eb-01b0-9d79e87d998f
# ╠═a79abb08-6476-11eb-05cb-8f91beb6062b
# ╠═41c66bc0-647a-11eb-3d82-839abf75159d
# ╠═57fc2dd8-647a-11eb-0ee1-1f8c31cd939c
# ╠═da3b96c8-647a-11eb-067d-4d3b966a912a
# ╠═20fc52a2-6483-11eb-26c1-695ca621c6b9
# ╠═3d071e36-647c-11eb-1cf6-f5556a8e8f6a
# ╠═3d174390-649a-11eb-1d75-2ffc2dcb1a21
# ╠═b6dd655a-647d-11eb-2c2e-71024baf1dda
# ╠═c80d981e-6483-11eb-0fe1-c177d735e7c1
# ╠═5c0782c2-6483-11eb-302a-576fad5f79cc
# ╠═d921d762-6483-11eb-2d46-275d428e13ca
# ╠═ee2a090e-6483-11eb-2de9-05fac790ac70
# ╠═25a78c46-64b7-11eb-20b9-f3f7946a71fd
# ╠═802f0488-64ba-11eb-21ea-2d3fdefb438e
