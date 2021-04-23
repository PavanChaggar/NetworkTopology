### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ e60fc192-7b75-11eb-35dd-5bcbcfddfc10
using Pkg

# ╔═╡ 001df892-7b76-11eb-0166-8b6b83569627
using Random, DifferentialEquations, Turing, Plots, StatsPlots, MCMCChains, LightGraphs, Base.Threads

# ╔═╡ d7ded858-7c1a-11eb-3067-195eabce4155
md"# Network FKPP

In this notebook, I will test using `Turing` to perform inference on NetworkFKPP model on networks of varying size."

# ╔═╡ fc9d5bae-7b75-11eb-1716-8f6bf3f8e7c8
Pkg.status()

# ╔═╡ f261c50c-7b76-11eb-2717-4f76b90a8850
Turing.setadbackend(:forwarddiff);

# ╔═╡ 54369d46-7b77-11eb-08f8-77ef8913750c
Random.seed!(1);

# ╔═╡ 264a5cb8-7c1b-11eb-2cba-ff82534e9fde
md"## Building the Network and Model

As with the diffusion model, the first step will be to construct a random network using `LightGraphs`. 

After this, we can define the Network FKPP equation: 

$\frac{d\mathbf{p}_i}{dt} = -k \sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i\left(1-\mathbf{p}_i\right)$"

# ╔═╡ 6d048a7e-7b77-11eb-1c4e-f5e4c4157768
function make_graph(N::Int64, P::Float64)
    G = erdos_renyi(N, P)
    L = laplacian_matrix(G)
    A = adjacency_matrix(G)
    return L, A
end

# ╔═╡ 7b141a62-7b77-11eb-328a-df5196d6b66b
N = 5

# ╔═╡ 7fba4e3a-7b77-11eb-2076-2d5f46b2146f
P = 0.5

# ╔═╡ 83c38ca6-7b77-11eb-2eda-c5d97d2592f4
L, A = make_graph(N, P);

# ╔═╡ adfccf2c-7c1b-11eb-0704-d1c048c26592
md"Note that here we use element wise operations to define the ODE model"

# ╔═╡ 8afbbfc0-7b77-11eb-08fb-1d290de1d63e
function NetworkFKPP(u, p, t)
    κ, α = p 
    du = -κ * L * u .+ α .* u .* (1 .- u)
end

# ╔═╡ cb745fb6-7c1b-11eb-2aca-d3b2324c85f5
md"## Solving the model for N = 5 

We can now solve the model for the network defined with five nodes. We do this using `DifferentialEquations`, using an adaptive step size numerical method."

# ╔═╡ bc0ebb44-7b77-11eb-3331-e93127758c21
u0 = rand(N)

# ╔═╡ cac8fb22-7b77-11eb-0864-7df28ddc7bd5
p = 1.5, 3

# ╔═╡ ce6d3e64-7b77-11eb-3c4a-7f4c1d530eb3
t_span = (0.0, 2.0)

# ╔═╡ d5b28b66-7b77-11eb-1116-0560265f1d36
problem = ODEProblem(NetworkFKPP, u0, t_span, p);

# ╔═╡ ec0e2e7c-7b77-11eb-2cc8-1d8d7e207e33
sol = solve(problem, AutoTsit5(Rosenbrock23()), saveat=0.05);

# ╔═╡ f6aaaed4-7b77-11eb-3cdb-7130b9f4293d
data = Array(sol) + 0.02 * randn(size(Array(sol)));

# ╔═╡ 60c0e552-7b78-11eb-058b-c1edd8391b59
begin
	plot(Array(sol)', ylims=(0,1));
	scatter!(data')
end

# ╔═╡ a433eba6-7b78-11eb-2004-37cb179464a4
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

# ╔═╡ a1519422-7b78-11eb-1703-995012dc53e0
model = fit(data, problem);

# ╔═╡ 9557fd98-7b78-11eb-0820-07d98a284291
chain = sample(model, NUTS(.65), MCMCThreads(), 1_000, 10);

# ╔═╡ 8474fdbe-7b78-11eb-2657-333894b5bd08
#corner(chain, [:a, :k])

# ╔═╡ fa390dfa-7b79-11eb-322f-4d5f8a6b62d8
begin
	scatter(vcat(p[2],p[1],data[:,1],0))
	scatter!(mean(Array(chain), dims=1)')
end

# ╔═╡ Cell order:
# ╟─d7ded858-7c1a-11eb-3067-195eabce4155
# ╠═e60fc192-7b75-11eb-35dd-5bcbcfddfc10
# ╠═fc9d5bae-7b75-11eb-1716-8f6bf3f8e7c8
# ╠═001df892-7b76-11eb-0166-8b6b83569627
# ╠═f261c50c-7b76-11eb-2717-4f76b90a8850
# ╠═54369d46-7b77-11eb-08f8-77ef8913750c
# ╟─264a5cb8-7c1b-11eb-2cba-ff82534e9fde
# ╠═6d048a7e-7b77-11eb-1c4e-f5e4c4157768
# ╠═7b141a62-7b77-11eb-328a-df5196d6b66b
# ╠═7fba4e3a-7b77-11eb-2076-2d5f46b2146f
# ╠═83c38ca6-7b77-11eb-2eda-c5d97d2592f4
# ╟─adfccf2c-7c1b-11eb-0704-d1c048c26592
# ╠═8afbbfc0-7b77-11eb-08fb-1d290de1d63e
# ╟─cb745fb6-7c1b-11eb-2aca-d3b2324c85f5
# ╠═bc0ebb44-7b77-11eb-3331-e93127758c21
# ╠═cac8fb22-7b77-11eb-0864-7df28ddc7bd5
# ╠═ce6d3e64-7b77-11eb-3c4a-7f4c1d530eb3
# ╠═d5b28b66-7b77-11eb-1116-0560265f1d36
# ╠═ec0e2e7c-7b77-11eb-2cc8-1d8d7e207e33
# ╠═f6aaaed4-7b77-11eb-3cdb-7130b9f4293d
# ╟─60c0e552-7b78-11eb-058b-c1edd8391b59
# ╠═a433eba6-7b78-11eb-2004-37cb179464a4
# ╠═a1519422-7b78-11eb-1703-995012dc53e0
# ╠═9557fd98-7b78-11eb-0820-07d98a284291
# ╠═8474fdbe-7b78-11eb-2657-333894b5bd08
# ╠═fa390dfa-7b79-11eb-322f-4d5f8a6b62d8
