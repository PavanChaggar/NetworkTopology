### A Pluto.jl notebook ###
# v0.14.3

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

# ╔═╡ b05c3842-a1a5-434d-9376-7f070d246f21
begin
	using Turing
	using Plots
	using PlutoUI
	using Distributions
	using BenchmarkTools
	using StatsPlots
	using MCMCChains
end

# ╔═╡ 7447efe8-a3a8-11eb-31ae-1f3d35622c05
md"# Test Hierarchical Model
 
Testing generating irregularly sampled data from a time-series process and fitting an hierarchical model using `Turing`."

# ╔═╡ 396775e3-0933-47dd-bc9d-3b9b7ff8a767
gr()

# ╔═╡ 34e8cb74-08ca-42c1-bfba-632bf44b82d0
const t = 1:1:12

# ╔═╡ e9d4029b-c31e-4cda-b34a-d0781b26f45e
f(a, λ) = a*exp.(-λ*t)

# ╔═╡ 5e6f7920-856e-4de3-acbc-780d1760d8af
data = Array{Float64}(undef, 12, 5)

# ╔═╡ f9f3fbbb-5390-4162-8678-ef534e1ba809
rand(Normal(2,1))

# ╔═╡ df906759-7510-48bf-b1f6-f843799085d8
Λ = rand(Normal(2,1), 5)

# ╔═╡ da2af64c-77ba-4fd0-86d1-b925fe147d78
for i ∈ 1:5
	a = rand(Normal(10,2))
	λ = rand(Normal(2,1))
	data[:,i] .= f(a, λ)
end

# ╔═╡ 3136c0fe-7768-43fc-a3ba-03e10f7cd0f3
@bind i Slider(1:5)

# ╔═╡ 56dc877e-aba8-4f5f-b4d1-64991c2baa1b
plot(t, data[:,i])

# ╔═╡ a4d096d0-55d9-4e3d-8155-aee1dcdf01a6
@model function fit(data)
	T, N = size(data)
	
	σ ~ InverseGamma(2, 3)
	
	Am ~ Normal(0,10)
	As ~ truncated(Normal(0,5), 0, Inf)
	
	Λm ~ Normal(0,2)
	Λs ~ truncated(Normal(0,5), 0, Inf)
	
	α ~ filldist(truncated(Normal(Am, As), 0, Inf), N)
	λ ~ filldist(truncated(Normal(Λm, Λs), 0, Inf), N)
	
	for i in 1:N
		predicted = f(α[i], λ[i])
		data[:,i] .~ MvNormal(predicted, σ)
	end
end	

# ╔═╡ 9176f8b1-108f-4cca-aabf-71362b31e0a6
@model function fit2(data)
	T, N = size(data)
	
	σ ~ InverseGamma(2, 3)
	
	Am ~ Normal(0,10)
	As ~ truncated(Normal(0,5), 0, Inf)
	
	Λm ~ Normal(0,2)
	Λs ~ truncated(Normal(0,5), 0, Inf)
	
	α ~ filldist(truncated(Normal(Am, As), 0, Inf), N)
	λ ~ filldist(truncated(Normal(Λm, Λs), 0, Inf), N)
	
	for i in 1:N
		predicted = f(α[i], λ[i])
		data[:,i] ~ MvNormal(predicted, σ)
	end
end	

# ╔═╡ 73b682e5-8d32-4f6a-bdee-2d3157b826dd
@model function fit3(data)
	T, N = size(data)
	
	σ ~ InverseGamma(2, 3)
	
	Am ~ Normal(0,10)
	As ~ truncated(Normal(0,5), 0, Inf)
	
	Λm ~ Normal(0,2)
	Λs ~ truncated(Normal(0,5), 0, Inf)
	
	α ~ filldist(truncated(Normal(Am, As), 0, Inf), N)
	λ ~ filldist(truncated(Normal(Λm, Λs), 0, Inf), N)
	
	for i in 1:N
		predicted = f(α[i], λ[i])
		Turing.@addlogprob! loglikelihood(MvNormal(predicted, σ), data[:,i])
	end
end


# ╔═╡ e963f64c-33e4-4cc6-b691-124af48884eb
Normal.(ones(10),0)

# ╔═╡ 7720f559-ea5e-463a-8c22-a8ebd1e56855
loglikelihood.(Normal.(data[:,1], 1), data[:,1])

# ╔═╡ 7911ef37-2eb2-4c42-8265-bb2122e5d1a9
loglikelihood(MvNormal(ones(10), 1), ones(10))

# ╔═╡ 9b8fa843-1276-41c6-8e90-2d0718ec986b
sample(fit(data), Prior(), 1)

# ╔═╡ cc394908-66cf-4500-8666-6f64df9102e3
chain = sample(fit(data), NUTS(0.65), 2000)

# ╔═╡ 2e886348-f80a-4871-a6cc-cb94cf231d25
chain2 = sample(fit2(data), NUTS(0.65), 2000)

# ╔═╡ b234d309-c076-4565-b694-9dd3ceee611b
chain3 = sample(fit3(data), NUTS(0.65), 2000)

# ╔═╡ e893f8fe-4c2a-43b1-9d1e-794fc6257a00
plot(chain)

# ╔═╡ 72b42320-ac94-4d37-8dd8-6288688555da
plot(chain2)

# ╔═╡ 3cc3b35c-a786-4b10-b39b-e28b5ca1581a
plot(chain3)

# ╔═╡ Cell order:
# ╟─7447efe8-a3a8-11eb-31ae-1f3d35622c05
# ╠═b05c3842-a1a5-434d-9376-7f070d246f21
# ╠═396775e3-0933-47dd-bc9d-3b9b7ff8a767
# ╠═e9d4029b-c31e-4cda-b34a-d0781b26f45e
# ╠═34e8cb74-08ca-42c1-bfba-632bf44b82d0
# ╠═5e6f7920-856e-4de3-acbc-780d1760d8af
# ╠═f9f3fbbb-5390-4162-8678-ef534e1ba809
# ╠═df906759-7510-48bf-b1f6-f843799085d8
# ╠═da2af64c-77ba-4fd0-86d1-b925fe147d78
# ╠═3136c0fe-7768-43fc-a3ba-03e10f7cd0f3
# ╠═56dc877e-aba8-4f5f-b4d1-64991c2baa1b
# ╠═a4d096d0-55d9-4e3d-8155-aee1dcdf01a6
# ╠═9176f8b1-108f-4cca-aabf-71362b31e0a6
# ╠═73b682e5-8d32-4f6a-bdee-2d3157b826dd
# ╠═e963f64c-33e4-4cc6-b691-124af48884eb
# ╠═7720f559-ea5e-463a-8c22-a8ebd1e56855
# ╠═7911ef37-2eb2-4c42-8265-bb2122e5d1a9
# ╠═9b8fa843-1276-41c6-8e90-2d0718ec986b
# ╠═cc394908-66cf-4500-8666-6f64df9102e3
# ╠═2e886348-f80a-4871-a6cc-cb94cf231d25
# ╠═b234d309-c076-4565-b694-9dd3ceee611b
# ╠═e893f8fe-4c2a-43b1-9d1e-794fc6257a00
# ╠═72b42320-ac94-4d37-8dd8-6288688555da
# ╠═3cc3b35c-a786-4b10-b39b-e28b5ca1581a
