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
const t = 0:1:12

# ╔═╡ e9d4029b-c31e-4cda-b34a-d0781b26f45e
f(a, λ) = a*exp.(-λ*t)

# ╔═╡ 5e6f7920-856e-4de3-acbc-780d1760d8af
data = Array{Float64}(undef, 13, 10);

# ╔═╡ f9f3fbbb-5390-4162-8678-ef534e1ba809
rand(Normal(2,1))

# ╔═╡ df906759-7510-48bf-b1f6-f843799085d8
Λ = rand(Normal(2,1), 5)

# ╔═╡ da2af64c-77ba-4fd0-86d1-b925fe147d78
for i ∈ 1:10
	a = rand(Normal(10,1))
	λ = rand(Normal(2,1))
	data[:,i] .= f(a, λ)
end

# ╔═╡ f6a3b18d-b688-4c28-b951-88208a726ba7
data

# ╔═╡ 3136c0fe-7768-43fc-a3ba-03e10f7cd0f3
@bind i Slider(1:10)

# ╔═╡ 56dc877e-aba8-4f5f-b4d1-64991c2baa1b
plot(data[:,i])

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

# ╔═╡ 0e001e76-7be0-4a90-bd37-8cc987d0f7e4


# ╔═╡ 9b8fa843-1276-41c6-8e90-2d0718ec986b
sample(fit(data), Prior(), 1)

# ╔═╡ cc394908-66cf-4500-8666-6f64df9102e3
chain = sample(fit(data), NUTS(0.65), 2000)

# ╔═╡ e893f8fe-4c2a-43b1-9d1e-794fc6257a00
plot(chain)

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
# ╠═f6a3b18d-b688-4c28-b951-88208a726ba7
# ╠═3136c0fe-7768-43fc-a3ba-03e10f7cd0f3
# ╠═56dc877e-aba8-4f5f-b4d1-64991c2baa1b
# ╠═a4d096d0-55d9-4e3d-8155-aee1dcdf01a6
# ╠═0e001e76-7be0-4a90-bd37-8cc987d0f7e4
# ╠═9b8fa843-1276-41c6-8e90-2d0718ec986b
# ╠═cc394908-66cf-4500-8666-6f64df9102e3
# ╠═e893f8fe-4c2a-43b1-9d1e-794fc6257a00
