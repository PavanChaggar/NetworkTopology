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

# ╔═╡ b05c3842-a1a5-434d-9376-7f070d246f21
begin
	using Turing
	using Plots
	using PlutoUI
	using Distributions
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

# ╔═╡ 80039d50-60fd-421d-a55e-9408c9403c4d
MvNormal(data[:,1], 0)

# ╔═╡ a4d096d0-55d9-4e3d-8155-aee1dcdf01a6
@model function fit(data)
	T, N = size(data)
	
	σ ~ InverseGamma(2, 3)
	
	Am ~ Normal(0,10)
	As ~ truncated(Normal(0,5), 0, Inf)
	
	Λm ~ Normal(0,2)
	Λs ~ truncated(Normal(0,5), 0, Inf)
	
	for i ∈ 1:N
		α ~ truncated(Normal(Am, As), 0, Inf)
		λ ~ truncated(Normal(Λm, Λs), 0, Inf)
		predicted = f(α, λ)
		data[:,i] .~ MvNormal(predicted, σ)
	end
end	

# ╔═╡ cc394908-66cf-4500-8666-6f64df9102e3
chain = sample(fit(data), NUTS(0.65), 1000)

# ╔═╡ b3b8e1b2-67d7-4f65-9590-b44e230b2a12
describe(chain)

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
# ╠═80039d50-60fd-421d-a55e-9408c9403c4d
# ╠═a4d096d0-55d9-4e3d-8155-aee1dcdf01a6
# ╠═cc394908-66cf-4500-8666-6f64df9102e3
# ╠═b3b8e1b2-67d7-4f65-9590-b44e230b2a12
