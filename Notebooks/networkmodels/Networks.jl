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

# ╔═╡ 6a18f3e0-8110-11eb-360e-d998359b8bd9
begin
	using DelimitedFiles
	using SparseArrays
	using Plots
	using Statistics
	using PlutoUI
	using LightGraphs
	using SimpleWeightedGraphs
	gr()
end;

# ╔═╡ 15579ad7-dec6-4270-97b6-2b54e0d86dec
md"# Brief Intro 

Hi! 

I'm currently a DPhil student on the SABS CDT programme working with Alain Goriely and Saad Jbabdi. 

The main thrust of my research programme is to use mathematical and statistical methods to investigate Alzheimer's pathology. At present, this involves combining simple (reaction-)diffusion equations on graphs, such as: 

$\frac{d\mathbf{p}_i}{dt} = -\rho \mathbf{L} \mathbf{p}$  

with probablistic inference tools such as Hamiltonian Monte Carlo (NUTS) and variational inference (ADVI). 

I'm going to present a little of my work so far and hopefully teach you something about the Julia progamming language along the way."

# ╔═╡ 5239e2d6-f43c-42fb-89df-1d2180bb449f
md"
# Looking at Connectomes

At the heart of our reaction-diffusion models are graphs representing white matter connections between regions, i.e. **connectomes**. 

I have been working on creating connectomes from HCP data to run simulations using our models. Using Julia and Pluto, I can look at and interact with the data and visualise how preprocessing steps change the outputs.

I'll go through a few steps: 
* Loading the packages we need
* Loading in csv and matrix data. 
* Creating adjacency and Laplacian matrices 
* Visualising data
"

# ╔═╡ 6f0789c5-6200-411e-b007-4b4e21a20d2a
md" 
## Using Packages 

Using packages is easy and similar to Python. Instead of `import`, in Julia we use `using`.

In practice, `using <pkg>` is the same as performing `from <pkg> import *`. It's a bad idea to do this in Python since function namescapes may overlap and cause functions to dispatch in unexpected ways. In Julia, this is not usually a problem since functions will dispatch on input types.
"

# ╔═╡ 723bda74-ae72-42f8-9464-bbc5d7ca73a3
md" 
## Loading Connectomes 

Let's start loading some connectomes! 
We'll start by defining some functions. I could do this in another .jl file and import them using `include(filename)`. 
"

# ╔═╡ 4e3ed2b9-1f6d-44cf-9062-cb4894b381b1
begin
	import LightGraphs.LinAlg.laplacian_matrix

	read_subjects(csv_path) = Int.(readdlm(csv_path))
	
	symmetrise(M) = 0.5 * (M + transpose(M))
	
	max_norm(M) = M ./ maximum(M)
	
	adjacency_matrix(file::String) = readdlm(file)
	
	laplacian_matrix(A::Array{Float64,2}) = laplacian_matrix(SimpleWeightedGraph(A))
	
	mean_connectome(M) = mean(M, dims=3)[:,:]
	
	filter(A, cutoff) = A .* (A .> cutoff)
	
	function load_connectome(subjects, subject_dir, N, size; length)
	    
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
	
	function diffusive_weight(An, Al)
	    A = An ./ Al.^2
	    [A[i,i] = 0.0 for i in 1:size(A)[1]]
	    return A
	end	
end;

# ╔═╡ 8a32d118-3ff6-4f47-bad0-6e4f303c9291
md"
#### Set some paths 
Let's set some paths and compose some string to set up reading in the necessary files.
"

# ╔═╡ 0defba63-25e4-47f7-ae8c-b2660535828c
begin 
	connectome_dir = "/home/chaggar/Projects/Connectomes/"
	csv_path = connectome_dir * "all_subjects"
	subject_dir = connectome_dir * "standard_connectome/scale1/subjects/"
end;

# ╔═╡ 0cd489b3-b52a-42f8-906a-2436ae7cf6cf
md"Load in the subject IDs by reading the csv file" 

# ╔═╡ 80447dd6-8114-11eb-32fa-4ff09d65bb38
subjects = read_subjects(csv_path);

# ╔═╡ 9cf974c5-1278-4b35-8f1f-1b04e8611214
md"
## Visualising Data

We can visualise the data using `Plots`. Plots supports a number of backends, including PyPlot, GR and Plotly. GR is the fastest among these and it's speed allows one to interact with data and easily visualise changes when variables are modified. 

For example, let's scroll through some subject data. 
"

# ╔═╡ 2f23c662-8116-11eb-27d3-999a69914383
#heatmap(max_norm(A_all[:,:,x]), title="Adjacency Matrix")

# ╔═╡ 59d7b012-8119-11eb-11ba-29aa721afd74
#heatmap(Matrix(laplacian_matrix(max_norm(A_all[:,:,x]))), title="Laplacian Matrix")

# ╔═╡ 96865a05-8a8d-4a75-96b5-984b18e205da
md"
##### N = $(@bind N Slider(0:5:100, show_value=true, default=10))
"

# ╔═╡ 0c84da7f-f465-4a9b-8535-56105c1ef20f
A_all = load_connectome(subjects, subject_dir, N, 83, length=false);

# ╔═╡ e66effc7-ba1a-49d1-b166-87c2f467807a
A_all

# ╔═╡ 7006a5fe-811c-11eb-3b19-a9e4e1c7dd5b
#begin
	A = max_norm(mean_connectome(A_all))
#	heatmap(A, title="Avg. Adjacency Matrix")
#end

# ╔═╡ 8821166c-8115-11eb-222e-51e91bc765ef
md"
##### x = $(@bind x Slider(1:N, show_value=true, default=1))
"

# ╔═╡ a629ac36-2471-45c6-9417-c72e8462cc66
md"
##### scale5node = $(@bind scale5node Slider(1:83, show_value=true, default=1))
"

# ╔═╡ 7d0b8dfe-818b-11eb-3114-1d5ec2c08288
md"
##### cutoff = $(@bind cutoff Slider(0:0.01:0.5, show_value=true, default=0.0))
"

# ╔═╡ 3edc21c0-c280-40cd-9511-e088b81620b0
mean_degree = mean(degree(Graph(filter(A, cutoff))));

# ╔═╡ d28e03c0-8121-11eb-31e3-69beeca2592f
md"
##### mean degree: $mean_degree
"

# ╔═╡ ad0c705a-f0c7-46a3-963f-8be4a424ab88
begin
	gr()
	bar(filter(A[scale5node,:], cutoff), ylim=(0,1))
	plot!([cutoff],  linetype=[:hline])
end

# ╔═╡ Cell order:
# ╟─15579ad7-dec6-4270-97b6-2b54e0d86dec
# ╟─5239e2d6-f43c-42fb-89df-1d2180bb449f
# ╟─6f0789c5-6200-411e-b007-4b4e21a20d2a
# ╠═6a18f3e0-8110-11eb-360e-d998359b8bd9
# ╟─723bda74-ae72-42f8-9464-bbc5d7ca73a3
# ╠═4e3ed2b9-1f6d-44cf-9062-cb4894b381b1
# ╟─8a32d118-3ff6-4f47-bad0-6e4f303c9291
# ╠═0defba63-25e4-47f7-ae8c-b2660535828c
# ╟─0cd489b3-b52a-42f8-906a-2436ae7cf6cf
# ╠═80447dd6-8114-11eb-32fa-4ff09d65bb38
# ╠═0c84da7f-f465-4a9b-8535-56105c1ef20f
# ╠═e66effc7-ba1a-49d1-b166-87c2f467807a
# ╟─9cf974c5-1278-4b35-8f1f-1b04e8611214
# ╟─8821166c-8115-11eb-222e-51e91bc765ef
# ╠═2f23c662-8116-11eb-27d3-999a69914383
# ╠═59d7b012-8119-11eb-11ba-29aa721afd74
# ╠═7006a5fe-811c-11eb-3b19-a9e4e1c7dd5b
# ╟─96865a05-8a8d-4a75-96b5-984b18e205da
# ╟─a629ac36-2471-45c6-9417-c72e8462cc66
# ╟─7d0b8dfe-818b-11eb-3114-1d5ec2c08288
# ╟─d28e03c0-8121-11eb-31e3-69beeca2592f
# ╟─3edc21c0-c280-40cd-9511-e088b81620b0
# ╟─ad0c705a-f0c7-46a3-963f-8be4a424ab88
