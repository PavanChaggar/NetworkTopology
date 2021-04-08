### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ e00f449f-7f45-4a4a-8f90-a31480880884
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
	Turing.setadbackend(:forwarddiff)
end;

# ╔═╡ 1228cf75-0b5a-472d-9c26-eb2bddf99628
using LinearAlgebra

# ╔═╡ a5e63c36-96c3-11eb-1549-7f91bb54dd73
begin
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

end

# ╔═╡ f7ac9405-7d5c-43f9-900b-b6610cea40b8
csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"

# ╔═╡ d5e279bd-94d4-462a-b00e-eaac7d95bec0
subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"

# ╔═╡ 4f59ea19-90b0-4336-841f-177d40cb6cb4
subjects = read_subjects(csv_path);

# ╔═╡ a02e80ea-f5ac-4369-952e-ba3975659c15
An = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, length=false))

# ╔═╡ 45677ced-58ea-4c46-a8fd-c42a4a76fc00
Al = mean_connectome(load_connectome(subjects, subject_dir, 100, 83, length=true))

# ╔═╡ 99b65910-5069-4011-a2df-d93ac2e4961b
A = diffusive_weight(An, Al)

# ╔═╡ ae17b8a2-abed-41f2-a682-cd4b9c60dbb8
Array(max_norm(laplacian_matrix(A)))

# ╔═╡ f550126c-d46b-429a-b618-dd117c13b259
M = load_connectome(subjects, subject_dir, 100, 83, length=false)

# ╔═╡ 0421e950-8f35-465f-b821-e2f07732e5a2
ran = reshape(rand(16), (4,4))

# ╔═╡ 87cde243-fd02-44ad-9980-23e42c8e158a
reshape(ran,(1,16))

# ╔═╡ bea3de9b-fcd6-4717-9f20-d9fb61521f9d
Mnew =  reshape(M, (83*83,100))

# ╔═╡ d1373efc-e2cf-41aa-a3d5-e4e1d8e1a032
U, Σ, Vt = svd(Mnew)

# ╔═╡ 66e8c1e3-bb3b-45f8-8790-f898aeaa0994
sn = 1

# ╔═╡ 07585e2e-189c-40ad-b4f3-1956f1e7b580
Msvd = reshape(U[:,sn] * Σ[sn] * Vt[sn,:]', (83,83,100))

# ╔═╡ ab6d0bc4-a743-4eea-870b-4aca55ee7edc
heatmap(mean_connectome(Msvd))

# ╔═╡ Cell order:
# ╠═e00f449f-7f45-4a4a-8f90-a31480880884
# ╟─a5e63c36-96c3-11eb-1549-7f91bb54dd73
# ╠═f7ac9405-7d5c-43f9-900b-b6610cea40b8
# ╠═d5e279bd-94d4-462a-b00e-eaac7d95bec0
# ╠═4f59ea19-90b0-4336-841f-177d40cb6cb4
# ╠═a02e80ea-f5ac-4369-952e-ba3975659c15
# ╠═45677ced-58ea-4c46-a8fd-c42a4a76fc00
# ╠═99b65910-5069-4011-a2df-d93ac2e4961b
# ╠═ae17b8a2-abed-41f2-a682-cd4b9c60dbb8
# ╠═f550126c-d46b-429a-b618-dd117c13b259
# ╠═0421e950-8f35-465f-b821-e2f07732e5a2
# ╠═87cde243-fd02-44ad-9980-23e42c8e158a
# ╠═bea3de9b-fcd6-4717-9f20-d9fb61521f9d
# ╠═1228cf75-0b5a-472d-9c26-eb2bddf99628
# ╠═d1373efc-e2cf-41aa-a3d5-e4e1d8e1a032
# ╠═07585e2e-189c-40ad-b4f3-1956f1e7b580
# ╠═66e8c1e3-bb3b-45f8-8790-f898aeaa0994
# ╠═ab6d0bc4-a743-4eea-870b-4aca55ee7edc
