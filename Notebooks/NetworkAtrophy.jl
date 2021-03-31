### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ dadc259e-9214-11eb-2b38-33b4cdb9e26b
begin
	using DelimitedFiles
	using SparseArrays
	using Statistics
	using PlutoUI
	using SimpleWeightedGraphs
	using LightGraphs
end;

# ╔═╡ 8bc35944-920c-11eb-1db3-052983e3b30b
include("/home/chaggar/Projects/NetworkTopology/scripts/Models/Models.jl");

# ╔═╡ b442e1b4-920c-11eb-0a77-411787683d43
md"# Network FKPP & Atrophy

Here, we will look at performing inference on a coupled network model of protein diffusion giveb by FKPP dynamics and strutural ROI atrophy given by logistic growth. 
Where as before we considered random networks, here we will consider networks produced using tractography."

# ╔═╡ 04360dc6-920e-11eb-3450-ff37ab783adc
md"## Load the Brain Network 

We will load the brain network from 100 subjects who have undergone tractography using FSL. Each subject contains two networks: number of connected streamlines and avergae length of streamlines. These are directed networks and both will be symmetrised as: 

$A_{sym} = \frac{1}{2}(A + A^{T})$

I will use weighted networks given by: 

$A_{i,j} = \frac{n_{i,j}}{l_{i,j}^2}$ 

However, options can be provided to switch the weighting of the network." 

# ╔═╡ 432a42e4-920f-11eb-1d98-87b08b70318d
begin
	# Functions to Load graph
	
	read_subjects(csv_path::String) = Int.(readdlm(csv_path))
	
	function norm_adjacency(file::String)
		A = sparse(readdlm(file))
		symA = 0.5 * (A + A')
		norm = symA/maximum(symA)
		return norm
	end
		
	function mean_connectome(subjects::Array, subject_dir::String, N::Int64, size::Int64; length::Bool)
		M = Array{Float64}(undef, size, size, N)
		
		if length == true
			connectome_type = "/fdt_network_matrix_lengths"
		else
			connectome_type = "/fdt_network_matrix"
		end
		
		for i ∈ 1:N
			file = subject_dir * string(subjects[i]) * connectome_type
			M[:,:,i] = norm_adjacency(file)
		end
		
		return mean(M, dims=3)[:,:]
	end
end

# ╔═╡ f92933dc-9211-11eb-091c-11c31d59df21
csv_path = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/all_subjects"

# ╔═╡ 6c77c6b6-9215-11eb-1289-5558d371050e
subject_dir = "/scratch/oxmbm-shared/Code-Repositories/Connectomes/standard_connectome/scale1/subjects/"

# ╔═╡ cbae4f5c-9214-11eb-0807-cd37295f37ba
subjects = read_subjects(csv_path);

# ╔═╡ f54ae6e0-9214-11eb-2baa-fba9babd502e
An = mean_connectome(subjects, subject_dir, 100, 83, length=false);

# ╔═╡ 1e560938-9216-11eb-15fa-4d0e02bff32d
Al = mean_connectome(subjects, subject_dir, 100, 83, length=true);

# ╔═╡ 75684bec-9215-11eb-2c7a-1be0d062b441
function diffusive_weight(An, Al)
	A = An ./ Al.^2
	[A[i,i] = 0 for i in 1:size(A)[1]]
	return A
end	

# ╔═╡ 7c9b1dfe-9217-11eb-068b-d17b2a1192aa
A = diffusive_weight(An, Al)

# ╔═╡ a79f01a8-9217-11eb-2466-79e51dd4fd98
L = laplacian_matrix(SimpleWeightedGraph(A));

# ╔═╡ 05e312f6-9216-11eb-2a63-31aa25bbdd62
md"Now we have an adjacency matrix and a Laplacian matrix!"

# ╔═╡ Cell order:
# ╟─b442e1b4-920c-11eb-0a77-411787683d43
# ╠═8bc35944-920c-11eb-1db3-052983e3b30b
# ╠═dadc259e-9214-11eb-2b38-33b4cdb9e26b
# ╟─04360dc6-920e-11eb-3450-ff37ab783adc
# ╠═432a42e4-920f-11eb-1d98-87b08b70318d
# ╠═f92933dc-9211-11eb-091c-11c31d59df21
# ╠═6c77c6b6-9215-11eb-1289-5558d371050e
# ╠═cbae4f5c-9214-11eb-0807-cd37295f37ba
# ╠═f54ae6e0-9214-11eb-2baa-fba9babd502e
# ╠═1e560938-9216-11eb-15fa-4d0e02bff32d
# ╠═75684bec-9215-11eb-2c7a-1be0d062b441
# ╠═7c9b1dfe-9217-11eb-068b-d17b2a1192aa
# ╠═a79f01a8-9217-11eb-2466-79e51dd4fd98
# ╟─05e312f6-9216-11eb-2a63-31aa25bbdd62
