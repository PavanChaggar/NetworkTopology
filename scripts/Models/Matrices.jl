using SimpleWeightedGraphs
using LightGraphs
using SparseArrays
using Statistics
using DelimitedFiles
import LightGraphs.LinAlg.laplacian_matrix

"""
    MakeSimpleWeightedGraph(n, p)

Create a simple weighted random graph of `n` nodes with connection probability `p`.
"""
function MakeSimpleWeightedGraph(n::Int64, p::Float64)
    GW = SimpleWeightedGraph(erdos_renyi(n, p))
    for e in edges(GW)
        add_edge!(GW, src(e), dst(e), rand())
    end
    return GW
end

read_subjects(csv_path::String) = Int.(readdlm(csv_path))
	
symmetrise(M) = 0.5 * (M + transpose(M))

max_norm(M) = M ./ maximum(M)

adjacency_matrix(file::String) = sparse(readdlm(file))

laplacian_matrix(A::Array{Float64,2}) = SimpleWeightedGraphs.laplacian_matrix(SimpleWeightedGraph(A))

mean_connectome(M) = mean(M, dims=3)[:,:]

filter(A, cutoff) = A .* (A .> cutoff)

function load_connectome(subjects, subject_dir, N, size, length)
    
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

#=
laplacian_matrix(A::Array{Float64,2}) = laplacian_matrix(SimpleWeightedGraph(A))

connectome_dir = "/home/chaggar/Projects/Connectomes/"
csv_path = connectome_dir * "all_subjects"
subject_dir = connectome_dir * "standard_connectome/scale1/subjects/"
subjects = read_subjects(csv_path);
A_all = load_connectome(subjects, subject_dir, 100, 83, false);
A = max_norm(mean_connectome(A_all))

laplacian_matrix(SimpleWeightedGraph(A))

laplacian_matrix(A)
=#