using Plots, MCMCChains
using DelimitedFiles

function plot_predictive(chain_array, prob, sol, data, node::Int)
    plot(Array(sol)[node,:], w=2, legend = false)
    for k in 1:300
        par = chain_array[rand(1:1_000), 1:13]
        resol = solve(remake(prob,u0=par[4:13], p=[par[3],par[1],par[2]]),Tsit5(),saveat=0.1)
        plot!(Array(resol)[node,:], alpha=0.5, color = "#BBBBBB", legend = false)
    end
    scatter!(data[node,:], legend = false)
end

function plot_connectome(coords::Matrix{Float64}, size::Float64, alpha::Float64)
    x, y, z = coords[:,1], coords[:,2], coords[:,3]
    scatter(x, y, z, grid=false, showaxis=false, markersize=size, markeralpha=alpha)
end

function plot_connectome(coords::Matrix{Float64}, A::Matrix{Float64}, size::Float64, alpha::Float64)
    fig = plot_connectome(coords, size, alpha)
	
    x, y, z = coords[:,1], coords[:,2], coords[:,3]
    coordindex = findall(x->x>0, A)
	
    for i ∈ 1:length(coordindex)
        j, k = coordindex[i][1], coordindex[i][2]
        plot!(fig, x[[j,k]], y[[j,k]], z[[j,k]], primary=false)
    end
	
    display(fig)
end
#=
using DelimitedFiles
connectome_dir = "/home/chaggar/Projects/Connectomes/"
coord_path = connectome_dir * "standard_connectome/parcellation/parcellation-files/sub-01_label-L2018_desc-scale1_atlas_coordinates.csv"
coords = readdlm(coord_path);

using Plots
plotly()

@code_warntype plot_connectome(coords, 3.0, 1.0)
plot_connectome(coords, 3.0, 1.0)

csv_path = connectome_dir * "all_subjects"
subject_dir = connectome_dir * "standard_connectome/scale1/subjects/"

include("Matrices.jl")
subjects = read_subjects(csv_path);
A = load_connectome(subjects, subject_dir, 100, 83, false)
filteredA = filter(max_norm(mean_connectome(A)), 0.05)

using SparseArrays

plot_connectome(coords, filteredA, 3.0, 1.0)
=#