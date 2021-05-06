using GLMakie
using DelimitedFiles
using FileIO
using SimpleWeightedGraphs
using LinearAlgebra
include("../../scripts/Models/Matrices.jl")

root_dir = "/Users/pavanchaggar/Projects/"

coord_file = root_dir * "Connectomes/standard_connectome/parcellation/parcellation-files/sub-01_label-L2018_desc-scale1_atlas_coordinates.csv"

coords = readdlm(coord_file)

x, y, z, = coords[:,1],coords[:,2],coords[:,3]

fspath = root_dir * "NetworkTopology/Notebooks/connectomes/cortical.obj"

fsbrain = load(fspath)

csv_path = root_dir * "Connectomes/all_subjects"
subjects_dir = root_dir * "Connectomes/standard_connectome/scale1/subjects/"

subjects = read_subjects(csv_path)

An = mean_connectome(load_connectome(subjects, subjects_dir, 100, 83, false))

Anorm = max_norm(An)

A = filter(Anorm, 0.1)

D = diag(degree_matrix(SimpleWeightedGraph(A)))

coordindex = findall(x->x>0, A)

fig = mesh(fsbrain, color=(:grey, 0.1), transparency=true, show_axis=false)

meshscatter!(x, y, z, markersize=Array(D)*2, color=(:blue,0.5))

for i ∈ 1:length(coordindex)
    j, k = coordindex[i][1], coordindex[i][2]
    lines!(x[[j,k]], y[[j,k]], z[[j,k]], primary=false)
end

label = readdlm("/Users/pavanchaggar/Projects/FSLabels/rh/rh.entorhinal.label")

brain = load(root_dir * "NetworkTopology/Notebooks/connectomes/cortical.stl")

mesh(
    brain,
    color = [tri[1][2] for tri in brain for i in 1:3],
)