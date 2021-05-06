using GLMakie
using DelimitedFiles
using FileIO
include("/home/chaggar/Projects/NetworkTopology/scripts/Models/Matrices.jl")

coord_file = "/home/chaggar/Projects/Connectomes/standard_connectome/parcellation/parcellation-files/sub-01_label-L2018_desc-scale1_atlas_coordinates.csv"

coords = readdlm(coord_file)

x, y, z, = coords[:,1],coords[:,2],coords[:,3]

fspath = "/home/chaggar/Projects/NetworkTopology/Notebooks/connectomes/cortical.obj"

fsbrain = load(fspath)

fig = mesh(fsbrain, color=(:grey, 0.1), transparency=true, show_axis=false)

meshscatter!(x, y, z, markersize=3, color=(:black,0.5))

csv_path = "/home/chaggar/Projects/Connectomes/all_subjects"
subjects_dir = "/home/chaggar/Projects/Connectomes/standard_connectome/scale1/subjects/"

subjects = read_subjects(csv_path)

An = mean_connectome(load_connectome(subjects, subjects_dir, 100, 83, false))

Anorm = max_norm(An)

A = filter(Anorm, 0.1)

coordindex = findall(x->x>0, A)

for i ∈ 1:length(coordindex)
    j, k = coordindex[i][1], coordindex[i][2]
    lines!(x[[j,k]], y[[j,k]], z[[j,k]], primary=false)
end

save("fig.png", fig)


